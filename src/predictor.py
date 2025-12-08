import re
import e3nn
import torch
import time
from torch import Tensor
import numpy as np
import torch.nn as nn
from e3nn import o3

from e3nn.o3 import Irreps
from e3nn.nn import Activation

from src import so3_utils
from src.models import (
    SpatialS2Projector,
    HarmonicS2Projector,
    SpatialS2Features,
    HarmonicS2Features,
    ResNet,
    SO3Convolution,
)


class BaseSO3Predictor(nn.Module):
    def __init__(self,
                 num_classes: int = 1,
                 encoder: str = 'resnet18',
                 pool_features: bool = False,
                 **kwargs
                 ):
        super().__init__()
        self.num_classes = num_classes

        pretrained = encoder.find('pretrained') > -1
        size = int(re.findall('\d+', encoder)[0])
        self.encoder = ResNet(size, pretrained, pool_features)

    def save(self, path):
        torch.save(self.state_dict(), path)


class I2S(BaseSO3Predictor):
    def __init__(self,
                 num_classes: int = 1,
                 sphere_fdim: int = 512,
                 encoder: str = 'resnet50_pretrained',
                 projection_mode='spatialS2',
                 feature_sphere_mode='harmonicS2',
                 lmax: int = 6,
                 f_hidden: int = 8,
                 train_grid_rec_level: int = 3,
                 train_grid_n_points: int = 4096,
                 train_grid_include_gt: bool = False,
                 train_grid_mode: str = 'healpix',
                 eval_grid_rec_level: int = 5,
                 eval_use_gradient_ascent: bool = False,
                 include_class_label: bool = False,
                 pred_translation: bool = False,
                 trans_hidden: int = 256,
                 translation_head: str = 'mlp',
                 ):
        super().__init__(num_classes, encoder, pool_features=False)

        print("in model create")
        self.pred_translation = pred_translation
        self.register_buffer("trans_mean", torch.zeros(3))
        self.register_buffer("trans_std", torch.ones(3))
        self.lmax = lmax
        self.include_class_label = include_class_label

        proj_input_shape = list(self.encoder.output_shape)
        if self.include_class_label:
            proj_input_shape[0] += num_classes

        # translation head (from encoder features only)
        if self.translation_head == "mlp":
            c, h, w = self.encoder.output_shape
            self.translation_head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),   # (B, C, H, W) -> (B, C, 1, 1)
                nn.Flatten(),              # (B, C, 1, 1) -> (B, C)
                nn.Linear(c, trans_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(trans_hidden, 3),   # (x, y, z)
            )
        elif self.translation_head is "conv":
            c, h, w = self.encoder.output_shape
            self.translation_head = None
        else:
            self.translation_head = None

        # projection stuff
        self.projector = {
            'spatialS2': SpatialS2Projector,
            'harmonicS2': HarmonicS2Projector,
        }[projection_mode](proj_input_shape, sphere_fdim, lmax)

        # spherical conv stuff
        self.feature_sphere = {
            'spatialS2': SpatialS2Features,
            'harmonicS2': HarmonicS2Features,
        }[feature_sphere_mode](sphere_fdim, lmax, f_out=f_hidden)

        irreps_in = so3_utils.s2_irreps(lmax)
        self.o3_conv = o3.Linear(
            irreps_in,
            so3_utils.so3_irreps(lmax),
            f_in=sphere_fdim,
            f_out=f_hidden,
            internal_weights=False
        )

        self.so3_activation = e3nn.nn.SO3Activation(lmax, lmax, torch.relu, 10)
        so3_grid = so3_utils.so3_near_identity_grid()
        self.so3_conv = SO3Convolution(f_hidden, 1, lmax, so3_grid)

        # output rotations for training and evaluation
        self.train_grid_rec_level = train_grid_rec_level
        self.train_grid_n_points = train_grid_n_points
        self.train_grid_include_gt = train_grid_include_gt
        self.train_grid_mode = train_grid_mode
        self.eval_grid_rec_level = eval_grid_rec_level
        self.eval_use_gradient_ascent = eval_use_gradient_ascent

        output_xyx = so3_utils.so3_healpix_grid(rec_level=train_grid_rec_level)
        self.register_buffer(
            "output_wigners",
            so3_utils.flat_wigner(lmax, *output_xyx).transpose(0, 1)
        )
        self.register_buffer(
            "output_rotmats",
            o3.angles_to_matrix(*output_xyx)
        )

        output_xyx = so3_utils.so3_healpix_grid(rec_level=eval_grid_rec_level)
        try:
            self.eval_wigners = torch.load('eval_rec5.pt')
        except FileNotFoundError:
            self.eval_wigners = so3_utils.flat_wigner(
                lmax, *output_xyx).transpose(0, 1)

        self.eval_rotmats = o3.angles_to_matrix(*output_xyx)

    def forward(self, x, o, return_translation: bool = False):
        # encoder features
        feat = self.encoder(x)  # (B, C, H, W)

        # translation head from encoder features only
        trans_pred = None
        if self.pred_translation and self.translation_head is not None:
            trans_pred = self.translation_head(feat)  # (B, 3)

        # rotation path input
        proj_in = feat
        if self.include_class_label:
            o_oh = nn.functional.one_hot(
                o.squeeze(1), num_classes=self.num_classes
            )
            o_oh_fmap = o_oh.unsqueeze(-1).unsqueeze(-1).repeat(
                1, 1, feat.size(-2), feat.size(-1)
            )
            proj_in = torch.cat((proj_in, o_oh_fmap), dim=1)

        # projection to S2
        x = self.projector(proj_in)

        # S2 -> SO(3) lifting
        weight, _ = self.feature_sphere()
        x = self.o3_conv(x, weight=weight)

        x = self.so3_activation(x)

        # SO(3) convolution
        x = self.so3_conv(x)  # SO(3) Fourier coefficients

        if return_translation:
            return x, trans_pred

        return x

    def compute_loss(self, img, cls, rot, trans=None, lambda_trans: float = 1.0):
        """
        img:   (B, 3, H, W)
        cls:   (B, 1)
        rot:   ground-truth rotation (as expected by nearest_rotmat)
        trans: (B, 3) translation vector, if available
        """
        if self.pred_translation and trans is not None:
            x, trans_pred = self.forward(img, cls, return_translation=True)
        else:
            x = self.forward(img, cls)
            trans_pred = None

        # rotation classification loss
        grid_signal, rotmats = self.query_train_grid(x, rot)
        rot_id = so3_utils.nearest_rotmat(rot, rotmats)
        rot_loss = nn.CrossEntropyLoss()(grid_signal, rot_id)

        with torch.no_grad():
            pred_id = grid_signal.max(dim=1)[1]
            pred_rotmat = rotmats[pred_id]
            rot_acc = so3_utils.rotation_error(rot, pred_rotmat)  # (B,)

        if self.pred_translation and trans is not None:
            # forward gives normalized prediction
            x, trans_pred_norm = self.forward(
                img, cls, return_translation=True)

            # normalize GT using model's mean/std
            trans_norm = (trans - self.trans_mean) / self.trans_std

            trans_loss = nn.functional.mse_loss(trans_pred_norm, trans_norm)
            loss = rot_loss + lambda_trans * trans_loss
        else:
            trans_loss = torch.tensor(0.0, device=img.device)
            loss = rot_loss

        stats = {
            "rot_acc": rot_acc.cpu().numpy(),   # per-sample rotation error (radians)
            "rot_loss": rot_loss.item(),
            "trans_loss": trans_loss.item(),
        }

        return loss, stats

    def query_train_grid(self, x, gt_rot=None):
        '''x is signal over fourier basis'''
        if self.train_grid_mode == 'random':
            idx = torch.randint(len(self.output_rotmats),
                                (self.train_grid_n_points,))

            wigners = self.output_wigners[:, idx]
            rotmats = self.output_rotmats[idx]
            if self.train_grid_include_gt:
                # creating wigners is slightly faster on cpu
                try:
                    abg = o3.matrix_to_angles(gt_rot.cpu())
                    wigners[:, :gt_rot.size(0)] = so3_utils.flat_wigner(
                        self.lmax, *abg).transpose(0, 1).to(x.device)
                    rotmats[:gt_rot.size(0)] = gt_rot
                except AssertionError:
                    # sometimes dataloader generates invalid rot matrix according to o3
                    pass

        elif self.train_grid_mode == 'healpix':
            wigners = self.output_wigners
            rotmats = self.output_rotmats

        return torch.matmul(x, wigners).squeeze(1), rotmats

    def predict(self, x, o, lr=1e-3, n_iters=10):
        with torch.no_grad():
            fourier = self.forward(x, o)
            fourier = fourier.cpu()
            probs = torch.matmul(fourier, self.eval_wigners).squeeze(1)
            pred_id = probs.max(dim=1)[1]
            rots = self.eval_rotmats[pred_id]

        if self.eval_use_gradient_ascent:
            a, b, g = o3.matrix_to_angles(rots)
            a.requires_grad = True
            b.requires_grad = True
            g.requires_grad = True
            for _ in range(n_iters):
                wigners = so3_utils.flat_wigner(
                    self.lmax, a, b, g).transpose(0, 1)
                val = torch.diagonal(torch.matmul(fourier, wigners).squeeze(1))
                da, db, dg = torch.autograd.grad(val.mean(), (a, b, g))
                a = a + lr * da
                b = b + lr * db
                g = g + lr * dg
            rots = o3.angles_to_matrix(a, b, g).detach()

        return rots

    @torch.no_grad()
    def compute_probabilities(self, x, o):
        ''' compute probabilities over eval grid'''
        harmonics = self.forward(x, o)

        # move to cpu to avoid memory issues, at expense of speed
        harmonics = harmonics.cpu()

        probs = torch.matmul(harmonics, self.eval_wigners).squeeze(1)

        return nn.Softmax(dim=1)(probs)
