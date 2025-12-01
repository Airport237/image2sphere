from src.speedplus_utils import kelvins_pose_score
from src.predictor import I2S
from src.so3_utils import rotation_error, nearest_rotmat
from src.pascal_dataset import Pascal3D
from src.dataset import ModelNet10Dataset, SPEEDPLUSDataset, SymsolDataset
from src.train import evaluate_speedplus_kelvins
import re
import argparse
import os
import time
from datetime import datetime
import logging
import numpy as np
import torch
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore', category=UserWarning)





def create_dataloaders(args):
    if args.dataset_name.find('modelnet10') > -1:
        train_set = ModelNet10Dataset(args.dataset_path,
                                      train=True,
                                      limited=(args.dataset_name.find('limited') > -1))
        test_set = ModelNet10Dataset(args.dataset_path,
                                     train=False)
    elif args.dataset_name.find('pascal3d') > -1:
        train_set = Pascal3D(args.dataset_path,
                             train=True,
                             use_warp=args.dataset_name.find('warp') > -1,
                             use_synth=args.dataset_name.find('synth') > -1,
                             )
        test_set = Pascal3D(args.dataset_path,
                            train=False)
    elif args.dataset_name.find('symsol') > -1:
        train_set = SymsolDataset(args.dataset_path,
                                  train=True,
                                  set_number=args.dataset_name.count('I'),
                                  num_views=int(re.findall(r'\d+', args.dataset_name)[0]))
        test_set = SymsolDataset(args.dataset_path,
                                 train=False,
                                 set_number=args.dataset_name.count('I'),
                                 num_views=5000)
    elif args.dataset_name == 'speed+':
        print("in speedplus")
        train_set = SPEEDPLUSDataset(
            root=args.dataset_path,
            split=args.speedplus_split,   # domain name
            train=True
        )
        print("aftter train")
        test_set = SPEEDPLUSDataset(
            root=args.dataset_path,
            split=args.speedplus_test,
            train=False
        )
    else:
        raise TypeError('Invalid dataset name')

    print(f'{len(train_set)} train imgs; {len(test_set)} test imgs')

    args.img_shape = train_set.img_shape
    args.num_classes = train_set.num_classes
    args.class_names = train_set.class_names

    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=args.num_workers,
                                               drop_last=True)

    test_loader = torch.utils.data.DataLoader(test_set,
                                              batch_size=args.batch_size,
                                              shuffle=True,
                                              num_workers=args.num_workers,
                                              drop_last=True)
    return train_loader, test_loader, args


def create_model(args):
    model = I2S(num_classes=args.num_classes,
                encoder=args.encoder,
                sphere_fdim=args.sphere_fdim,
                lmax=args.lmax,
                train_grid_rec_level=args.train_grid_rec_level,
                train_grid_n_points=args.train_grid_n_points,
                train_grid_include_gt=args.train_grid_include_gt,
                train_grid_mode=args.train_grid_mode,
                eval_grid_rec_level=args.eval_grid_rec_level,
                eval_use_gradient_ascent=args.eval_use_gradient_ascent,
                include_class_label=args.include_class_label,
                pred_translation=True,  # set them hardcoded for now
                trans_hidden=256,
                ).to(args.device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'num params: {num_params/1e6:.3f}M')

    model.train()
    return model



def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    print("starting main")
    if args.device != 'cpu':
        torch.cuda.manual_seed(args.seed)

    fname = f"{args.dataset_name}_{args.encoder.replace('_', '-')}_seed{args.seed}"
    if args.include_class_label:
        fname += "_cls-label"

    if args.desc != '':
        fname += f"_{args.desc}"
    args.fdir = os.path.join(args.results_dir, fname)
    print(args.fdir)

    if not os.path.exists(args.fdir):
        os.makedirs(args.fdir)

    with open(os.path.join(args.fdir, 'args.txt'), 'w') as f:
        f.write(str(args.__dict__))

    logger = logging.getLogger("test")
    logger.setLevel(logging.DEBUG)
    logger.handlers = [logging.StreamHandler(),
                       logging.FileHandler(os.path.join(args.fdir, "log.txt"))]

    train_loader, test_loader, args = create_dataloaders(args)

    print("post trainloader")
    model = create_model(args)
    print("post model creatiion")


    # resume if checkpoint exists (allow shape mismatches when you've changed the model)
    if os.path.exists(os.path.join(args.fdir, "checkpoint.pt")):
        checkpoint = torch.load(os.path.join(args.fdir, "checkpoint.pt"),
                                map_location=args.device)
        if checkpoint.get('done', False):
            print("Found completed checkpoint; exiting.")
            evaluate_speedplus_kelvins(args, model, test_loader)
            return
        else:
            print("NO COMPLETE CHECKPOINT TO EVALUATE")
    else:
        print("NO COMPLETE CHECKPOINT TO EVALUATE")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, default='results')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda')

    parser.add_argument('--desc', type=str, default='')
    parser.add_argument('--encoder', type=str, default='resnet50_pretrained')

    parser.add_argument('--lmax', type=int, default=6,
                        help='Maximum degree of harmonics to use in spherical convolution')
    parser.add_argument('--sphere_fdim', type=int, default=512,
                        help='Feature dimension projected onto sphere')

    parser.add_argument('--train_grid_rec_level', type=int, default=3)
    parser.add_argument('--train_grid_n_points', type=int, default=4096)
    parser.add_argument('--train_grid_include_gt', type=int, default=0)
    parser.add_argument('--train_grid_mode', type=str, default='healpix',
                        choices=['healpix', 'random'])
    parser.add_argument('--eval_grid_rec_level', type=int, default=5)
    parser.add_argument('--eval_use_gradient_ascent', type=int, default=0)
    parser.add_argument('--include_class_label', type=int, default=0)

    parser.add_argument('--num_epochs', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr_initial', type=float, default=0.001)
    parser.add_argument('--lr_step_size', type=int, default=15)
    parser.add_argument('--lr_decay_rate', type=float, default=0.1)
    parser.add_argument('--sgd_momentum', type=float, default=0.9)
    parser.add_argument('--use_nesterov', type=int, default=1)
    parser.add_argument('--weight_decay', type=float, default=0)

    parser.add_argument('--dataset_path', type=str, default='./datasets')
    parser.add_argument('--dataset_name', type=str, default='modelnet10',
                        choices=['modelnet10',          # modelnet10 with 100 training views per instance
                                 'modelnet10-limited',  # modelnet10 with 20 training views per instance
                                 'pascal3d-warp-synth',  # pascal3D with warping and synthetic data
                                 'symsolI-50000',  # 5 classes of symsolI with 50k training views each
                                 'symsolII-50000',  # symsol sphX with 50k training views each
                                 'symsolIII-50000',  # symsol cylO with 50k training views each
                                 # symsol tetX with 50k training views each# Stanford speed+ dataset (a small sample)
                                 'symsolIIII-50000',
                                 # Stanford speed+ dataset (a small sample)
                                 'speed+',
                                 ]
                        )

    parser.add_argument('--num_workers', type=int, default=4,
                        help='workers used by dataloader')

    parser.add_argument('--speedplus_split', type=str, default="synthetic")
    parser.add_argument('--speedplus_test', type=str, default="lightbox")
    args = parser.parse_args()

    start_time = datetime.now()
    main(args)
