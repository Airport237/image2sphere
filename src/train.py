from src.speedplus_utils import kelvins_pose_score
from src.predictor import I2S
from src.so3_utils import rotation_error, nearest_rotmat
from src.pascal_dataset import Pascal3D
from src.dataset import ModelNet10Dataset, SPEEDPLUSDataset, SymsolDataset
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


def evaluate_ll(args, model, test_loader):
    # log likelihood estimation
    model.eval()
    lls = []
    clss = []
    for batch_idx, batch in enumerate(test_loader):
        batch = {k: v.to(args.device) for k, v in batch.items()}
        probs = model.compute_probabilities(batch['img'], batch['cls'])

        gt_rotmats = batch['rot'].cpu()
        gt_inds = nearest_rotmat(gt_rotmats, model.eval_rotmats)
        gt_probs = probs[torch.arange(gt_rotmats.size(0)), gt_inds]
        log_likelihood = torch.log(
            gt_probs * model.eval_rotmats.shape[0] / np.pi**2)

        lls.append(log_likelihood.numpy())
        clss.append(batch['cls'].squeeze().cpu().numpy())

    lls = np.concatenate(lls)
    clss = np.concatenate(clss)

    per_class_ll = {}
    for i in range(args.num_classes):
        mask = clss == i
        per_class_ll[args.class_names[i]] = lls[mask]

    np.save(os.path.join(args.fdir, f'eval_log_likelihood.npy'), per_class_ll)


def evaluate_error(args, model, test_loader):
    model.eval()
    errors = []
    clss = []

    for batch_idx, batch in enumerate(test_loader):
        batch = {k: v.to(args.device) for k, v in batch.items()}
        pred_rotmat = model.predict(batch['img'], batch['cls']).cpu()
        gt_rotmat = batch['rot'].cpu()
        err = rotation_error(pred_rotmat, gt_rotmat)

        errors.append(err.numpy())
        # make sure we always have a 1D array so concatenate works
        clss.append(batch['cls'].cpu().numpy().reshape(-1))

    errors = np.concatenate(errors)
    clss = np.concatenate(clss)

    per_class_err = {}
    for i in range(args.num_classes):
        mask = clss == i
        per_class_err[args.class_names[i]] = errors[mask]

    np.save(os.path.join(args.fdir, 'eval.npy'), per_class_err)


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

    logger = logging.getLogger("train")
    logger.setLevel(logging.DEBUG)
    logger.handlers = [logging.StreamHandler(),
                       logging.FileHandler(os.path.join(args.fdir, "log.txt"))]

    train_loader, test_loader, args = create_dataloaders(args)

    print("post trainloader")
    model = create_model(args)
    print("post model creatiion")

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr_initial,
        momentum=args.sgd_momentum,
        weight_decay=args.weight_decay,
        nesterov=bool(args.use_nesterov),
    )

    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        args.lr_step_size,
        args.lr_decay_rate
    )

    # resume if checkpoint exists (allow shape mismatches when you've changed the model)
    if os.path.exists(os.path.join(args.fdir, "checkpoint.pt")):
        checkpoint = torch.load(os.path.join(args.fdir, "checkpoint.pt"),
                                map_location=args.device)
        if checkpoint.get('done', False):
            print("Found completed checkpoint; exiting.")
            evaluate_speedplus_kelvins(args, model, test_loader)
            return

        starting_epoch = checkpoint['epoch'] + 1
        try:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            print("Loaded checkpoint with strict=False (ignoring mismatched layers).")
        except RuntimeError as e:
            print("Could not load checkpoint state_dict:", e)
            starting_epoch = 1

        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        except Exception as e:
            print("Could not load optimizer / scheduler state:", e)
        model.train()
    else:
        starting_epoch = 1

    data = []
    print("befor eepocs")
    for epoch in range(starting_epoch, args.num_epochs + 1):
        # --------------------
        # Train
        # --------------------
        train_loss = 0.0
        train_rot_errs = []      # per-sample rotation errors (radians)
        train_trans_losses = []  # scalar per-batch translation losses
        time_before_epoch = time.perf_counter()

        model.train()
        for batch_idx, batch in enumerate(train_loader):
            batch = {k: v.to(args.device) for k, v in batch.items()}
            loss, stats = model.compute_loss(**batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_rot_errs.append(stats["rot_acc"])
            train_trans_losses.append(stats["trans_loss"])

        # aggregate train stats
        train_loss /= (batch_idx + 1)
        train_rot_errs = np.concatenate(train_rot_errs)          # radians
        train_rot_err_deg_median = np.degrees(np.median(train_rot_errs))
        train_rot_err_deg_mean = np.degrees(np.mean(train_rot_errs))
        train_trans_losses = np.array(train_trans_losses)
        train_trans_loss_mean = float(train_trans_losses.mean())

        # --------------------
        # Eval
        # --------------------
        test_loss = 0.0
        test_rot_errs = []
        test_trans_losses = []
        test_cls = []

        model.eval()
        for batch_idx, batch in enumerate(test_loader):
            batch = {k: v.to(args.device) for k, v in batch.items()}
            with torch.no_grad():
                loss, stats = model.compute_loss(**batch)

            test_loss += loss.item()
            test_rot_errs.append(stats["rot_acc"])
            test_trans_losses.append(stats["trans_loss"])
            test_cls.append(batch['cls'].cpu().numpy().reshape(-1))

        test_loss /= (batch_idx + 1)
        test_rot_errs = np.concatenate(test_rot_errs)
        test_rot_err_deg_median = np.degrees(np.median(test_rot_errs))
        test_rot_err_deg_mean = np.degrees(np.mean(test_rot_errs))
        test_trans_losses = np.array(test_trans_losses)
        test_trans_loss_mean = float(test_trans_losses.mean())
        test_cls = np.concatenate(test_cls)

        # per-class rotation error stats (in degrees)
        per_class_err = {}
        for i, cls_name in enumerate(args.class_names):
            mask = test_cls == i
            if mask.any():
                per_class_err[cls_name] = f"{np.degrees(np.median(test_rot_errs[mask])):.1f}"
            else:
                per_class_err[cls_name] = "nan"

        logger.info(str(per_class_err))

        data.append(dict(
            epoch=epoch,
            time_elapsed=time.perf_counter() - time_before_epoch,
            train_loss=train_loss,
            test_loss=test_loss,
            train_acc_median=np.median(train_rot_errs),
            test_acc_median=np.median(test_rot_errs),
            lr=optimizer.param_groups[0]['lr'],
        ))
        lr_scheduler.step()

        # checkpointing
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler_state_dict': lr_scheduler.state_dict(),
            'done': False,
        }, os.path.join(args.fdir, "checkpoint.pt"))

        log_str = (
            f"Epoch {epoch}/{args.num_epochs} | "
            f"LOSS train={train_loss:.4f}, test={test_loss:.4f} | "
            f"ROT-ERR train={train_rot_err_deg_median:.1f}°, "
            f"test={test_rot_err_deg_median:.1f}° | "
            f"T-LOSS train={train_trans_loss_mean:.4f}, "
            f"test={test_trans_loss_mean:.4f} | "
            f"time={time.perf_counter() - time_before_epoch:.1f}s | "
            f"lr={lr_scheduler.get_last_lr()[0]:.1e}"
        )

        logger.info(log_str)
        time_before_epoch = time.perf_counter()

    # quick sanity check on a few predictions
    print("\n=== Debugging predictions on test set ===")
    debug_predictions(args, model, test_loader, n_samples=3, visualize=True)

    if args.dataset_name.find('symsol') > -1:
        evaluate_ll(args, model, test_loader)
    else:
        evaluate_error(args, model, test_loader)

    # if args.dataset_name == 'speed+':
    #     evaluate_speedplus_kelvins(args, model, test_loader)
    #     return

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'done': True,
    }, os.path.join(args.fdir, "checkpoint.pt"))


def evaluate_speedplus_kelvins(args, model, loader):
    """
    Compute Kelvins/SPEED+ pose score:
      - orientation score (deg with tolerance)
      - position score (normalized with tolerance)
      - combined pose score
    """
    model.eval()

    all_pose_scores = []
    all_orient_scores = []
    all_pos_scores = []

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(args.device) for k, v in batch.items()}
            img = batch['img']
            cls = batch['cls']
            R_gt = batch['rot']      # (B, 3, 3)
            t_gt = batch['trans']    # (B, 3)

            R_gt = R_gt.squeeze()
            t_gt = t_gt.squeeze()

            # print(f"R-gt: {R_gt}")
            # print(f"t-gt: {t_gt}")
            # rotation prediction (model.predict returns CPU tensors!)
            R_pred = model.predict(img, cls).to(
                img.device)  # Move to GPU if needed

            # translation prediction
            _, t_pred = model.forward(img, cls, return_translation=True)

            # ensure both are on same device as GT
            R_pred = R_pred.to(R_gt.device)
            t_pred = t_pred.to(t_gt.device)

            R_pred = R_pred.squeeze()
            t_pred = t_pred.squeeze()

            pose_score, s_orient, s_pos = kelvins_pose_score(
                t_pred, R_pred, t_gt, R_gt
            )

            pose_score = torch.as_tensor(pose_score,  device='cpu').reshape(-1)
            s_orient = torch.as_tensor(s_orient,    device='cpu').reshape(-1)
            s_pos = torch.as_tensor(s_pos,       device='cpu').reshape(-1)

            all_pose_scores.append(pose_score)
            all_orient_scores.append(s_orient)
            all_pos_scores.append(s_pos)

    all_pose_scores = np.array(all_pose_scores)
    all_orient_scores = np.array(all_orient_scores)
    all_pos_scores = np.array(all_pos_scores)

    print("\n=== Kelvins / SPEED+ scoring (lightbox) ===")
    print(f"Mean orientation score (deg):   {all_orient_scores.mean():.4f}")
    print(f"Mean position score (norm):     {all_pos_scores.mean():.6f}")
    print(f"Mean pose score (leaderboard):  {all_pose_scores.mean():.4f}")
    print("===========================================\n")

    return {
        "pose_score_mean": float((all_pose_scores.mean())),
        "orient_score_mean": float(all_orient_scores.mean()),
        "pos_score_mean": float(all_pos_scores.mean()),
    }


def debug_predictions(args, model, loader, n_samples=5, visualize=True):
    """Print a few GT vs predicted rotations/translations and optionally visualize."""
    model.eval()
    printed = 0

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(args.device) for k, v in batch.items()}

            # rotation prediction (same as evaluate_error)
            rot_pred = model.predict(batch['img'], batch['cls'])   # (B, 3, 3)
            rot_gt = batch['rot']                                # (B, 3, 3)

            rot_pred = rot_pred.to(args.device)
            rot_gt = rot_gt.to(args.device)

            # translation
            trans_gt = batch.get('trans', None)
            trans_pred = None
            try:
                _, trans_pred = model.forward(
                    batch['img'], batch['cls'], return_translation=True
                )
            except TypeError:
                pass

            # rotation error
            rot_err = rotation_error(rot_pred, rot_gt)     # radians
            rot_err_deg = torch.rad2deg(rot_err)

            bsz = rot_pred.size(0)
            for i in range(bsz):
                if printed >= n_samples:
                    return

                print(f"\nSample #{printed+1}")
                print(f"  rot_err = {rot_err_deg[i].item():.1f}°")

                if trans_gt is not None and trans_pred is not None:
                    print(
                        f"  GT trans:   {trans_gt[i].detach().cpu().numpy()}")
                    print(
                        f"  Pred trans: {trans_pred[i].detach().cpu().numpy()}")

                    if visualize:
                        save_path = f"debug_vis/sample_{printed+1}.png"

                        visualize_prediction(
                            img_tensor=batch['img'][i],
                            trans_pred=trans_pred[i],
                            trans_gt=trans_gt[i],
                            save_path=save_path,
                            title=f"Sample {printed+1}"
                        )

                        print(f"Saved visualization → {save_path}")

                printed += 1

        if printed == 0:
            print("No samples found in loader for debug_predictions.")


def visualize_prediction(img_tensor, trans_pred, trans_gt, save_path, title=None):
    """
    img_tensor: (3, H, W) torch tensor in [0,1]
    trans_pred: (3,) predicted xyz (meters)
    trans_gt:   (3,) ground-truth xyz (meters)
    save_path:  file path to save output PNG
    """

    img = img_tensor.detach().cpu().permute(1, 2, 0).numpy()
    H, W, _ = img.shape
    t_pred = trans_pred.detach().cpu().numpy()
    px, py, pz = t_pred

    t_gt = trans_gt.detach().cpu().numpy()
    gx, gy, gz = t_gt

    # === Fake projection (simple tanh normalization) ===
    u_pred = int((np.tanh(px) + 1) / 2 * W)
    v_pred = int((np.tanh(py) + 1) / 2 * H)

    u_gt = int((np.tanh(gx) + 1) / 2 * W)
    v_gt = int((np.tanh(gy) + 1) / 2 * H)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(4, 4))
    plt.imshow(img)
    plt.scatter([u_pred], [v_pred], c='red', s=60, label="Predicted")
    plt.scatter([u_gt], [v_gt], c='cyan', s=60, label="Ground Truth")
    plt.axis("off")

    legend_text = (
        f"Predicted:\n"
        f"x={px:.3f} m, y={py:.3f} m, z={pz:.3f} m\n\n"
        f"Ground Truth:\n"
        f"x={gx:.3f} m, y={gy:.3f} m, z={gz:.3f} m"
    )

    plt.text(
        5, 5, legend_text,
        fontsize=9,
        color="white",
        bbox=dict(facecolor="black", alpha=0.65, edgecolor="none"),
        verticalalignment="top"
    )

    if title:
        plt.title(title)

    # Save file
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


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
