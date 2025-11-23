#!/usr/bin/env python3
import os
import sys
import zipfile
import subprocess
import json

import cv2
import numpy as np
import tqdm
import argparse


# -------------------------------
# Helper functions
# -------------------------------

def unzip_dataset(zip_path: str, extract_path: str):
    """Unzip SPEED+ archive if provided."""
    if not zip_path:
        print("[INFO] No ZIP path provided, assuming dataset already extracted.")
        return

    if not os.path.isfile(zip_path):
        print(f"[ERROR] ZIP file not found: {zip_path}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(extract_path, exist_ok=True)
    print(f"[INFO] Unzipping {zip_path} -> {extract_path}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_path)
    print("[INFO] Unzip completed.")


def preprocess_speedplus(dataset_root: str):
    """
    Preprocess SPEED+ lightbox images:
      - read /lightbox/test.json
      - resize images to 768x512 grayscale
      - write CSV labels to /lightbox/labels/test.csv
    """
    datadir = os.path.join(dataset_root, "lightbox")
    jsonfile = os.path.join(datadir, "test.json")

    if not os.path.isfile(jsonfile):
        print(f"[ERROR] test.json not found at {jsonfile}", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] Reading labels from: {jsonfile}")
    with open(jsonfile, "r") as f:
        labels = json.load(f)  # list of dicts

    # Output dirs
    labels_dir = os.path.join(datadir, "labels")
    os.makedirs(labels_dir, exist_ok=True)
    csvfile = os.path.join(labels_dir, "test.csv")

    imagedir = os.path.join(datadir, "images_768x512_RGB")
    os.makedirs(imagedir, exist_ok=True)

    print(f"[INFO] Writing label CSV to: {csvfile}")
    print(f"[INFO] Writing resized images to: {imagedir}")

    csv = open(csvfile, "w")

    for idx in tqdm.tqdm(range(len(labels)), desc="Preprocessing images"):
        filename = labels[idx]["filename"]

        img_path = os.path.join(datadir, "images", filename)
        if not os.path.isfile(img_path):
            print(f"[WARN] Image file missing: {img_path} (skipping)")
            continue

        # Load image → grayscale → resize
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (768, 512))  # (width, height)
        cv2.imwrite(os.path.join(imagedir, filename), image)

        # Quaternion + translation
        q_vbs2tango = np.array(labels[idx]["q_vbs2tango_true"], dtype=np.float32)
        r_Vo2To_vbs = np.array(labels[idx]["r_Vo2To_vbs_true"], dtype=np.float32)

        row = [filename] + q_vbs2tango.tolist() + r_Vo2To_vbs.tolist()
        csv.write(", ".join(map(str, row)) + "\n")

    csv.close()
    print("[INFO] Preprocessing completed.")


def pip_install_requirements(repo_root: str):
    """Try to install requirements.txt (non-fatal if it fails)."""
    req_path = os.path.join(repo_root, "requirements.txt")
    if not os.path.isfile(req_path):
        print(f"[INFO] No requirements.txt found at {req_path}, skipping.")
        return

    print(f"[INFO] Installing requirements from {req_path} (errors will not stop script).")
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", req_path],
            check=False,
        )
    except Exception as e:
        print(f"[WARN] Failed to install from requirements.txt: {e}")


def pip_install_extra():
    """Install extra deps used in your snippet."""
    for pkg in ["e3nn", "healpy"]:
        print(f"[INFO] Installing {pkg}...")
        subprocess.run([sys.executable, "-m", "pip", "install", pkg], check=False)


def run_training(
    repo_root: str,
    dataset_root: str,
    results_dir: str,
    device: str,
    num_epochs: int,
    batch_size: int,
    num_workers: int,
    encoder: str,
    sphere_fdim: int,
    train_grid_n_points: int,
    train_grid_rec_level: int,
    eval_grid_rec_level: int,
):
    """Run src.train with low-memory SPEED+ config."""

    if not os.path.isdir(repo_root):
        print(f"[ERROR] Repo root not found: {repo_root}", file=sys.stderr)
        sys.exit(1)

    os.chdir(repo_root)
    print(f"[INFO] Changed working directory to: {os.getcwd()}")

    cmd = [
        sys.executable, "-m", "src.train",
        "--dataset_name", "speed+",
        "--dataset_path", dataset_root,
        "--results_dir", results_dir,
        "--device", device,
        "--num_epochs", str(num_epochs),
        "--batch_size", str(batch_size),
        "--num_workers", str(num_workers),
        "--encoder", encoder,
        "--sphere_fdim", str(sphere_fdim),
        "--train_grid_n_points", str(train_grid_n_points),
        "--train_grid_rec_level", str(train_grid_rec_level),
        "--eval_grid_rec_level", str(eval_grid_rec_level),
    ]

    print("[INFO] Running training command:")
    print("       " + " ".join(cmd))

    subprocess.run(cmd, check=True)
    print("[INFO] Training completed successfully.")


# -------------------------------
# Main
# -------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run image2sphere SPEED+ training (server version, no Colab)."
    )

    parser.add_argument(
        "--repo_root",
        type=str,
        default="/content/image2sphere",
        help="Path to the image2sphere repo root.",
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="/content/speedplus_data",
        help="Root directory where SPEED+ data will live.",
    )
    parser.add_argument(
        "--zip_path",
        type=str,
        default=None,
        help="Optional path to smallspeed.zip. If provided, will be unzipped to dataset_root.",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="/content/results_speedplus_lowmem",
        help="Directory to store training results/checkpoints.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for training: 'cpu' or 'cuda'.",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=1,
        help="Number of epochs.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Dataloader workers.",
    )
    parser.add_argument(
        "--sphere_fdim",
        type=int,
        default=512,
        help="sphere_fdim (feature dimension on sphere).",
    )
    parser.add_argument(
        "--train_grid_n_points",
        type=int,
        default=4096,
        help="Number of points for training grid.",
    )
    parser.add_argument(
        "--train_grid_rec_level",
        type=int,
        default=3,
        help="Recursion level for training grid.",
    )
    parser.add_argument(
        "--eval_grid_rec_level",
        type=int,
        default=5,
        help="Recursion level for eval grid.",
    )
    parser.add_argument(
        "--skip_deps",
        action="store_true",
        help="Skip pip installing requirements/e3nn/healpy.",
    )

    args = parser.parse_args()

    # 1. Unzip dataset if ZIP provided
    unzip_dataset(args.zip_path, args.dataset_root)

    # 2. Preprocess SPEED+ images/labels
    preprocess_speedplus(args.dataset_root)

    # 3. Install dependencies (optional)
    if not args.skip_deps:
        pip_install_requirements(args.repo_root)
        pip_install_extra()

    # 4. Run training
    run_training(
        repo_root=args.repo_root,
        dataset_root=args.dataset_root,
        results_dir=args.results_dir,
        device=args.device,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        encoder="resnet18_pretrained",
        sphere_fdim=args.sphere_fdim,
        train_grid_n_points=args.train_grid_n_points,
        train_grid_rec_level=args.train_grid_rec_level,
        eval_grid_rec_level=args.eval_grid_rec_level,
    )


if __name__ == "__main__":
    main()
