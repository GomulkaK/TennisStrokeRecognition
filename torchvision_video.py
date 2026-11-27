
import argparse
import csv
import os
import random
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.functional as TF

from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import RandomResizedCrop
from torchvision.transforms import InterpolationMode


# ============================================================
# PATHS AND CLASSES
# ============================================================

REPO_ROOT = Path(__file__).resolve().parent

CLASS_NAMES = [
    "Backhand",
    "Backhand with two hands",
    "Backhand slice",
    "Backhand volley",
    "Service flat",
    "Forehand flat",
    "Forehand open stands",
    "Forehand slice",
    "Forehand volley",
    "Service kick",
    "Service slice",
    "Smash",
]

NUM_CLASSES = len(CLASS_NAMES)

MODEL_CONFIGS = {
    "mvit_v1_b": {
        "factory": "mvit_v1_b",
        "weights_enum": "MViT_V1_B_Weights",
        "head_type": "mvit",
        "num_frames": 16,
        "sampling_rate": 4,
        "image_size": 224,
        "resize_size": 256,
        "mean": [0.45, 0.45, 0.45],
        "std": [0.225, 0.225, 0.225],
        "batch_size": 4,
        "backbone_lr": 1e-5,
        "head_lr": 1e-4,
        "crop_scale": (0.7, 1.0),
        "crop_ratio": (0.9, 1.1),
        "horizontal_flip_prob": 0.5,
        "checkpoint_prefix": "mvitv1b",
    },

    "mvit_v2_s": {
        "factory": "mvit_v2_s",
        "weights_enum": "MViT_V2_S_Weights",
        "head_type": "mvit",
        "num_frames": 16,
        "sampling_rate": 4,
        "image_size": 224,
        "resize_size": 256,
        "mean": [0.45, 0.45, 0.45],
        "std": [0.225, 0.225, 0.225],
        "batch_size": 4,
        "backbone_lr": 1e-5,
        "head_lr": 1e-4,
        "crop_scale": (0.7, 1.0),
        "crop_ratio": (0.9, 1.1),
        "horizontal_flip_prob": 0.5,
        "checkpoint_prefix": "mvitv2s",
    },

    "swin3d_t": {
        "factory": "swin3d_t",
        "weights_enum": "Swin3D_T_Weights",
        "head_type": "linear",
        "num_frames": 32,
        "sampling_rate": 2,
        "image_size": 224,
        "resize_size": 256,
        "mean": [0.43216, 0.394666, 0.37645],
        "std": [0.22803, 0.22145, 0.216989],
        "batch_size": 4,
        "backbone_lr": 1e-5,
        "head_lr": 1e-4,
        "crop_scale": (0.7, 1.0),
        "crop_ratio": (0.9, 1.1),
        "horizontal_flip_prob": 0.5,
        "checkpoint_prefix": "swin3dt",
    },

    "swin3d_b": {
        "factory": "swin3d_b",
        "weights_enum": "Swin3D_B_Weights",
        "head_type": "linear",
        "num_frames": 32,
        "sampling_rate": 2,
        "image_size": 224,
        "resize_size": 256,
        "mean": [0.43216, 0.394666, 0.37645],
        "std": [0.22803, 0.22145, 0.216989],
        "batch_size": 4,
        "backbone_lr": 1e-5,
        "head_lr": 1e-4,
        "crop_scale": (0.7, 1.0),
        "crop_ratio": (0.9, 1.1),
        "horizontal_flip_prob": 0.5,
        "checkpoint_prefix": "swin3db",
    },

    "r2plus1d_18": {
        "factory": "r2plus1d_18",
        "weights_enum": "R2Plus1D_18_Weights",
        "head_type": "linear",
        "num_frames": 16,
        "sampling_rate": 4,
        "image_size": 112,
        "resize_size": 128,
        "mean": [0.43216, 0.394666, 0.37645],
        "std": [0.22803, 0.22145, 0.216989],
        "batch_size": 4,
        "backbone_lr": 1e-5,
        "head_lr": 1e-4,
        "crop_scale": (0.7, 1.0),
        "crop_ratio": (0.9, 1.1),
        "horizontal_flip_prob": 0.5,
        "checkpoint_prefix": "r2plus1d18",
    },
}


# ============================================================
# GLOBAL CONFIGURATION
# ============================================================

EPOCHS = 40
EARLY_STOPPING_PATIENCE = 7

WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.05

NUM_WORKERS = 0
SEED = 42
LOG_EVERY = 10

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Device: {DEVICE}")


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="Training and evaluation of torchvision video models."
    )

    parser.add_argument(
        "--mode",
        choices=["train", "test"],
        default="train",
    )

    parser.add_argument(
        "--model",
        choices=list(MODEL_CONFIGS.keys()),
        default="mvit_v2_s",
    )

    parser.add_argument(
        "--protocol",
        choices=["subject_independent", "subject_dependent"],
        default="subject_independent",
    )

    parser.add_argument(
        "--split-mode",
        choices=["single_split", "five_fold"],
        default="single_split",
    )

    parser.add_argument(
        "--fold",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=REPO_ROOT / "splits",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs",
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=EARLY_STOPPING_PATIENCE,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=NUM_WORKERS,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
    )

    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=LABEL_SMOOTHING,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=WEIGHT_DECAY,
    )

    parser.add_argument(
        "--log-every",
        type=int,
        default=LOG_EVERY,
    )

    args = parser.parse_args()

    if args.data_dir is None:
        env_dir = os.environ.get("THETIS_RGB_DIR")

        if env_dir:
            args.data_dir = Path(env_dir).expanduser()
        else:
            args.data_dir = REPO_ROOT / "VIDEO_RGB"

    args.data_dir = args.data_dir.expanduser()
    args.splits_dir = args.splits_dir.expanduser()
    args.output_dir = args.output_dir.expanduser()

    if args.mode == "test" and args.checkpoint is None:
        parser.error("--checkpoint is required when --mode test")

    if args.split_mode == "five_fold":
        if args.fold is None or not 1 <= args.fold <= 5:
            parser.error(
                "--fold must be between 1 and 5 for five_fold"
            )
    elif args.fold is not None:
        parser.error(
            "--fold should only be specified for five_fold"
        )

    if args.epochs < 1:
        parser.error("--epochs must be positive")

    if args.patience < 1:
        parser.error("--patience must be positive")

    return args


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# SPLIT PATHS
# ============================================================

def get_split_directory(args):

    split_dir = (
        args.splits_dir
        / args.protocol
        / args.split_mode
    )

    if args.split_mode == "five_fold":
        split_dir = split_dir / f"fold_{args.fold}"

    return split_dir


def get_output_directory(args):

    output_dir = (
        args.output_dir
        / args.model
        / args.protocol
        / args.split_mode
    )

    if args.split_mode == "five_fold":
        output_dir = output_dir / f"fold_{args.fold}"

    output_dir.mkdir(parents=True, exist_ok=True)

    return output_dir


# ============================================================
# VIDEO PATH RESOLUTION
# ============================================================

def resolve_video_path(path_string, data_dir):

    path_string = path_string.strip()

    path = Path(path_string)

    if path.is_absolute() and path.is_file():
        return path

    # Support paths from the previous Linux environment:
    # /tf/data/VIDEO_RGB/VIDEO_RGB/backhand/video.avi

    normalized = path_string.replace("\\", "/")

    marker = "/VIDEO_RGB/"

    if marker in normalized:
        relative_part = normalized.rsplit(marker, 1)[1]
        candidate = data_dir / relative_part

        if candidate.is_file():
            return candidate

    # Support README paths:
    # VIDEO_RGB/backhand/video.avi

    if normalized.startswith("VIDEO_RGB/"):
        normalized = normalized[len("VIDEO_RGB/"):]

    candidate = data_dir / Path(normalized)

    if candidate.is_file():
        return candidate

    # Also try a path relative to the repository.

    candidate = REPO_ROOT / Path(path_string)

    if candidate.is_file():
        return candidate

    raise FileNotFoundError(
        f"Video file not found: {path_string}\n"
        f"Checked data directory: {data_dir}"
    )


# ============================================================
# CSV SPLIT LOADING
# ============================================================

def read_split(csv_path, data_dir):

    samples = []

    missing_files = []

    with open(csv_path, "r", encoding="utf-8") as file:

        reader = csv.reader(file, delimiter="|")

        for row_number, row in enumerate(reader, start=1):

            if not row:
                continue

            if len(row) != 2:
                raise ValueError(
                    f"Invalid CSV row {row_number} in {csv_path}: {row}"
                )

            video_path, class_id = row

            class_id = int(class_id)

            if not 0 <= class_id < NUM_CLASSES:
                raise ValueError(
                    f"Invalid class ID {class_id} in {csv_path}, "
                    f"row {row_number}"
                )

            try:
                resolved_path = resolve_video_path(
                    video_path,
                    data_dir,
                )

                samples.append(
                    (str(resolved_path), class_id)
                )

            except FileNotFoundError:
                missing_files.append(video_path)

    if missing_files:

        examples = "\n".join(missing_files[:10])

        raise FileNotFoundError(
            f"{len(missing_files)} video files are missing.\n"
            f"Examples:\n{examples}"
        )

    if not samples:
        raise ValueError(f"No samples found in {csv_path}")

    return samples


# ============================================================
# VIDEO LOADING
# ============================================================

def load_video(
    path,
    training,
    num_frames,
    sampling_rate,
):

    cap = cv2.VideoCapture(path)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")

    frame_count = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    if frame_count <= 0:
        cap.release()
        raise RuntimeError(
            f"Video contains no frames: {path}"
        )

    clip_span = (
        (num_frames - 1) * sampling_rate + 1
    )

    if frame_count >= clip_span:

        max_start = frame_count - clip_span

        if training:
            start = random.randint(0, max_start)
        else:
            start = max_start // 2

        indices = (
            start
            + np.arange(num_frames) * sampling_rate
        )

    else:

        indices = (
            np.arange(num_frames) * sampling_rate
        )

        if training:
            start = random.randint(
                0,
                max(0, frame_count - 1),
            )
        else:
            start = 0

        indices = np.clip(
            indices + start,
            0,
            frame_count - 1,
        )

    frames = []

    for index in indices:

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            int(index),
        )

        success, frame = cap.read()

        if not success:
            cap.release()

            raise RuntimeError(
                f"Cannot read frame {index} from {path}"
            )

        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )

        frames.append(
            Image.fromarray(frame)
        )

    cap.release()

    return frames


# ============================================================
# VIDEO TRANSFORMATIONS
# ============================================================

def transform_clip(
    images,
    training,
    config,
):

    image_size = config["image_size"]

    resize_size = config["resize_size"]

    if training:

        top, left, height, width = (
            RandomResizedCrop.get_params(
                images[0],
                scale=config["crop_scale"],
                ratio=config["crop_ratio"],
            )
        )

        images = [
            TF.resized_crop(
                image,
                top,
                left,
                height,
                width,
                size=[image_size, image_size],
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            )
            for image in images
        ]

        if random.random() < config["horizontal_flip_prob"]:

            images = [
                TF.hflip(image)
                for image in images
            ]

    else:

        images = [
            TF.center_crop(
                TF.resize(
                    image,
                    resize_size,
                    interpolation=InterpolationMode.BILINEAR,
                    antialias=True,
                ),
                [image_size, image_size],
            )
            for image in images
        ]

    images = [
        TF.to_tensor(image)
        for image in images
    ]

    mean = config["mean"]
    std = config["std"]

    images = [
        TF.normalize(
            image,
            mean=mean,
            std=std,
        )
        for image in images
    ]

    # T, C, H, W -> C, T, H, W

    video = torch.stack(images, dim=0)

    video = video.permute(1, 0, 2, 3)

    return video


# ============================================================
# DATASET
# ============================================================

class TennisVideoDataset(Dataset):

    def __init__(
        self,
        samples,
        config,
        training=False,
    ):

        self.samples = samples
        self.config = config
        self.training = training

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        path, label = self.samples[index]

        images = load_video(
            path,
            training=self.training,
            num_frames=self.config["num_frames"],
            sampling_rate=self.config["sampling_rate"],
        )

        video = transform_clip(
            images,
            training=self.training,
            config=self.config,
        )

        return video, label


# ============================================================
# MODEL CREATION
# ============================================================

def create_model(model_name, pretrained=True):

    config = MODEL_CONFIGS[model_name]

    video_models = torchvision.models.video

    factory_name = config["factory"]
    weights_enum_name = config["weights_enum"]

    if not hasattr(video_models, factory_name):
        raise RuntimeError(
            f"Model {factory_name} is unavailable "
            f"in torchvision {torchvision.__version__}"
        )

    if not hasattr(video_models, weights_enum_name):
        raise RuntimeError(
            f"Weights {weights_enum_name} are unavailable "
            f"in torchvision {torchvision.__version__}"
        )

    factory = getattr(video_models, factory_name)

    weights_enum = getattr(
        video_models,
        weights_enum_name,
    )

    weights = (
        weights_enum.KINETICS400_V1
        if pretrained
        else None
    )

    model = factory(weights=weights)

    if config["head_type"] == "mvit":

        in_features = model.head[1].in_features

        model.head[1] = nn.Linear(
            in_features,
            NUM_CLASSES,
        )

    else:

        in_features = model.head[-1].in_features \
            if model_name.startswith("swin3d") \
            else model.fc.in_features

        if model_name.startswith("swin3d"):

            model.head[-1] = nn.Linear(
                in_features,
                NUM_CLASSES,
            )

        else:

            model.fc = nn.Linear(
                in_features,
                NUM_CLASSES,
            )

    return model


# ============================================================
# OPTIMIZER
# ============================================================

def create_optimizer(
    model,
    config,
    weight_decay,
):

    if config["head_type"] == "mvit":
        head_parameters = model.head.parameters()

    elif config["factory"].startswith("swin3d"):
        head_parameters = model.head.parameters()

    else:
        head_parameters = model.fc.parameters()

    head_parameter_ids = {
        id(parameter)
        for parameter in head_parameters
    }

    backbone_parameters = [
        parameter
        for parameter in model.parameters()
        if id(parameter) not in head_parameter_ids
    ]

    if config["head_type"] == "mvit":
        head_parameters = list(model.head.parameters())

    elif config["factory"].startswith("swin3d"):
        head_parameters = list(model.head.parameters())

    else:
        head_parameters = list(model.fc.parameters())

    optimizer = torch.optim.AdamW(
        [
            {
                "params": backbone_parameters,
                "lr": config["backbone_lr"],
            },
            {
                "params": head_parameters,
                "lr": config["head_lr"],
            },
        ],
        weight_decay=weight_decay,
    )

    return optimizer


# ============================================================
# EPOCH
# ============================================================

def run_epoch(
    model,
    loader,
    criterion,
    optimizer=None,
    scaler=None,
    training=False,
    log_every=10,
    epoch=None,
):

    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0

    all_labels = []
    all_predictions = []

    for batch_idx, (videos, labels) in enumerate(loader):

        videos = videos.to(
            DEVICE,
            non_blocking=True,
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True,
        )

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):

            with torch.autocast(
                device_type=DEVICE.type,
                dtype=torch.float16,
                enabled=(DEVICE.type == "cuda"),
            ):

                outputs = model(videos)

                loss = criterion(
                    outputs,
                    labels,
                )

            if training:

                scaler.scale(loss).backward()

                scaler.step(optimizer)

                scaler.update()

        predictions = outputs.argmax(dim=1)

        total_loss += (
            loss.item() * labels.size(0)
        )

        all_labels.extend(
            labels.detach().cpu().numpy()
        )

        all_predictions.extend(
            predictions.detach().cpu().numpy()
        )

        if (
            log_every > 0
            and (batch_idx + 1) % log_every == 0
        ):

            phase = "Train" if training else "Eval"

            print(
                f"[{phase}] "
                f"Epoch {epoch if epoch is not None else '-'} "
                f"Batch {batch_idx + 1}/{len(loader)} "
                f"Loss: {loss.item():.4f}"
            )

    average_loss = total_loss / len(loader.dataset)

    accuracy = accuracy_score(
        all_labels,
        all_predictions,
    )

    macro_f1 = f1_score(
        all_labels,
        all_predictions,
        labels=list(range(NUM_CLASSES)),
        average="macro",
        zero_division=0,
    )

    return {
        "loss": average_loss,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "labels": all_labels,
        "predictions": all_predictions,
    }


# ============================================================
# CHECKPOINT LOADING
# ============================================================

def load_checkpoint_file(checkpoint_path):

    try:

        checkpoint = torch.load(
            checkpoint_path,
            map_location=DEVICE,
            weights_only=True,
        )

    except TypeError:



        checkpoint = torch.load(
            checkpoint_path,
            map_location=DEVICE,
        )

    return checkpoint


def extract_model_state(checkpoint):

    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:
            return checkpoint["model_state_dict"]

        if "state_dict" in checkpoint:
            return checkpoint["state_dict"]

    return checkpoint


def load_model_checkpoint(model, checkpoint_path):

    checkpoint = load_checkpoint_file(
        checkpoint_path
    )

    state_dict = extract_model_state(checkpoint)



    if (
        isinstance(state_dict, dict)
        and any(key.startswith("module.") for key in state_dict)
    ):

        state_dict = {
            key.replace("module.", "", 1): value
            for key, value in state_dict.items()
        }

    model.load_state_dict(
        state_dict,
        strict=True,
    )

    if isinstance(checkpoint, dict):
        epoch = checkpoint.get("epoch", 0)
    else:
        epoch = 0

    return epoch


# ============================================================
# CHECKPOINT SAVING
# ============================================================

def save_checkpoint(
    path,
    model,
    optimizer,
    scaler,
    epoch,
    val_metrics,
    model_name,
    args,
    config,
):

    checkpoint = {
        "model_name": model_name,
        "epoch": epoch,

        "model_state_dict": model.state_dict(),

        "optimizer_state_dict": optimizer.state_dict(),

        "scaler_state_dict": scaler.state_dict(),

        "val_accuracy": val_metrics["accuracy"],

        "val_macro_f1": val_metrics["macro_f1"],

        "val_loss": val_metrics["loss"],

        "num_classes": NUM_CLASSES,

        "num_frames": config["num_frames"],

        "sampling_rate": config["sampling_rate"],

        "image_size": config["image_size"],

        "protocol": args.protocol,

        "split_mode": args.split_mode,

        "fold": args.fold,

        "weights": "KINETICS400_V1",
    }

    torch.save(
        checkpoint,
        path,
    )


# ============================================================
# TRAINING
# ============================================================

def train_model(
    model,
    train_loader,
    val_loader,
    config,
    args,
    output_dir,
):

    criterion = nn.CrossEntropyLoss(
        label_smoothing=args.label_smoothing
    )

    optimizer = create_optimizer(
        model,
        config,
        args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(DEVICE.type == "cuda"),
    )

    prefix = config["checkpoint_prefix"]

    best_path = (
        output_dir / f"{prefix}_best.pth"
    )

    last_path = (
        output_dir / f"{prefix}_last.pth"
    )

    history = []

    best_macro_f1 = -1.0

    best_epoch = 0

    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):

        print("\n" + "=" * 70)
        print(f"EPOCH {epoch}/{args.epochs}")
        print("=" * 70)

        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            training=True,
            log_every=args.log_every,
            epoch=epoch,
        )

        val_metrics = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            training=False,
            log_every=args.log_every,
            epoch=epoch,
        )

        scheduler.step()

        print(
            f"\nTrain: "
            f"loss={train_metrics['loss']:.4f}, "
            f"accuracy={train_metrics['accuracy']:.4f}, "
            f"macro-F1={train_metrics['macro_f1']:.4f}"
        )

        print(
            f"Val: "
            f"loss={val_metrics['loss']:.4f}, "
            f"accuracy={val_metrics['accuracy']:.4f}, "
            f"macro-F1={val_metrics['macro_f1']:.4f}"
        )

        history.append(
            {
                "epoch": epoch,

                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "train_macro_f1": train_metrics["macro_f1"],

                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],

                "learning_rate_backbone": optimizer.param_groups[0]["lr"],
                "learning_rate_head": optimizer.param_groups[1]["lr"],
            }
        )

        # Save the latest checkpoint.

        save_checkpoint(
            path=last_path,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            epoch=epoch,
            val_metrics=val_metrics,
            model_name=args.model,
            args=args,
            config=config,
        )

        # Select the best checkpoint using validation macro-F1.

        if val_metrics["macro_f1"] > best_macro_f1:

            best_macro_f1 = val_metrics["macro_f1"]

            best_epoch = epoch

            epochs_without_improvement = 0

            save_checkpoint(
                path=best_path,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch,
                val_metrics=val_metrics,
                model_name=args.model,
                args=args,
                config=config,
            )

            print(
                f"New best checkpoint: "
                f"macro-F1={best_macro_f1:.4f}"
            )

        else:

            epochs_without_improvement += 1

        # Save training history.

        pd.DataFrame(history).to_csv(
            output_dir / "training_history.csv",
            index=False,
        )

        if epochs_without_improvement >= args.patience:

            print(
                f"\nEarly stopping after "
                f"{args.patience} epochs without improvement."
            )

            break

    print("\nTraining completed.")

    print(f"Best epoch: {best_epoch}")
    print(f"Best validation macro-F1: {best_macro_f1:.4f}")
    print(f"Best checkpoint: {best_path}")

    return best_path, best_epoch


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(
    model,
    test_loader,
    checkpoint_path,
    output_dir,
):

    print("\nLoading checkpoint:")
    print(checkpoint_path)



    checkpoint_epoch = load_model_checkpoint(
        model,
        checkpoint_path,
    )

    model.eval()

    criterion = nn.CrossEntropyLoss()

    test_metrics = run_epoch(
        model=model,
        loader=test_loader,
        criterion=criterion,
        training=False,
        log_every=0,
        epoch=checkpoint_epoch,
    )

    labels = test_metrics["labels"]

    predictions = test_metrics["predictions"]

    report = classification_report(
        labels,
        predictions,
        labels=list(range(NUM_CLASSES)),
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
    )

    cm = confusion_matrix(
        labels,
        predictions,
        labels=list(range(NUM_CLASSES)),
    )

    print("\n" + "=" * 70)
    print("TEST RESULTS")
    print("=" * 70)

    print(
        f"Test loss: {test_metrics['loss']:.4f}"
    )

    print(
        f"Test accuracy: {test_metrics['accuracy']:.4f}"
    )

    print(
        f"Test macro-F1: {test_metrics['macro_f1']:.4f}"
    )

    print("\nClassification report:")
    print(report)

    # Classification report.

    with open(
        output_dir / "test_classification_report.txt",
        "w",
        encoding="utf-8",
    ) as file:

        file.write(report)

    # Confusion matrix.

    pd.DataFrame(
        cm,
        index=CLASS_NAMES,
        columns=CLASS_NAMES,
    ).to_csv(
        output_dir / "test_confusion_matrix.csv"
    )

    # Test summary.

    summary = pd.DataFrame(
        [
            {
                "checkpoint": str(checkpoint_path),
                "checkpoint_epoch": checkpoint_epoch,
                "test_loss": test_metrics["loss"],
                "test_accuracy": test_metrics["accuracy"],
                "test_macro_f1": test_metrics["macro_f1"],
            }
        ]
    )

    summary.to_csv(
        output_dir / "test_summary.csv",
        index=False,
    )

    print("\nEvaluation files saved to:")
    print(output_dir)

    return test_metrics


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()

    set_seed(args.seed)

    config = MODEL_CONFIGS[args.model].copy()

    if args.batch_size is not None:
        config["batch_size"] = args.batch_size

    output_dir = get_output_directory(args)

    split_dir = get_split_directory(args)

    print("\n" + "=" * 70)
    print("CONFIGURATION")
    print("=" * 70)

    print(f"Mode: {args.mode}")
    print(f"Model: {args.model}")
    print(f"Protocol: {args.protocol}")
    print(f"Split mode: {args.split_mode}")
    print(f"Fold: {args.fold}")
    print(f"Data directory: {args.data_dir}")
    print(f"Split directory: {split_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Device: {DEVICE}")

    print("\nModel parameters:")
    print(f"Frames: {config['num_frames']}")
    print(f"Sampling rate: {config['sampling_rate']}")
    print(f"Image size: {config['image_size']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Backbone LR: {config['backbone_lr']}")
    print(f"Head LR: {config['head_lr']}")

    # --------------------------------------------------------
    # TEST-ONLY MODE
    # --------------------------------------------------------

    if args.mode == "test":

        test_csv = split_dir / "test.csv"

        if not test_csv.is_file():
            raise FileNotFoundError(
                f"Test split not found: {test_csv}"
            )

        test_samples = read_split(
            test_csv,
            args.data_dir,
        )

        print(
            f"\nTest samples: {len(test_samples)}"
        )

        test_dataset = TennisVideoDataset(
            samples=test_samples,
            config=config,
            training=False,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(DEVICE.type == "cuda"),
        )

        if not args.checkpoint.is_file():
            raise FileNotFoundError(
                f"Checkpoint not found: {args.checkpoint}"
            )



        model = create_model(
            args.model,
            pretrained=False,
        )

        model = model.to(DEVICE)

        evaluate_model(
            model=model,
            test_loader=test_loader,
            checkpoint_path=args.checkpoint,
            output_dir=output_dir,
        )

        return

    # --------------------------------------------------------
    # TRAINING MODE
    # --------------------------------------------------------

    train_csv = split_dir / "train.csv"
    val_csv = split_dir / "val.csv"
    test_csv = split_dir / "test.csv"

    for csv_path in [train_csv, val_csv, test_csv]:

        if not csv_path.is_file():
            raise FileNotFoundError(
                f"Split file not found: {csv_path}"
            )

    train_samples = read_split(
        train_csv,
        args.data_dir,
    )

    val_samples = read_split(
        val_csv,
        args.data_dir,
    )

    test_samples = read_split(
        test_csv,
        args.data_dir,
    )

    print("\nDataset sizes:")
    print(f"Train: {len(train_samples)}")
    print(f"Validation: {len(val_samples)}")
    print(f"Test: {len(test_samples)}")

    train_dataset = TennisVideoDataset(
        samples=train_samples,
        config=config,
        training=True,
    )

    val_dataset = TennisVideoDataset(
        samples=val_samples,
        config=config,
        training=False,
    )

    test_dataset = TennisVideoDataset(
        samples=test_samples,
        config=config,
        training=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(DEVICE.type == "cuda"),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(DEVICE.type == "cuda"),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(DEVICE.type == "cuda"),
    )

    # Load Kinetics400 pretrained weights for training.

    model = create_model(
        args.model,
        pretrained=True,
    )

    model = model.to(DEVICE)

    best_checkpoint, best_epoch = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        args=args,
        output_dir=output_dir,
    )

    # Evaluate the best checkpoint on the test set.
    # Loading takes place once inside evaluate_model().

    evaluate_model(
        model=model,
        test_loader=test_loader,
        checkpoint_path=best_checkpoint,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
