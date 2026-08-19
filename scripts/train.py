"""Train the contrastive image encoder from a dataset snapshot."""

import argparse

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from representation_learning.data.augmentations import (
    ContrastiveTransform,
)
from representation_learning.data.dataset import (
    ContrastiveImageDataset,
    MountedBlobImageReader,
    load_dataset_manifest,
)
from representation_learning.data.splitting import (
    DeterministicDatasetSplitter,
)
from representation_learning.losses.contrastive_loss import (
    ContrastiveLoss,
)
from representation_learning.models.contrastive_model import (
    ContrastiveModel,
)
from representation_learning.training.checkpointing import (
    CheckpointManager,
)
from representation_learning.training.experiment import (
    ExperimentConfig,
    TrainingExperiment,
)
from representation_learning.training.trainer import (
    ContrastiveTrainer,
)
from representation_learning.utils.config import (
    load_training_config,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the contrastive image encoder",
    )
    parser.add_argument(
        "--config",
        default="configs/train.yaml",
        help="Training configuration path",
    )
    parser.add_argument(
        "--manifest-path",
        help="Local path to manifest.jsonl",
    )
    parser.add_argument(
        "--image-mount-directory",
        help="Directory containing the mounted accepted image blobs",
    )
    parser.add_argument(
        "--dataset-snapshot-id",
        required=True,
        help="ID of the dataset snapshot used for this run",
    )

    return parser.parse_args()


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def main() -> None:
    arguments = parse_arguments()
    config = load_training_config(arguments.config)

    manifest_path = arguments.manifest_path or config.data.manifest_path
    image_mount_directory = (
        arguments.image_mount_directory or config.data.image_mount_directory
    )

    if manifest_path is None:
        raise ValueError(
            "Manifest path must be provided through --manifest-path "
            "or configs/train.yaml"
        )

    if image_mount_directory is None:
        raise ValueError(
            "Image mount directory must be provided through "
            "--image-mount-directory or configs/train.yaml"
        )

    torch.manual_seed(config.training.seed)

    items = load_dataset_manifest(manifest_path)

    splitter = DeterministicDatasetSplitter(
        train_ratio=config.data.train_ratio,
        validation_ratio=config.data.validation_ratio,
        test_ratio=config.data.test_ratio,
        seed=config.data.split_seed,
    )
    partitions = splitter.split(items)

    print(f"Snapshot: {arguments.dataset_snapshot_id}")
    print(f"Total images: {len(items)}")
    print(f"Training images: {len(partitions.train)}")
    print(f"Validation images: {len(partitions.validation)}")
    print(f"Test images: {len(partitions.test)}")

    if len(partitions.train) < 2:
        raise ValueError("Training partition requires at least two images")

    if len(partitions.validation) < 2:
        raise ValueError("Validation partition requires at least two images")

    transform = ContrastiveTransform(
        image_size=config.data.image_size,
    )
    image_reader = MountedBlobImageReader(
        image_mount_directory,
    )

    training_dataset = ContrastiveImageDataset(
        items=partitions.train,
        image_reader=image_reader,
        transform=transform,
    )
    validation_dataset = ContrastiveImageDataset(
        items=partitions.validation,
        image_reader=image_reader,
        transform=transform,
    )

    common_loader_options = {
        "num_workers": config.data.num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": config.data.num_workers > 0,
    }

    training_batch_size = min(
        config.data.batch_size,
        len(training_dataset),
    )

    validation_batch_size = min(
        config.data.batch_size,
        len(validation_dataset),
    )

    training_loader = DataLoader(
        training_dataset,
        batch_size=training_batch_size,
        shuffle=True,
        drop_last=True,
        **common_loader_options,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=validation_batch_size,
        shuffle=False,
        drop_last=True,
        **common_loader_options,
    )

    device = select_device()
    print(f"Device: {device}")

    model = ContrastiveModel(
        projection_dimension=config.model.projection_dimension,
    )

    if config.model.input_channels != 3:
        raise ValueError("The current encoder supports exactly three input channels")

    if model.encoder.feature_dimension != config.model.embedding_dimension:
        raise ValueError("Configured embedding dimension does not match the encoder")

    loss_function = ContrastiveLoss(
        temperature=config.training.temperature,
    )
    optimizer = AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )

    checkpoint_manager = CheckpointManager(
        config.checkpointing.output_directory,
    )

    if config.checkpointing.resume_from is not None:
        checkpoint_manager.load(
            checkpoint_path=config.checkpointing.resume_from,
            model=model,
            optimizer=optimizer,
            device=device,
        )

    trainer = ContrastiveTrainer(
        model=model,
        loss_function=loss_function,
        optimizer=optimizer,
        device=device,
    )

    experiment = TrainingExperiment(
        trainer=trainer,
        checkpoint_manager=checkpoint_manager,
        config=ExperimentConfig(
            maximum_epochs=config.training.maximum_epochs,
            patience=config.early_stopping.patience,
            minimum_improvement=(config.early_stopping.minimum_improvement),
        ),
        device=device,
    )

    result = experiment.run(
        training_loader=training_loader,
        validation_loader=validation_loader,
        dataset_snapshot_id=arguments.dataset_snapshot_id,
    )

    print(f"Best epoch: {result.best_epoch}")
    print(f"Best validation loss: {result.best_validation_loss:.6f}")
    print(f"Epochs completed: {result.epochs_completed}")
    print(f"Stopped early: {result.stopped_early}")
    print(f"Checkpoint: {result.checkpoint_path}")


if __name__ == "__main__":
    main()
