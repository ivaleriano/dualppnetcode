import argparse
import json
from abc import ABCMeta, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Sequence, Tuple

import torch
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from shape_continuum.data_utils import adni_hdf, mesh_utils
from shape_continuum.models.base import BaseModel
from shape_continuum.networks import mesh_networks, point_networks, vol_networks
from shape_continuum.training.hooks import CheckpointSaver, TensorBoardLogger
from shape_continuum.training.metrics import Accuracy, Mean, Metric
from shape_continuum.training.optim import ClippedStepLR
from shape_continuum.training.train_and_eval import train_and_evaluate
from shape_continuum.training.wrappers import LossWrapper, MeshNamedDataLoader, NamedDataLoader


def create_parser():
    parser = argparse.ArgumentParser("PointNet")
    parser.add_argument("--batchsize", type=int, default=20, help="input batch size")
    parser.add_argument("--workers", type=int, default=4, help="number of data loading workers")
    parser.add_argument("--epoch", type=int, default=200, help="number of epochs for training")
    parser.add_argument("--num_points", type=int, default=1500, help="number of epochs for training")
    parser.add_argument("--pretrain", type=Path, help="whether use pretrain model")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="learning rate for training")
    parser.add_argument("--decay_rate", type=float, default=1e-4, help="weight decay")
    parser.add_argument("--optimizer", choices=["Adam", "SGD"], default="Adam", help="type of optimizer")
    parser.add_argument("--task", choices=["clf", "surv"], default="clf", help="classification or survival analysis")
    parser.add_argument("--train_data", type=Path, required=True, help="path to training dataset")
    parser.add_argument("--test_data", type=Path, required=True, help="path to testing dataset")
    parser.add_argument(
        "--discriminator_net", default="pointnet", help="which architecture to use for discriminator",
    )
    parser.add_argument(
        "--shape",
        default="pointcloud",
        help="which shape representation to use "
        "(pointcloud,mesh,mask,vol_with_bg,vol_without_bg",
    )
    parser.add_argument("--num_classes", type=int, default=3, help="number of classes")
    parser.add_argument(
        "--experiment_name",
        action="store_true",
        default=False,
        help="True if input a particular name for the experiment (default False: current date and time)",
    )
    parser.add_argument(
        "--tb_comment", action="store_true", default=False, help="any comment for storing on tensorboard",
    )
    parser.add_argument(
        "--tensorboard", action="store_true", default=False, help="visualize training progress on tensorboard",
    )

    return parser


def get_number_of_parameters(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


class BaseModelFactory(metaclass=ABCMeta):
    """Abstract base class for creating models and data loaders.

    Args:
      arguments:
        Command line arguments.
    """

    def __init__(self, arguments: argparse.Namespace) -> None:
        self.args = arguments

    def make_directories(self) -> Tuple[str, str, str]:
        """"Create directories to hold logs and checkpoints.

        Returns:
          experiment_dir (Path):
            Path to base directory.
          checkpoints_dir (Path):
            Path to directory where checkpoints should be saved to.
          tb_dir (Path):
            Path to directory where TensorBoard log should be written to.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")

        args = self.args
        base_dir = Path(f"experiments_{args.task}")
        if args.experiment_name:
            experiment = input("input a name for your experiment")
        else:
            experiment = f"shape_{args.shape}_network_{args.discriminator_net}"

        experiment_dir = base_dir / experiment / timestamp
        experiment_dir.mkdir(parents=True, exist_ok=True)

        checkpoints_dir = experiment_dir / "checkpoints"
        checkpoints_dir.mkdir(exist_ok=True)

        tb_dir = experiment_dir / "tb_log"
        return experiment_dir, checkpoints_dir, tb_dir

    def get_optimizer(self, params: Sequence[torch.Tensor]) -> Optimizer:
        """Create an optimizer.

        Args:
          params (list of of torch.Tensor):
            List of parameters to optimize.

        Returns:
          optim (Optimizer):
            Instance of the selected optimizer.
        """
        args = self.args
        if args.optimizer == "SGD":
            optimizerD = torch.optim.SGD(params, lr=0.01, momentum=0.9)
        elif args.optimizer == "Adam":
            optimizerD = torch.optim.Adam(
                params, lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.decay_rate,
            )
        else:
            raise ValueError(f"unknown optimizer {args.optimizer}")
        return optimizerD

    def write_args(self, filename: str) -> None:
        """Write command line arguments to JSON file.

        Args:
          filename (str):
            Path to JSON file.
        """
        args = vars(self.args)
        for k, v in args.items():
            if isinstance(v, Path):
                args[k] = str(v.resolve())

        with open(filename, "w") as f:
            json.dump(args, f, indent=2)

    def _init_model(self, model: BaseModel) -> None:
        """Initialize the model.

        If path to checkpoint is provided, initializes weights from checkpoint.

        Args:
          model (BaseModel):
            Model to initialize.
        """
        args = self.args
        model_name = model.__class__.__name__
        if args.pretrain is not None:
            print(f"Load {model_name} model from {args.pretrain}")
            model.load_state_dict(torch.load(args.pretrain))
        else:
            print(f"Training {model_name} from scratch")

        n_params = get_number_of_parameters(model)
        print(f"Number of parameters: {n_params:,}")

    def get_loss(self) -> LossWrapper:
        """Return the loss to optimize."""
        loss = LossWrapper(
            torch.nn.CrossEntropyLoss(), input_names=["logits", "target"], output_names=["cross_entropy"]
        )
        return loss

    def get_metrics(self) -> Sequence[Metric]:
        """Returna a list of metrics to compute."""
        metrics = [Mean("cross_entropy"), Accuracy("logits", "target")]
        return metrics

    def get_and_init_model(self) -> BaseModel:
        """Create and initialize a model."""
        model = self.get_model()
        self._init_model(model)
        return model

    @abstractmethod
    def get_model(self) -> BaseModel:
        """Returns a model instance."""

    @abstractmethod
    def get_data(self) -> Tuple[DataLoader, DataLoader]:
        """Returns a data loader instance for training and evaluation, respectively."""


class ImageModelFactory(BaseModelFactory):
    """Factory for models taking 3D image volumes."""

    def get_data(self):
        args = self.args
        train_data, transform_kwargs = adni_hdf.get_image_dataset_for_train(args.train_data, args.shape, rescale=True)
        trainDataLoader = NamedDataLoader(
            train_data, output_names=["image", "target"], batch_size=args.batchsize, shuffle=True, drop_last=True,
        )

        eval_data = adni_hdf.get_image_dataset_for_eval(args.test_data, transform_kwargs, args.shape)
        valDataLoader = NamedDataLoader(eval_data, output_names=["image", "target"], batch_size=args.batchsize)
        return trainDataLoader, valDataLoader

    def get_model(self):
        args = self.args
        in_channels = 1
        if args.discriminator_net == "resnet":
            return vol_networks.ResNet(in_channels, args.num_classes)
        elif args.discriminator_net == "convnet":
            return vol_networks.Vol_classifier(in_channels, args.num_classes)
        else:
            raise ValueError("network {!r} is unsupported".format(args.discriminator_net))


class PointCloudModelFactory(BaseModelFactory):
    """Factory for models taking 3D point clouds."""

    def get_data(self):
        args = self.args
        train_data, transform_kwargs = adni_hdf.get_point_cloud_dataset_for_train(args.train_data)
        trainDataLoader = NamedDataLoader(
            train_data, output_names=["pointcloud", "target"], batch_size=args.batchsize, shuffle=True, drop_last=True,
        )

        eval_data = adni_hdf.get_point_cloud_dataset_for_eval(args.test_data, transform_kwargs)
        valDataLoader = NamedDataLoader(eval_data, output_names=["pointcloud", "target"], batch_size=args.batchsize)
        return trainDataLoader, valDataLoader

    def get_model(self):
        args = self.args
        use_batch_norm = args.batchsize > 1
        if args.discriminator_net == "pointnet":
            return point_networks.PointNet(args.num_points, args.num_classes, use_batch_norm)
        elif args.discriminator_net == "pointnet++":
            return point_networks.PointNet2ClsMsg(args.num_classes)
        else:
            raise ValueError("network {!r} is unsupported".format(args.discriminator_net))


class MeshModelFactory(BaseModelFactory):
    def get_data(self):
        args = self.args
        train_data, transform_kwargs, template = adni_hdf.get_mesh_dataset_for_train(args.train_data)
        trainDataLoader = MeshNamedDataLoader(
            train_data, output_names=["mesh", "target"], batch_size=args.batchsize, shuffle=True, drop_last=True,
        )
        eval_data = adni_hdf.get_mesh_dataset_for_eval(args.test_data, transform_kwargs)
        valDataLoader = MeshNamedDataLoader(eval_data, output_names=["mesh", "target"], batch_size=args.batchsize)
        self.template = template
        return trainDataLoader, valDataLoader

    def get_model(self):
        args = self.args
        if args.discriminator_net == "spiralnet":
            in_channels = 3
            seq_length = [9, 9]
            latent_channels = 32
            out_channels = [16, 16]
            dilation = [1, 1]
            device = torch.device("cuda")
            # Creating the spiral sequences and the downsample matrices
            spiral_indices_list = [
                mesh_utils.preprocess_spiral(
                    self.template["face"][idx], seq_length[idx], self.template["vertices"][idx], dilation[idx]
                ).to(device)
                for idx in range(len(self.template["face"]) - 1)
            ]
            down_transform_list = [
                mesh_utils.to_sparse(down_transform).to(device) for down_transform in self.template["down_transform"]
            ]
            return mesh_networks.SpiralNet(
                in_channels, out_channels, latent_channels, spiral_indices_list, down_transform_list, args.num_classes
            )
        else:
            raise ValueError("network {!r} is unsupported".format(args.discriminator_net))


def get_factory(args: argparse.Namespace) -> BaseModelFactory:
    """Returns a factory depending on selected data type from command line arguments."""
    if args.shape == "pointcloud":
        factory = PointCloudModelFactory(args)
    elif args.shape.startswith("vol_") or args.shape == "mask":
        factory = ImageModelFactory(args)
    elif args.shape == "mesh":
        factory = MeshModelFactory(args)
    else:
        raise ValueError("shape {!r} is unsupported".format(args.shape))

    return factory


def main(args=None):
    parser = create_parser()
    args = parser.parse_args(args=args)

    factory = get_factory(args)
    experiment_dir, checkpoints_dir, tb_log_dir = factory.make_directories()

    factory.write_args(experiment_dir / "experiment_args.json")

    trainDataLoader, valDataLoader = factory.get_data()
    discriminator = factory.get_and_init_model()
    optimizerD = factory.get_optimizer(discriminator.parameters())
    loss = factory.get_loss()
    train_metrics = factory.get_metrics()
    train_hooks = [] #[CheckpointSaver(discriminator, checkpoints_dir, save_every_n_epochs=3, max_keep=5)]
    eval_hooks = []
    eval_metrics = factory.get_metrics()
    if args.tensorboard:
        if args.tb_comment:
            comment = input("comment to add to TB visualization: ")
            tb_log_dir /= comment


        train_hooks.append(TensorBoardLogger(str(tb_log_dir / "train"), train_metrics))


        eval_hooks = [TensorBoardLogger(str(tb_log_dir / "eval"), eval_metrics)]
    eval_hooks.append(CheckpointSaver(discriminator, checkpoints_dir, save_every_n_epochs=3, max_keep=5,metrics=eval_metrics,save_best=True))

    train_and_evaluate(
        model=discriminator,
        loss=loss,
        train_data=trainDataLoader,
        optimizer=optimizerD,
        scheduler=ClippedStepLR(optimizerD, step_size=30, gamma=0.5),
        num_epochs=args.epoch,
        eval_data=valDataLoader,
        train_hooks=train_hooks,
        eval_hooks=eval_hooks,
        device=torch.device("cuda"),
    )


if __name__ == "__main__":
    main()
