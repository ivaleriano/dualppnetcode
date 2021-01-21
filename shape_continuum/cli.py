import argparse
import json
from abc import ABCMeta, abstractmethod
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Optional, Sequence, Tuple

import torch
import torch.optim as optim
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, Dataset

from .data_utils import adni_hdf, mesh_utils
from .data_utils.surv_data import cox_collate_fn
from .models.base import BaseModel
from .models.losses import CoxphLoss
from .networks import mesh_networks, point_networks, vol_networks
from .training.metrics import Accuracy, BalancedAccuracy, ConcordanceIndex, Mean, Metric
from .training.wrappers import LossWrapper, NamedDataLoader, mesh_collate


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
    parser.add_argument("--val_data", type=Path, required=True, help="path to validation dataset")
    parser.add_argument("--test_data", type=Path, required=True, help="path to testing dataset")
    parser.add_argument(
        "--discriminator_net", default="pointnet", help="which architecture to use for discriminator",
    )
    parser.add_argument(
        "--shape",
        default="pointcloud",
        help="which shape representation to use (pointcloud,mesh,mask,vol_with_bg,vol_without_bg)",
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


class Trapezoid(optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        n_iterations: int,
        max_lr: float,
        start_lr: Optional[float] = None,
        annihilate: bool = True,
        last_epoch: int = -1,
    ) -> None:
        """
            First warmup: Linearly increase from lower lr to max_lr, train with max_lr for most of the training.
            After 80% of iterations, start linear decline of lr.
            If annihilation, lr is linearly decreased towards an extremely small lr for the last 5% of iteratioins.
            This helps the optimizer to find the abosulte minimum at the current valey.
            Lazy: n_iterations is the total amount of iterations that this scheduler will be used for!
            Developer's note:
            if cyclic momentum would be implemented, according to Superconvergence paper
            https://arxiv.org/abs/1708.07120
            0.85 as min val works just fine. Take that value!
        """

        self.n_iters = n_iterations
        self.max_lr = max_lr
        if start_lr is None:
            self.start_lr = max_lr / 10
        else:
            self.start_lr = start_lr
        self.stop_warmup = int(0.1 * n_iterations)
        self.start_decline = int(0.8 * n_iterations)
        self.start_annihilate = int(0.95 * n_iterations) if annihilate else n_iterations

        super(Trapezoid, self).__init__(optimizer, last_epoch)

    def get_lr(self):

        if self.last_epoch < self.stop_warmup:
            step_size = (self.max_lr - self.start_lr) / self.stop_warmup
            new_lr = self.start_lr + step_size * self.last_epoch
        elif self.last_epoch < self.start_decline:
            new_lr = self.max_lr
        elif self.last_epoch <= self.start_annihilate:
            step_size = (self.max_lr - self.start_lr) / (self.start_annihilate - self.start_decline)
            new_lr = self.max_lr - step_size * (self.last_epoch - self.start_decline)
        else:
            step_size = (self.start_lr - self.start_lr / 20) / (self.n_iters - self.start_annihilate)
            new_lr = self.start_lr - step_size * (self.last_epoch - self.start_annihilate)

        return [new_lr for group in self.optimizer.param_groups]


class BaseModelFactory(metaclass=ABCMeta):
    """Abstract base class for creating models and data loaders.

    Args:
      arguments:
        Command line arguments.
    """

    def __init__(self, arguments: argparse.Namespace) -> None:
        self.args = arguments

        if arguments.task == "clf":
            self._task = adni_hdf.Task.CLASSIFICATION
        elif arguments.task == "surv":
            assert arguments.num_classes == 1
            self._task = adni_hdf.Task.SURVIVAL_ANALYSIS
        else:
            raise ValueError("task={!r} is not supported".format(arguments.task))

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
        if self._task == adni_hdf.Task.SURVIVAL_ANALYSIS:
            loss = LossWrapper(
                CoxphLoss(), input_names=["logits", "event", "riskset"], output_names=["partial_log_lik"]
            )
        else:
            if self.args.num_classes > 1:
                loss = LossWrapper(
                    torch.nn.CrossEntropyLoss(), input_names=["logits", "target"], output_names=["cross_entropy"]
                )
            else:
                loss = LossWrapper(
                    torch.nn.BCEWithLogitsLoss(),
                    input_names=["logits", "target"],
                    output_names=["cross_entropy"],
                    binary=True,
                )
        return loss

    def get_metrics(self) -> Sequence[Metric]:
        """Returns a list of metrics to compute."""
        args = self.args
        if self._task == adni_hdf.Task.SURVIVAL_ANALYSIS:
            metrics = [Mean("partial_log_lik"), ConcordanceIndex("logits", "event", "time")]
        else:
            metrics = [
                Mean("cross_entropy"),
                Accuracy("logits", "target"),
                BalancedAccuracy(args.num_classes, "logits", "target"),
            ]
        return metrics

    def get_and_init_model(self) -> BaseModel:
        """Create and initialize a model."""
        model = self.get_model()
        self._init_model(model)
        return model

    @property
    def data_loader_target_names(self):
        if self._task == adni_hdf.Task.SURVIVAL_ANALYSIS:
            target_names = ["event", "time", "riskset"]
        else:
            target_names = ["target"]
        return target_names

    def _make_named_data_loader(
        self, dataset: Dataset, model_data_names: Sequence[str], is_training: bool = False
    ) -> NamedDataLoader:
        """Create a NamedDataLoader for the given dataset.

        Args:
          dataset (Dataset):
            The dataset to wrap.
          model_data_names (list of str):
            Should correspond to the names of the first `len(model_data_names)` outputs
            of the dataset and that are fed to model in a forward pass. The names
            of the targets used to compute the loss will be retrieved from :meth:`data_loader_target_names`.
          is_training (bool):
            Whether to enable training mode or not.
        """
        batch_size = self.args.batchsize
        if len(dataset) < batch_size:
            if is_training:
                raise RuntimeError(
                    "batch size ({:d}) cannot exceed dataset size ({:d})".format(batch_size, len(dataset))
                )
            else:
                batch_size = len(dataset)

        collate_fn = mesh_collate
        if self._task == adni_hdf.Task.SURVIVAL_ANALYSIS:
            collate_fn = partial(cox_collate_fn, data_collate=collate_fn)

        kwargs = {"batch_size": batch_size, "collate_fn": collate_fn, "shuffle": is_training, "drop_last": is_training}

        output_names = list(model_data_names) + self.data_loader_target_names
        loader = NamedDataLoader(dataset, output_names=output_names, **kwargs)
        return loader

    @abstractmethod
    def get_model(self) -> BaseModel:
        """Returns a model instance."""

    @abstractmethod
    def get_data(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Returns a data loader instance for training, evaluation, and testing, respectively."""


class ImageModelFactory(BaseModelFactory):
    """Factory for models taking 3D image volumes."""

    def get_data(self):
        args = self.args
        train_data, transform_kwargs = adni_hdf.get_image_dataset_for_train(
            args.train_data, self._task, args.shape, rescale=True
        )
        trainDataLoader = self._make_named_data_loader(train_data, ["image"], is_training=True)

        eval_data = adni_hdf.get_image_dataset_for_eval(args.val_data, self._task, transform_kwargs, args.shape)
        valDataLoader = self._make_named_data_loader(eval_data, ["image"])

        test_data = adni_hdf.get_image_dataset_for_eval(args.test_data, self._task, transform_kwargs, args.shape)
        testDataLoader = self._make_named_data_loader(test_data, ["image"])
        return trainDataLoader, valDataLoader, testDataLoader

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
        train_data, transform_kwargs = adni_hdf.get_point_cloud_dataset_for_train(args.train_data, self._task)
        trainDataLoader = self._make_named_data_loader(train_data, ["pointcloud"], is_training=True)

        eval_data = adni_hdf.get_point_cloud_dataset_for_eval(args.val_data, self._task, transform_kwargs)
        valDataLoader = self._make_named_data_loader(eval_data, ["pointcloud"])

        test_data = adni_hdf.get_point_cloud_dataset_for_eval(args.test_data, self._task, transform_kwargs)
        testDataLoader = self._make_named_data_loader(test_data, ["pointcloud"])

        return trainDataLoader, valDataLoader, testDataLoader

    def get_model(self):
        args = self.args
        use_batch_norm = args.batchsize > 1
        if args.discriminator_net == "pointnet":
            return point_networks.PointNet(args.num_points, args.num_classes, use_batch_norm)
        elif args.discriminator_net == "pointnet++":
            return point_networks.PointNet2ClsSsg(args.num_classes)
        else:
            raise ValueError("network {!r} is unsupported".format(args.discriminator_net))


class MeshModelFactory(BaseModelFactory):
    def get_data(self):
        args = self.args
        train_data, transform_kwargs, template = adni_hdf.get_mesh_dataset_for_train(args.train_data, self._task)
        trainDataLoader = self._make_named_data_loader(train_data, ["mesh"], is_training=True)

        eval_data = adni_hdf.get_mesh_dataset_for_eval(args.val_data, self._task, transform_kwargs)
        valDataLoader = self._make_named_data_loader(eval_data, ["mesh"])

        test_data = adni_hdf.get_mesh_dataset_for_eval(args.test_data, self._task, transform_kwargs)
        testDataLoader = self._make_named_data_loader(test_data, ["mesh"])

        self.template = template
        return trainDataLoader, valDataLoader, testDataLoader

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
