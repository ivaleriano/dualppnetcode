import argparse
import inspect
import json
import warnings
from abc import ABCMeta, abstractmethod
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Sequence, Tuple

import torch
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
    parser = argparse.ArgumentParser("Shape Continuum")

    g = parser.add_argument_group("General")
    g.add_argument("--task", choices=["clf", "surv"], default="clf", help="classification or survival analysis")
    g.add_argument("--workers", type=int, default=4, help="number of data loading workers")

    g = parser.add_argument_group("Training")
    g.add_argument("--epoch", type=int, default=200, help="number of epochs for training")
    g.add_argument("--pretrain", type=Path, help="whether use pretrain model")
    g.add_argument("--learning_rate", type=float, default=0.001, help="learning rate for training")
    g.add_argument("--decay_rate", type=float, default=1e-4, help="weight decay")
    g.add_argument("--optimizer", choices=["Adam", "SGD", "AdamW"], default="Adam", help="type of optimizer")

    g = parser.add_argument_group("Architecture")
    g.add_argument(
        "--discriminator_net", default="pointnet", help="which architecture to use for discriminator",
    )
    g.add_argument("--heterogeneous", action="store_true", default=False, help="training of a heterogeneous model")

    g = parser.add_argument_group("Data")
    g.add_argument(
        "--num_classes",
        type=int,
        default=3,
        help="The number of output units of the network. For binary classification, 1, "
        "for multi-class classification, the number of classes.",
    )
    g.add_argument(
        "--shape",
        default="pointcloud",
        help="which shape representation to use (pointcloud,mesh,mask,vol_with_bg,vol_without_bg)",
    )
    g.add_argument("--batchsize", type=int, default=20, help="input batch size")
    g.add_argument("--num_points", type=int, default=1500, help="number of points each point cloud has")
    g.add_argument("--train_data", type=Path, required=True, help="path to training dataset")
    g.add_argument("--val_data", type=Path, required=True, help="path to validation dataset")
    g.add_argument("--test_data", type=Path, required=True, help="path to testing dataset")

    g = parser.add_argument_group("Logging")
    g.add_argument(
        "--experiment_name",
        nargs="?",
        const=True,  # present, but not followed by a command-line argument
        default=False,  # not present
        help="Whether to give the experiment a particular name (default: current date and time).",
    )
    g.add_argument(
        "--tb_comment", action="store_true", default=False, help="any comment for storing on tensorboard",
    )
    g.add_argument(
        "--tensorboard", action="store_true", default=False, help="visualize training progress on tensorboard",
    )

    # normalization
    g = parser.add_argument_group("Data Normalization")
    g.add_argument(
        "--normalize_image",
        choices=["rescale", "standardize", "minmax"],
        default="rescale",
        help="Normalization function to apply to image data. Default rescale. Options: "
        "rescale -> divide by maximum value of datatype; "
        "standardize -> mean and stddev of whole dataset; "
        "minmax -> normalize to [0..1] with minimum and maximum per sample.",
    )
    g.add_argument(
        "--normalize_tabular",
        action="store_true",
        default=False,
        help="Normalize tabular data with mean and variance of whole dataset",
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
        if arguments.task == "clf":
            if arguments.num_classes == 1:
                warnings.warn(
                    "Data for classification tasks should consist of more than one label:"
                    f" num_classes = {arguments.num_classes}."
                )
            if arguments.num_classes > 2:
                self._task = adni_hdf.Task.MULTI_CLASSIFICATION
            else:
                self._task = adni_hdf.Task.BINARY_CLASSIFICATION
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
            if isinstance(args.experiment_name, str):
                experiment = args.experiment_name
            else:
                experiment = input("Enter a name for your experiment: ")
        else:
            experiment = f"shape_{args.shape}_network_{args.discriminator_net}"

        experiment_dir = base_dir / experiment / timestamp
        experiment_dir.mkdir(parents=True, exist_ok=False)

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
            optimizerD = torch.optim.SGD(params, lr=args.learning_rate, momentum=0.9)
        elif args.optimizer == "Adam":
            optimizerD = torch.optim.Adam(
                params, lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.decay_rate,
            )
        elif args.optimizer == "AdamW":
            optimizerD = torch.optim.AdamW(
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
            if self.args.num_classes > 2:
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


class HeterogeneousModelFactory(BaseModelFactory):
    """Factory for models taking 3D image volumes and tabular data."""

    def get_data(self):
        args = self.args
        rescale = args.normalize_image == "rescale"
        standardize = args.normalize_image == "standardize"
        minmax = args.normalize_image == "minmax"
        train_data, transform_kwargs, transform_tabular_kwargs = adni_hdf.get_heterogeneous_dataset_for_train(
            args.train_data,
            self._task,
            args.shape,
            rescale=rescale,
            standardize=standardize,
            minmax=minmax,
            transform_age=False,
            transform_education=False,
            normalize_tabular=args.normalize_tabular,
        )
        self.tabular_size = len(train_data.meta["tabular"]["columns"])
        trainDataLoader = self._make_named_data_loader(train_data, ["image", "tabular"], is_training=True)

        eval_data = adni_hdf.get_heterogeneous_dataset_for_eval(
            args.val_data, self._task, transform_kwargs, args.shape, transform_tabular_kwargs
        )
        valDataLoader = self._make_named_data_loader(eval_data, ["image", "tabular"])

        test_data = adni_hdf.get_heterogeneous_dataset_for_eval(
            args.test_data, self._task, transform_kwargs, args.shape, transform_tabular_kwargs
        )
        testDataLoader = self._make_named_data_loader(test_data, ["image", "tabular"])
        return trainDataLoader, valDataLoader, testDataLoader

    def get_model(self):
        args = self.args
        class_dict = {
            "resnet": vol_networks.HeterogeneousResNet,
            "concat1fc": vol_networks.ConcatHNN1FC,
            "concat2fc": vol_networks.ConcatHNN2FC,
            "mlpcatmlp": vol_networks.ConcatHNNMCM,
            "duanmu": vol_networks.InteractiveHNN,
            "film": vol_networks.FilmHNN,
            "zecatnet": vol_networks.ZeCatNet,
            "zenullnet": vol_networks.ZeNullNet,
            "zenunet": vol_networks.ZeNuNet,
        }
        assert args.discriminator_net in class_dict, "network {!r} is unsupported".format(args.discriminator_net)
        model_args = {
            "in_channels": 1,
            "n_outputs": (args.num_classes if args.num_classes > 2 else 1),
        }
        if "ndim_non_img" in list(inspect.signature(class_dict[args.discriminator_net]).parameters.keys()):
            model_args["ndim_non_img"] = self.tabular_size
        elif "filmblock_args" in list(inspect.signature(class_dict[args.discriminator_net]).parameters.keys()):
            model_args["filmblock_args"] = {"ndim_non_img": self.tabular_size}
        return class_dict[args.discriminator_net](**model_args)


class ImageModelFactory(BaseModelFactory):
    """Factory for models taking 3D image volumes."""

    def get_data(self):
        args = self.args
        rescale = args.normalize_image == "rescale"
        standardize = args.normalize_image == "standardize"
        minmax = args.normalize_image == "minmax"
        train_data, transform_kwargs = adni_hdf.get_image_dataset_for_train(
            args.train_data, self._task, args.shape, rescale=rescale, standardize=standardize, minmax=minmax,
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
        n_outputs = args.num_classes if args.num_classes > 2 else 1
        if args.discriminator_net == "resnet":
            return vol_networks.ResNet(in_channels, n_outputs)
        elif args.discriminator_net == "convnet":
            return vol_networks.Vol_classifier(in_channels, n_outputs)
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
        n_outputs = args.num_classes if args.num_classes > 2 else 1
        use_batch_norm = args.batchsize > 1
        if args.discriminator_net == "pointnet":
            return point_networks.PointNet(args.num_points, n_outputs, use_batch_norm)
        elif args.discriminator_net == "pointnet++":
            return point_networks.PointNet2ClsSsg(n_outputs)
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
            n_outputs = args.num_classes if args.num_classes > 2 else 1
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
                in_channels, out_channels, latent_channels, spiral_indices_list, down_transform_list, n_outputs
            )
        else:
            raise ValueError("network {!r} is unsupported".format(args.discriminator_net))


def get_factory(args: argparse.Namespace) -> BaseModelFactory:
    """Returns a factory depending on selected data type from command line arguments."""
    if args.heterogeneous:
        factory = HeterogeneousModelFactory(args)
    else:
        if args.shape == "pointcloud":
            factory = PointCloudModelFactory(args)
        elif args.shape.startswith("vol_") or args.shape == "mask":
            factory = ImageModelFactory(args)
        elif args.shape == "mesh":
            factory = MeshModelFactory(args)
        else:
            raise ValueError("shape {!r} is unsupported".format(args.shape))

    return factory
