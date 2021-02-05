import enum
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import h5py
import numpy as np
import torch
from psbody.mesh import Mesh
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
from torchvision import transforms

from ..data_processing import mesh_sampling

DIAGNOSIS_CODES_BINARY = {
    "CN": np.array(0, dtype=np.int64),
    "Dementia": np.array(1, dtype=np.int64),
}
DIAGNOSIS_CODES_MULTICLASS = {
    "CN": np.array(0, dtype=np.int64),
    "MCI": np.array(1, dtype=np.int64),
    "Dementia": np.array(2, dtype=np.int64),
}
PROGRESSION_STATUS = {
    "no": np.array([0], dtype=np.uint8),
    "yes": np.array([1], dtype=np.uint8),
}

DataTransformFn = Callable[[Union[np.ndarray, torch.Tensor]], Union[np.ndarray, torch.Tensor]]
TargetTransformFn = Callable[[str], np.ndarray]


def AddChannelDim(img: np.ndarray) -> np.ndarray:
    return img[np.newaxis]


def NumpyToTensor(img: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(img)


def AsTensor(img: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(img)


def AsFloat32(img: np.ndarray) -> np.ndarray:
    return img.astype(np.float32)


def MinmaxRescaling(x: np.ndarray) -> np.ndarray:
    min_val = x.min()

    return (x - min_val) / (x.max() - min_val)


class Task(enum.Enum):
    BINARY_CLASSIFICATION = (["DX"], DIAGNOSIS_CODES_BINARY)
    MULTI_CLASSIFICATION = (["DX"], DIAGNOSIS_CODES_MULTICLASS)
    SURVIVAL_ANALYSIS = (["event", "time"], PROGRESSION_STATUS)

    def __init__(self, target_labels: Sequence[str], target2code: Dict[str, np.ndarray]):
        self._target_labels = target_labels
        self._target2code = target2code

    @property
    def labels(self) -> Sequence[str]:
        """The names of attributes storing labels."""
        return self._target_labels

    @property
    def label_transform(self) -> TargetTransformFn:
        """The transform function to convert labels to numbers."""
        return transforms.Lambda(lambda x: self._target2code[x])


class HDF5Dataset(Dataset):
    """Dataset to load ADNI data from HDF5 file.

    The HDF5 file has 3 levels:
      1. Image UID
      2. Region of interest
      3. Dataset

    This class only considers the Left Hippocampus ROI.

    Each Image UID is associated with a DX attribute
    denoting the diagnosis.

    Args:
      filename (str):
        Path to HDF5 file.
      dataset_name (str):
        Name of the dataset to load (e.g. 'pointcloud', 'mask', 'vol_with_bg').
      target_labels (list of str):
        The names of attributes to retrieve as labels.
      transform (callable):
        Optional; A function that takes an individual data point
        (e.g. images, point clouds) and returns transformed version.
      target_transform (dict mapping str to callable):
        Optional; The key should be the name of a label attribute passed as `target_labels`,
        the value a function that takes in a label and transforms it.
    """

    def __init__(
        self,
        filename: str,
        dataset_name: str,
        target_labels: Sequence[str],
        transform: Optional[DataTransformFn] = None,
        target_transform: Optional[Dict[str, TargetTransformFn]] = None,
    ) -> None:
        self.target_labels = target_labels
        self.transform = transform
        self.target_transform = target_transform
        self._load(filename, dataset_name)

    def _load(self, filename, dataset_name, roi="Left-Hippocampus"):
        data = []
        targets = {k: [] for k in self.target_labels}
        visits = []
        with h5py.File(filename, "r") as hf:
            for image_uid, g in hf.items():
                if image_uid == "stats":
                    continue
                visits.append((g.attrs["RID"], g.attrs["VISCODE"]))

                for label in self.target_labels:
                    targets[label].append(g.attrs[label])

                data.append(self._get_data(g[roi][dataset_name]))

            meta = self._get_meta_data(hf["stats"][roi][dataset_name])

        self.data = data
        self.targets = targets
        self.visits = visits
        self.meta = meta

    def _get_data(self, data: Union[h5py.Dataset, h5py.Group]) -> Any:
        img = data[:]
        return img

    def _get_meta_data(self, stats: h5py.Group) -> Dict[str, Any]:
        meta = {}
        for key, value in stats.items():
            if len(value.shape) > 0:
                meta[key] = value[:]
            else:
                meta[key] = np.array(value, dtype=value.dtype)
        return meta

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Sequence[np.ndarray]:
        img = self.data[index]
        if self.transform is not None:
            img = self.transform(img)

        data_point = [img]
        for label in self.target_labels:
            target = self.targets[label][index]
            if self.target_transform is not None:
                target = self.target_transform[label](target)
            data_point.append(target)

        return tuple(data_point)


class HDF5DatasetHeterogeneous(HDF5Dataset):
    """
    HDF5Dataset Subclass specific for heterogeneous data loading

    Args:
      filename (str):
        Path to HDF5 file.
      dataset_name (str):
        Name of the dataset to load (e.g. 'pointcloud', 'mask', 'vol_with_bg').
      target_labels (list of str):
        The names of attributes to retrieve as labels.
      transform (callable):
        Optional; A function that takes an image of an individual data point
        (e.g. images, point clouds) and returns transformed version.
      target_transform (dict mapping str to callable):
        Optional; The key should be the name of a label attribute passed as `target_labels`,
        the value a function that takes in a label and transforms it.
      tabular_transform (callable):
        Optional; A function that takes tabular data of an individual data point
        and returns transformed version.
    """

    def __init__(
        self,
        filename: str,
        dataset_name: str,
        target_labels: Sequence[str],
        transform: Optional[DataTransformFn] = None,
        target_transform: Optional[Dict[str, TargetTransformFn]] = None,
        tabular_transform: Optional[TargetTransformFn] = None,
    ) -> None:
        self.target_labels = target_labels
        self.transform = transform
        self.target_transform = target_transform
        self.tabular_transform = tabular_transform
        self._load(filename, dataset_name)

    # overrides
    def _get_data(self, data: Union[h5py.Dataset, h5py.Group]) -> Any:
        img = super()._get_data(data)
        tabular = super()._get_data(data.parent.parent["tabular"])
        return img, tabular

    # overrides
    def _get_meta_data(self, stats: h5py.Group) -> Dict[str, Any]:
        meta = super()._get_meta_data(stats)

        meta["tabular"] = {}
        for key, value in stats.parent.parent["tabular"].items():
            meta["tabular"][key] = value[:]
        return meta

    # overrides
    def __getitem__(self, index: int) -> Sequence[np.ndarray]:
        img, tabular = self.data[index]
        if self.transform is not None:
            img = self.transform(img)
        if self.tabular_transform is not None:
            tabular = self.tabular_transform(tabular)

        data_point = [img, tabular]
        for label in self.target_labels:
            target = self.targets[label][index]
            if self.target_transform is not None:
                target = self.target_transform[label](target)
            data_point.append(target)

        return tuple(data_point)


class HDF5DatasetMesh(HDF5Dataset):
    """
    HDF5Dataset Subclass specific for loading triangular meshes

    (based on code by Gong et al. https://github.com/sw-gong/spiralnet_plus)

    Args:
      filename (str):
        Path to HDF5 file.
      dataset_name (str):
        Name of the dataset to load (e.g. 'pointcloud', 'mask', 'vol_with_bg').
      target_labels (list of str):
        The names of attributes to retrieve as labels.
      transform (callable):
        Optional; A function that takes an individual data point
        (e.g. images, point clouds) and returns transformed version.
      target_transform (callable):
        Optional; A function that takes in a diagnosis (DX) label and
        transforms it.
      ds_factor (list[int]): down sampling factor in each pooling layer.
    """

    def __init__(self, filename, dataset_name, target_labels, transform=None, target_transform=None, ds_factors=[4, 4]):
        self.ds_factors = ds_factors
        super().__init__(
            filename=filename,
            dataset_name=dataset_name,
            target_labels=target_labels,
            transform=transform,
            target_transform=target_transform,
        )

    def _get_data(self, data: Union[h5py.Dataset, h5py.Group]) -> Any:
        face = torch.from_numpy(data["faces"][:]).type(torch.long)
        face = face.T
        x = torch.tensor(data["vertices"][:].astype(np.float32))

        edge_index = torch.cat([face[:2], face[1:], face[::2]], dim=1)
        edge_index = to_undirected(edge_index)

        img = Data(x=x, edge_index=edge_index, face=face)
        return img

    def _get_meta_data(self, stats: h5py.Group) -> Dict[str, Any]:
        meta = {}
        for key, value in stats.items():
            if key.startswith("max_dist"):
                if len(value.shape) > 0:
                    meta[key] = value[:]
                else:
                    meta[key] = np.array(value, dtype=value.dtype)

        # reading template from the hdf5
        mesh = Mesh(v=stats["template"]["vertices"][:], f=stats["template"]["faces"][:])
        _, A, D, U, F, V = mesh_sampling.generate_transform_matrices(mesh, self.ds_factors)
        self.template = {"vertices": V, "face": F, "adj": A, "down_transform": D, "up_transform": U}

        return meta


def _get_image_dataset_transform(
    dtype: np.dtype,
    rescale: bool,
    with_mean: Optional[np.ndarray],
    with_std: Optional[np.ndarray],
    minmax_rescale: bool = False,
) -> Callable[[np.ndarray], np.ndarray]:
    img_transforms = []

    img_transforms.append(AsFloat32)

    if rescale:
        max_val = np.array(np.iinfo(dtype).max, dtype=np.float32)
        img_transforms.append(transforms.Lambda(lambda x: x / max_val))

    if minmax_rescale:
        img_transforms.append(MinmaxRescaling)

    if with_mean is not None or with_std is not None:
        if with_mean is None:
            with_mean = np.array(0.0, dtype=np.float32)
        if with_std is None:
            with_std = np.array(1.0, dtype=np.float32)
        img_transforms.append(transforms.Lambda(lambda x: (x - with_mean) / with_std))

    if len(img_transforms) == 0:
        img_transforms.append(AsFloat32)

    img_transforms.append(AddChannelDim)
    img_transforms.append(NumpyToTensor)

    return transforms.Compose(img_transforms)


def _get_target_transform(task: Task) -> TargetTransformFn:
    if task in {Task.BINARY_CLASSIFICATION, Task.MULTI_CLASSIFICATION}:
        target_transform = {"DX": transforms.Compose([task.label_transform, AsTensor])}
    elif task == Task.SURVIVAL_ANALYSIS:
        target_transform = dict(
            zip(task.labels, (transforms.Compose([task.label_transform, AsTensor]), transforms.Compose([AsTensor])))
        )
    else:
        raise ValueError("{!r} task not supported".format(task))
    return target_transform


@dataclass
class NormContainer:
    name: str
    index: int
    mean: float
    stddev: float
    coded_as_missing: bool = False


def _transform_tabular(x: np.ndarray, indices: List[NormContainer]) -> np.ndarray:
    # >0: Biomarkers that were not acquired at a visit are 0 and their 'missing' variable is 1 -> don't normalize
    for feature_stats in indices:
        if (not feature_stats.coded_as_missing) or (x[feature_stats.index] > 0):
            x[feature_stats.index] = (x[feature_stats.index] - feature_stats.mean) / feature_stats.stddev
    return x


def _get_tabular_dataset_transform(
    transform_age: bool,
    transform_education: bool,
    feature_names: np.ndarray,
    with_mean: Optional[np.ndarray],
    with_std: Optional[np.ndarray],
) -> Callable[[np.ndarray], np.ndarray]:

    tabular_transforms = []

    tabular_transforms.append(AsFloat32)

    if transform_age:
        raise ValueError("transform_age not yet supported!")

    if transform_education:
        raise ValueError("transform_education not yet supported!")

    if with_mean is not None or with_std is not None:
        if with_mean is None:
            with_mean = np.zeros(len(feature_names), dtype=np.float32)
        if with_std is None:
            with_std = np.ones(len(feature_names), dtype=np.float32)
        # save indices as key in dict with
        norms = []
        missing_codes = {
            "APOE4": "C(APOE4_MISSING)[T.1]",
            "ABETA": "C(ABETA_MISSING)[T.1]",
            "TAU": "C(TAU_MISSING)[T.1]",
            "PTAU": "C(PTAU_MISSING)[T.1]",
            "FDG": "C(FDG_MISSING)[T.1]",
            "AV45": "C(AV45_MISSING)[T.1]",
        }
        for i, el in enumerate(feature_names):
            if "MISSING" not in el:  # normalize everything but 'MISSING' variables
                as_missing = el in missing_codes and missing_codes[el] in feature_names
                norms.append(NormContainer(el, i, with_mean[i], with_std[i], as_missing))
        transform_fn = partial(_transform_tabular, indices=norms)
        tabular_transforms.append(transforms.Lambda(transform_fn))

    tabular_transforms.append(NumpyToTensor)

    return transforms.Compose(tabular_transforms)


def get_heterogeneous_dataset_for_train(
    filename,
    task,
    dataset_name,
    rescale=False,
    standardize=False,
    minmax=False,
    transform_age=False,
    transform_education=False,
    normalize_tabular=False,
):
    """Loads 3D image volumes and tabular data from HDF5 file and converts them to Tensors.

    No data augmentation is applied.

    Args:
      filename (str):
        Path to HDF5 file.
      task (Task):
        Define the target label for given task.
      dataset_name (str):
        Name of the dataset to load (e.g. 'mask', 'vol_with_bg', 'vol_without_bg').
      rescale (bool):
        Optional; Whether to rescale intensities to 0-1 by dividing by maximum
        value a voxel can hold (e.g. 255 if voxels are bytes).
      standardize (bool):
        Optional; Whether to subtract the voxel-wise mean and divide by the
        voxel-wise standard deviation.
      minmax (bool):
        Optional; Wether to rescale the image volume to 0-1 with MinMax rescaling.
      transform_age (bool):
        Optional; Whether to transform tabular feature age with
        natural cubic spline with four degrees of freedom.
      transform_education (bool):
        Optional; Wether to transform tabular feature education with polynomial contrast codes.
      normalize_tabular (bool):
        Optional; Wether to normalize tabular data with mean and stddev per feature.

    Returns:
      dataset (HDF5Dataset):
        Dataset iterating over tuples of 3D ndarray and diagnosis.
      transform_kwargs (dict):
        A dict with arguments used for creating image transform pipeline.
      transform_tabular_kwargs (dict):
        A dict with arguments used for creating tabular transform pipeline.

    Raises:
      ValueError:
        If both rescale and standardize are True.
    """
    target_transform = _get_target_transform(task)

    ds = HDF5DatasetHeterogeneous(filename, dataset_name, task.labels, target_transform=target_transform)

    if dataset_name != "mask":
        if np.count_nonzero([rescale, standardize, minmax]) > 1:
            raise ValueError("only one of rescale, standardize and minmax can be True.")
    else:
        minmax = False
        rescale = False
        standardize = False

    if standardize:
        mean = ds.meta["mean"].astype(np.float32)
        std = ds.meta["stddev"].astype(np.float32)
    else:
        mean = None
        std = None

    transform_img_kwargs = {
        "dtype": ds.data[0][0].dtype,
        "rescale": rescale,
        "minmax_rescale": minmax,
        "with_mean": mean,
        "with_std": std,
    }
    ds.transform = _get_image_dataset_transform(**transform_img_kwargs)

    if normalize_tabular and (transform_age or transform_education):
        raise ValueError("only one of normalizing of transformation (age|education) can be True")
    if normalize_tabular:
        tab_mean = ds.meta["tabular"]["mean"].astype(np.float32)
        tab_std = ds.meta["tabular"]["stddev"].astype(np.float32)
    else:
        tab_mean = None
        tab_std = None

    transform_tabular_kwargs = {
        "transform_age": transform_age,
        "transform_education": transform_education,
        "feature_names": ds.meta["tabular"]["columns"],
        "with_mean": tab_mean,
        "with_std": tab_std,
    }
    ds.tabular_transform = _get_tabular_dataset_transform(**transform_tabular_kwargs)

    return ds, transform_img_kwargs, transform_tabular_kwargs


def get_heterogeneous_dataset_for_eval(filename, task, transform_kwargs, dataset_name, transform_tabular_kwargs):
    """Loads 3D image volumes from HDF5 file and converts them to Tensors.

    Args:
      filename (str):
        Path to HDF5 file.
      task (Task):
        Define the target label for given task.
      transform_kwargs (dict):
        Arguments for image transform pipeline used during training as
        returned by :func:`get_heterogeneous_dataset_for_train`.
      dataset_name (str):
        Name of the dataset to load (e.g. 'mask', 'vol_with_bg', 'vol_without_bg').
      transform_tabular_kwargs (dict):
        Arguments for tabular transform pipeline used during training as
        returned by :func:`get_heterogeneous_dataset_for_train`.

    Returns:
      dataset (HDF5Dataset):
        Dataset iterating over tuples of 4D ndarray and diagnosis.
    """
    target_transform = _get_target_transform(task)

    ds = HDF5DatasetHeterogeneous(filename, dataset_name, task.labels, target_transform=target_transform)

    ds.transform = _get_image_dataset_transform(**transform_kwargs)
    ds.tabular_transform = _get_tabular_dataset_transform(**transform_tabular_kwargs)

    return ds


def get_image_dataset_for_train(filename, task, dataset_name, rescale=False, standardize=False, minmax=False):
    """Loads 3D image volumes from HDF5 file and converts them to Tensors.

    No data augmentation is applied.

    Args:
      filename (str):
        Path to HDF5 file.
      task (Task):
        Define the target label for given task.
      dataset_name (str):
        Name of the dataset to load (e.g. 'mask', 'vol_with_bg', 'vol_without_bg').
      rescale (bool):
        Optional; Whether to rescale intensities to 0-1 by dividing by maximum
        value a voxel can hold (e.g. 255 if voxels are bytes).
      standardize (bool):
        Optional; Whether to subtract the voxel-wise mean and divide by the
        voxel-wise standard deviation.

    Returns:
      dataset (HDF5Dataset):
        Dataset iterating over tuples of 3D ndarray and diagnosis.
      transform_kwargs (dict):
        A dict with arguments used for creating image transform pipeline.

    Raises:
      ValueError:
        If both rescale and standardize are True.
    """
    target_transform = _get_target_transform(task)

    ds = HDF5Dataset(filename, dataset_name, task.labels, target_transform=target_transform)

    if dataset_name != "mask":
        if np.count_nonzero([rescale, standardize, minmax]) > 1:
            raise ValueError("only one of rescale, standardize and minmax can be True.")
    else:
        minmax = False
        rescale = False
        standardize = False

    if standardize:
        mean = ds.meta["mean"].astype(np.float32)
        std = ds.meta["stddev"].astype(np.float32)
    else:
        mean = None
        std = None

    transform_kwargs = {
        "dtype": ds.data[0].dtype,
        "rescale": rescale,
        "minmax_rescale": minmax,
        "with_mean": mean,
        "with_std": std,
    }

    ds.transform = _get_image_dataset_transform(**transform_kwargs)

    return ds, transform_kwargs


def get_image_dataset_for_eval(filename, task, transform_kwargs, dataset_name):
    """Loads 3D image volumes from HDF5 file and converts them to Tensors.

    Args:
      filename (str):
        Path to HDF5 file.
      task (Task):
        Define the target label for given task.
      transform_kwargs (dict):
        Arguments for image transform pipeline used during training as
        returned by :func:`get_image_dataset_for_train`.
      dataset_name (str):
        Name of the dataset to load (e.g. 'mask', 'vol_with_bg', 'vol_without_bg').

    Returns:
      dataset (HDF5Dataset):
        Dataset iterating over tuples of 4D ndarray and diagnosis.
    """
    target_transform = _get_target_transform(task)

    ds = HDF5Dataset(filename, dataset_name, task.labels, target_transform=target_transform)

    ds.transform = _get_image_dataset_transform(**transform_kwargs)

    return ds


def _get_point_cloud_transform(norm: Optional[float], transpose: bool):
    pc_transforms = []

    if norm is not None:
        pc_transforms.append(transforms.Lambda(lambda x: x / norm))
    if transpose:
        pc_transforms.append(transforms.Lambda(lambda x: x.transpose(1, 0)))
    pc_transforms.append(NumpyToTensor)

    return transforms.Compose(pc_transforms)


def get_point_cloud_dataset_for_train(filename, task, dataset_name="pointcloud"):
    """Loads 3D point cloud from HDF5 file and converts them to Tensors.

    No data augmentation is applied.

    Args:
      filename (str):
        Path to HDF5 file.
      task (Task):
        Define the target label for given task.
      dataset_name (str):
        Optional; Name of the dataset to load.

    Returns:
      dataset (HDF5Dataset):
        Dataset iterating over tuples of 3D ndarray and diagnosis.
      transform_kwargs (dict):
        A dict with arguments used for creating point cloud transform pipeline.
    """
    target_transform = _get_target_transform(task)

    ds = HDF5Dataset(filename, dataset_name, task.labels, target_transform=target_transform)

    transform_kwargs = {
        "norm": ds.meta["max_dist_q95"].astype(np.float32),
        "transpose": True,
    }
    ds.transform = _get_point_cloud_transform(**transform_kwargs)

    return ds, transform_kwargs


def get_point_cloud_dataset_for_eval(filename, task, transform_kwargs, dataset_name="pointcloud"):
    """Loads 3D point cloud from HDF5 file and converts them to Tensors.

    Args:
      filename (str):
        Path to HDF5 file.
      task (Task):
        Define the target label for given task.
      transform_kwargs (dict):
        Arguments for point cloud transform pipeline used during training as
        returned by :func:`get_point_cloud_dataset_for_train`.
      dataset_name (str):
        Optional; Name of the dataset to load.

    Returns:
      dataset (HDF5Dataset):
        Dataset iterating over tuples of 3D ndarray and diagnosis.
    """
    target_transform = _get_target_transform(task)

    ds = HDF5Dataset(filename, dataset_name, task.labels, target_transform=target_transform)

    ds.transform = _get_point_cloud_transform(**transform_kwargs)

    return ds


def _get_mesh_transform():
    mesh_transforms = []

    # TODO: look for the transforms needed for the mesh

    return transforms.Compose(mesh_transforms)


def get_mesh_dataset_for_train(filename, task, dataset_name="mesh"):
    """Loads 3D point cloud from HDF5 file and converts them to Tensors.

    No data augmentation is applied.

    Args:
      filename (str):
        Path to HDF5 file.
      task (Task):
        Define the target label for given task.
      dataset_name (str):
        Optional; Name of the dataset to load.

    Returns:
      dataset (HDF5Dataset):
        Dataset iterating over tuples of 3D ndarray and diagnosis.
      transform_kwargs (dict):
        A dict with arguments used for creating mesh transform pipeline.
      template (dict):
        dataset template (vertices and faces) as well as the down_sampling and up_sampling transform matrices
    """
    target_transform = _get_target_transform(task)

    ds = HDF5DatasetMesh(filename, dataset_name, task.labels, target_transform=target_transform)
    template = ds.template
    transform_kwargs = {}
    ds.transform = _get_mesh_transform(**transform_kwargs)

    return ds, transform_kwargs, template


def get_mesh_dataset_for_eval(filename, task, transform_kwargs, dataset_name="mesh"):
    """Loads 3D point cloud from HDF5 file and converts them to Tensors.

    Args:
      filename (str):
        Path to HDF5 file.
      task (Task):
        Define the target label for given task.
      transform_kwargs (dict):
        Arguments for mesh transform pipeline used during training as
        returned by :func:`get_mesh_dataset_for_train`.
      dataset_name (str):
        Optional; Name of the dataset to load.

    Returns:
      dataset (HDF5Dataset):
        Dataset iterating over tuples of 3D ndarray and diagnosis.
    """
    target_transform = _get_target_transform(task)

    ds = HDF5DatasetMesh(filename, dataset_name, task.labels, target_transform=target_transform)

    ds.transform = _get_mesh_transform(**transform_kwargs)

    return ds
