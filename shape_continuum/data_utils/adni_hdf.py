from typing import Callable, Optional, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
from psbody.mesh import Mesh
from ..data_processing import mesh_sampling

DIAGNOSIS_CODES = {
    "CN": np.array(0, dtype=np.int64),
    "MCI": np.array(1, dtype=np.int64),
    "Dementia": np.array(2, dtype=np.int64),
}

AddChannelDim = transforms.Lambda(lambda x: x[np.newaxis])
LabelsToIndex = transforms.Lambda(lambda x: DIAGNOSIS_CODES[x])
NumpyToTensor = transforms.Lambda(torch.from_numpy)
AsTensor = transforms.Lambda(torch.as_tensor)


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
      transform (callable):
        Optional; A function that takes an individual data point
        (e.g. images, point clouds) and returns transformed version.
      target_transform (callable):
        Optional; A function that takes in a diagnosis (DX) label and
        transforms it.
    """

    def __init__(self, filename, dataset_name, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self._load(filename, dataset_name)


    def _load(self, filename, dataset_name, roi="Left-Hippocampus"):
        data = []
        targets = []
        visits = []
        meta = {}
        with h5py.File(filename, "r") as hf:
            for image_uid, g in hf.items():
                if image_uid == "stats":
                    continue
                visits.append((g.attrs["RID"], g.attrs["VISCODE"]))

                targets.append(g.attrs["DX"])
                img = g[roi][dataset_name][:]
                data.append(img)

            for key, value in hf["stats"][roi][dataset_name].items():
                if len(value.shape) > 0:
                    meta[key] = value[:]
                else:
                    meta[key] = np.array(value, dtype=value.dtype)

        self.data = data
        self.targets = targets
        self.visits = visits
        self.meta = meta

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        img, target = self.data[index], self.targets[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class HDF5DatasetMesh(HDF5Dataset):
    """
    HDF5Dataset Subclass specific for loading triangular meshes (based on code by Gong et al. https://github.com/sw-gong/spiralnet_plus)
    Args:
      filename (str):
        Path to HDF5 file.
      dataset_name (str):
        Name of the dataset to load (e.g. 'pointcloud', 'mask', 'vol_with_bg').
      transform (callable):
        Optional; A function that takes an individual data point
        (e.g. images, point clouds) and returns transformed version.
      target_transform (callable):
        Optional; A function that takes in a diagnosis (DX) label and
        transforms it.
      ds_factor (list[int]): down sampling factor in each pooling layer.

    """
    def __init__(self, filename, dataset_name, transform=None, target_transform=None, ds_factors=[4, 4]):
        self.transform = transform
        self.target_transform = target_transform
        self.ds_factors = ds_factors
        self._load(filename, dataset_name)



    def _load(self, filename, dataset_name, roi="Left-Hippocampus"):
        data = []
        targets = []
        visits = []
        meta = {}
        with h5py.File(filename, "r") as hf:
            #reading template from the hdf5
            mesh = Mesh(v=hf['stats'][roi][dataset_name]['template']['vertices'][:], f=hf['stats'][roi][dataset_name]['template']['faces'][:])
            _, A, D, U, F, V = mesh_sampling.generate_transform_matrices(mesh, self.ds_factors)
            self.template = {
                'vertices': V,
                'face': F,
                'adj': A,
                'down_transform': D,
                'up_transform': U
            }

            for image_uid, g in hf.items():
                if image_uid == "stats":
                    continue
                visits.append((g.attrs["RID"], g.attrs["VISCODE"]))


                targets.append(g.attrs["DX"])

                face = torch.from_numpy(g[roi][dataset_name]["faces"][:]).type(torch.long)
                face = face.T
                x = torch.tensor(g[roi][dataset_name]["vertices"][:].astype('float32'))

                edge_index = torch.cat([face[:2], face[1:], face[::2]], dim=1)
                edge_index = to_undirected(edge_index)
                y = g.attrs["DX"]
                y = self.target_transform(y)
                img = Data(x=x, y=y,edge_index=edge_index, face=face)
                data.append(img)
            for key, value in hf["stats"][roi][dataset_name].items():
                if key.startswith("max_dist"):
                    if len(value.shape) > 0:
                        meta[key] = value[:]
                    else:
                        meta[key] = np.array(value, dtype=value.dtype)
        self.data = data
        self.targets = targets
        self.visits = visits
        self.meta = meta



def _get_image_dataset_transform(
    dtype: np.dtype, rescale: bool, with_mean: Optional[np.ndarray], with_std: Optional[np.ndarray]
) -> Callable[[np.ndarray], np.ndarray]:
    img_transforms = []

    if rescale:
        max_val = np.array(np.iinfo(dtype).max, dtype=np.float32)
        img_transforms.append(transforms.Lambda(lambda x: x / max_val))

    if with_mean is not None or with_std is not None:
        if with_mean is None:
            with_mean = np.array(0.0, dtype=np.float32)
        if with_std is None:
            with_std = np.array(1.0, dtype=np.float32)
        img_transforms.append(transforms.Lambda(lambda x: (x - with_mean) / with_std))

    if len(img_transforms) == 0:
        img_transforms.append(transforms.Lambda(lambda x: x.astype(np.float32)))

    img_transforms.append(AddChannelDim)
    img_transforms.append(NumpyToTensor)

    return transforms.Compose(img_transforms)


def get_image_dataset_for_train(filename, dataset_name, rescale=False, standardize=False):
    """Loads 3D image volumes from HDF5 file and converts them to Tensors.

    No data augmentation is applied.

    Args:
      filename (str):
        Path to HDF5 file.
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
    target_transform = transforms.Compose([LabelsToIndex, AsTensor])

    ds = HDF5Dataset(filename, dataset_name, target_transform=target_transform)

    if dataset_name != "mask":
        if rescale and standardize:
            raise ValueError("only one of rescale and standardize can be True.")
    else:
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
        "with_mean": mean,
        "with_std": std,
    }

    ds.transform = _get_image_dataset_transform(**transform_kwargs)

    return ds, transform_kwargs


def get_image_dataset_for_eval(filename, transform_kwargs, dataset_name):
    """Loads 3D image volumes from HDF5 file and converts them to Tensors.

    Args:
      filename (str):
        Path to HDF5 file.
      transform_kwargs (dict):
        Arguments for image transform pipeline used during training as
        returned by :func:`get_image_dataset_for_train`.
      dataset_name (str):
        Name of the dataset to load (e.g. 'mask', 'vol_with_bg', 'vol_without_bg').

    Returns:
      dataset (HDF5Dataset):
        Dataset iterating over tuples of 4D ndarray and diagnosis.
    """
    target_transform = transforms.Compose([LabelsToIndex, AsTensor])

    ds = HDF5Dataset(filename, dataset_name, target_transform=target_transform)

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


def get_point_cloud_dataset_for_train(filename, dataset_name="pointcloud"):
    """Loads 3D point cloud from HDF5 file and converts them to Tensors.

    No data augmentation is applied.

    Args:
      filename (str):
        Path to HDF5 file.
      dataset_name (str):
        Optional; Name of the dataset to load.

    Returns:
      dataset (HDF5Dataset):
        Dataset iterating over tuples of 3D ndarray and diagnosis.
      transform_kwargs (dict):
        A dict with arguments used for creating point cloud transform pipeline.
    """
    target_transform = transforms.Compose([LabelsToIndex, AsTensor])

    ds = HDF5Dataset(filename, dataset_name, target_transform=target_transform)

    transform_kwargs = {
        "norm": ds.meta["max_dist_q95"].astype(np.float32),
        "transpose": True,
    }
    ds.transform = _get_point_cloud_transform(**transform_kwargs)

    return ds, transform_kwargs


def get_point_cloud_dataset_for_eval(filename, transform_kwargs, dataset_name="pointcloud"):
    """Loads 3D point cloud from HDF5 file and converts them to Tensors.

    Args:
      filename (str):
        Path to HDF5 file.
      transform_kwargs (dict):
        Arguments for point cloud transform pipeline used during training as
        returned by :func:`get_point_cloud_dataset_for_train`.
      dataset_name (str):
        Optional; Name of the dataset to load.

    Returns:
      dataset (HDF5Dataset):
        Dataset iterating over tuples of 3D ndarray and diagnosis.
    """
    target_transform = transforms.Compose([LabelsToIndex, AsTensor])

    ds = HDF5Dataset(filename, dataset_name, target_transform=target_transform)

    ds.transform = _get_point_cloud_transform(**transform_kwargs)

    return ds

def _get_mesh_transform():
    mesh_transforms = []

    #ToDo: look for the transforms needed for the mesh


    return transforms.Compose(mesh_transforms)



def get_mesh_dataset_for_train(filename, dataset_name="mesh"):
    """Loads 3D point cloud from HDF5 file and converts them to Tensors.

    No data augmentation is applied.

    Args:
      filename (str):
        Path to HDF5 file.
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
    target_transform = transforms.Compose([LabelsToIndex, AsTensor])

    ds = HDF5DatasetMesh(filename, dataset_name, target_transform=target_transform)
    template = ds.template
    transform_kwargs = {}
    ds.transform = _get_mesh_transform(**transform_kwargs)

    return ds,transform_kwargs,template

def get_mesh_dataset_for_eval(filename, transform_kwargs, dataset_name="mesh"):
    """Loads 3D point cloud from HDF5 file and converts them to Tensors.

    Args:
      filename (str):
        Path to HDF5 file.
      transform_kwargs (dict):
        Arguments for mesh transform pipeline used during training as
        returned by :func:`get_mesh_dataset_for_train`.
      dataset_name (str):
        Optional; Name of the dataset to load.

    Returns:
      dataset (HDF5Dataset):
        Dataset iterating over tuples of 3D ndarray and diagnosis.
    """
    target_transform = transforms.Compose([LabelsToIndex, AsTensor])

    ds = HDF5DatasetMesh(filename, dataset_name, target_transform=target_transform)

    ds.transform = _get_mesh_transform(**transform_kwargs)

    return ds
