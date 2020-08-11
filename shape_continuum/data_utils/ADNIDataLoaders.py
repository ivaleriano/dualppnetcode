from __future__ import print_function

import random

import numpy as np
import pandas as pd
import torch

import nibabel as nib


class ADNI_base_loader:
    def __init__(
        self,
        csv_path,
        shape="pointcloud_free",
        task="cls",
        diagnosis=["CN", "AD"],
        shape1="vol_bb_wbg",
        shape2="vol_mask_free",
    ):
        self.df = pd.read_csv(csv_path)
        if shape == "multi_vol":
            self.files = self.df[shape1]
            self.files2 = self.df[shape2]
        else:
            self.files = self.df[shape]
        self.diagnosis = diagnosis
        self.shape = shape
        self.task = task

    def __getitem__(self, index):
        file = self.files[index]
        if self.shape == "pointcloud_free" or self.shape == "pointcloud_fsl":
            shape_tensor = self.load_pointcloud(file)
        elif self.shape == "mesh_fsl":
            shape_tensor = self.load_mesh(file)
        elif self.shape == "multi_vol":
            file1 = self.files[index]
            file2 = self.files2[index]
            shape_tensor_1 = self.load_volume(file1)
            shape_tensor_2 = self.load_volume(file2)
            shape_tensor = torch.cat((shape_tensor_1, shape_tensor_2), 0)
        else:
            shape_tensor = self.load_volume(file)

        if self.task == "surv":
            time = self.df.time[index]
            event = 1 if self.df.event[index] == "yes" else 0
            self.time = torch.FloatTensor([time])
            self.event = torch.Tensor([event])
            return shape_tensor, self.event, self.time
        else:
            dx0 = self.diagnosis.index(self.df.dx[index])
            return shape_tensor, torch.FloatTensor([dx0])

    def __len__(self):
        return len(self.files)

    def add_random_padding(c_vol, dim=64):
        shape = np.array(c_vol.shape)
        assert np.all(shape <= dim), "size of volume too big: {} > {}".format(dim, shape)

        pad_before = []
        pad_after = []
        borders = dim - shape
        for x in borders:
            idx = np.random.randint(x)
            pad_before.append(idx)
            pad_after.append(x - idx)

        pad_width = list(zip(pad_before, pad_after))
        pad_vol = np.pad(c_vol, pad_width, mode="constant")
        return pad_vol

    def load_pointcloud(self, file, n_points=1500):
        with open(file) as xyz_file:
            pointcloud0 = [p.replace("\n", "") for p in xyz_file.readlines()]
            pointcloud0 = random.sample(pointcloud0, n_points)
            pointcloud0 = [p.split() for p in pointcloud0]
            pc0_numpy = np.asarray(pointcloud0, dtype="float32")
            pc0_numpy -= np.mean(pc0_numpy, 0)
            pc0_numpy = pc0_numpy / np.max(np.linalg.norm(pc0_numpy, axis=1))
            pc0_tensor = torch.from_numpy(pc0_numpy)
        return pc0_tensor

    def load_mesh(self, file):

        return 0

    def load_volume(self, file):
        img = nib.load(file)
        vol = img.get_data()
        vol_tensor0 = torch.from_numpy(vol.astype("float32"))
        return torch.unsqueeze(vol_tensor0, 0)
