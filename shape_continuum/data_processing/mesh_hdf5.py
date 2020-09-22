"""
Code adapted from Gong et al. SpiralNet++ Pytorch implementation
 https://github.com/sw-gong/spiralnet_plus
"""



import os

import h5py
import openmesh as om
import pandas as pd


def read_mesh(path):
    mesh = om.read_trimesh(path)
    face = mesh.face_vertex_indices().T.astype('long')
    x = mesh.points().astype('float32')
    return {"vertices":x,"faces":face}



mesh_dir = "/home/ignacio/NAS/Data_Neuro/ADNI3/FSL/meshes_and_pcs/L_Hipp/mesh/"
template_dir = os.path.join('./template', 'L_Hipp_template_pm_2922.ply')
mesh_dirs = os.listdir(mesh_dir)
csv_path_train = "./pcs_mesh_mask_vols_train_set_0.csv"
csv_path_test = "pcs_mesh_mask_vols_test_set_0.csv"

df = pd.read_csv(csv_path_train)
fps = df['mesh_fsl']
dxs = df['dx']
vscs = df['vsc']
ids = df['id']

with h5py.File("../../mesh_dataset_train.h5","w") as f:
    s = f.create_group('stats')
    u = s.create_group('Left-Hippocampus')
    ms = u.create_group('mesh')
    tp = ms.create_group('template')
    tmp_dict = read_mesh(template_dir)
    for key,value in tmp_dict.items():
        dset = tp.create_dataset(key,data=value,compression='gzip')
    for i,file in enumerate(fps):
        if os.path.isfile(file):
            image_uid = str(ids[i])
            print(image_uid)
            mesh_dict = read_mesh(file)
            g = f.create_group(image_uid)
            g.attrs['DX'] = dxs[i]
            g.attrs['RID'] = '0'
            g.attrs['VISCODE'] = vscs[i]
            r = g.create_group('Left-Hippocampus')
            h = r.create_group('mesh')
            for key,value in mesh_dict.items():
                dset = h.create_dataset(key,data=value,compression='gzip')

df = pd.read_csv(csv_path_test)
fps = df['mesh_fsl']
dxs = df['dx']
vscs = df['vsc']
ids = df['id']

with h5py.File("../../mesh_dataset_test.h5","w") as f:
    s = f.create_group('stats')
    u = s.create_group('Left-Hippocampus')
    ms = u.create_group('mesh')
    tp = ms.create_group('template')
    tmp_dict = read_mesh(template_dir)
    for key,value in tmp_dict.items():
        dset = tp.create_dataset(key,data=value,compression='gzip')
    for i,file in enumerate(fps):
        if os.path.isfile(file):
            image_uid = str(ids[i])
            print(image_uid)
            mesh_dict = read_mesh(file)
            g = f.create_group(image_uid)
            g.attrs['DX'] = dxs[i]
            g.attrs['RID'] = '0'
            g.attrs['VISCODE'] = vscs[i]
            r = g.create_group('Left-Hippocampus')
            h = r.create_group('mesh')
            for key,value in mesh_dict.items():
                dset = h.create_dataset(key,data=value,compression='gzip')
