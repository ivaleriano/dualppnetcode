import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
import numpy as np

def timeit(tag, t):
    print("{}: {}s".format(tag, time() - t))
    return time()

def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    max_batch = torch.max(batch_indices)
    min_idx = torch.min(idx)
    max_idx = torch.max(idx)
    new_points = points[batch_indices, idx, :]
    return new_points

# def decomp_points(group_indices,n_points):



def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, C]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx



def k_nearest(pointcloudA,centroids_xyz,n_samples=32):
    distsA = square_distance(pointcloudA,centroids_xyz)
    B, n_centroids, C = centroids_xyz.size()
    for i in range(n_centroids):
        dists2centroidA = distsA[:,:,i]
        indicesA = torch.argsort(dists2centroidA,1,descending=True)[:,:n_samples]
        sub_pcA = torch.unsqueeze(index_points(pointcloudA,indicesA),1)
        if i==0:
            group_xyz = sub_pcA
        else:
            group_xyz = torch.cat([group_xyz,sub_pcA],1)


    return group_xyz


def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False,returnidx = False):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, C]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, C]
        new_points: sampled points data, [B, 1, N, C+D]
    """
    B, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint) # [B, npoint, C]
    new_xyz = index_points(xyz, fps_idx)
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    elif returnidx:
        return new_xyz, new_points, fps_idx, idx
    else:
        return new_xyz, new_points




def sample_and_group_all(xyz, points):
    """
    Input:
        xyz: input points position data, [B, N, C]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, C]
        new_points: sampled points data, [B, 1, N, C+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


def sample_and_group_2pc(npoint, radius, nsample, xyz1, points, xyz2=None, returnfps=False, returnidx=False):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz1: input points position data, [B, N, C]
        points: input points data, [B, N, D]
    Return:
        new_xyz1: sampled points position data, [B, 1, C]
        new_points: sampled points data, [B, 1, N, C+D]
    """
    B, N, C = xyz1.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz1, npoint)  # [B, npoint, C]
    new_xyz1 = index_points(xyz1, fps_idx)
    idx = query_ball_point(radius, nsample, xyz1, new_xyz1)
    grouped_xyz1 = index_points(xyz1, idx)  # [B, npoint, nsample, C]
    grouped_xyz1_norm = grouped_xyz1 - new_xyz1.view(B, S, 1, C)
    if points is not None:
        grouped_points1 = index_points(points[:B, :, :], idx)
        new_points1 = torch.cat([grouped_xyz1_norm, grouped_points1], dim=-1)  # [B, npoint, nsample, C+D]
    else:
        new_points1 = grouped_xyz1_norm

    if xyz2 is not None:
        grouped_xyz2 = k_nearest(xyz2, new_xyz1, nsample)
        grouped_xyz2_norm = grouped_xyz2 - new_xyz1.view(B, S, 1, C)
    else:
        grouped_xyz2_norm = grouped_xyz1_norm

    if points is not None:
        grouped_points2 = index_points(points[B:, :, :], idx)
        new_points2 = torch.cat([grouped_xyz2_norm, grouped_points2], dim=-1)  # [B, npoint, nsample, C+D]
    else:
        new_points2 = grouped_xyz2_norm

    new_points = torch.cat([new_points1, new_points2], dim=0)
    if returnfps:
        return new_xyz1, new_points, grouped_xyz1, fps_idx
    elif returnidx:
        return new_xyz1, new_points, fps_idx, idx
    else:
        return new_xyz1, new_points



