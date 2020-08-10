from typing import List, Optional
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, SequentialSampler, BatchSampler, RandomSampler
from torch.utils.data.dataloader import default_collate


def make_riskset(time: np.ndarray) -> np.ndarray:
    # sort in descending order
    o = np.argsort(-np.squeeze(time), kind="mergesort")
    n_samples = time.shape[0]
    risk_set = np.zeros((n_samples, n_samples), dtype=np.uint8)
    for i_org, i_sort in enumerate(o):
        ti = time[i_sort]
        k = i_org
        while k < n_samples and ti == time[o[k]]:
            k += 1
        risk_set[i_sort, o[:k]] = 1
    return risk_set


def cox_collate_fn(batch: List[np.ndarray],
                   time_index: Optional[int] = -1) -> List[torch.Tensor]:
    """Sort samples in batch by observed time (descending)"""
    transposed_data = list(zip(*batch))
    y_time = np.array(transposed_data[time_index])
    # sort in descending order
    # o = np.argsort(-np.squeeze(y_time), kind="mergesort")
    data = []
    for j, b in enumerate(transposed_data):
        bt = [torch.as_tensor(v) for v in b]
        data.append(torch.stack(bt, 0))

    data.append(torch.tensor(make_riskset(y_time)))

    return data


def make_loader(dataset: Dataset,
                batch_size: Optional[int] = None,shuffle: Optional[bool] = True) -> DataLoader:
    # if not hasattr(dataset, "time"):
    #     raise ValueError("dataset must have a time attribute.")

    # n = dataset.time.shape[0]
    # batch_size = batch_size or n
    # if batch_size < n:
    #     sampler = RandomSampler(dataset)
    # else:
    #     sampler = SequentialSampler(dataset)
    if shuffle:
        sampler = RandomSampler(dataset)
    else:
        sampler = SequentialSampler(dataset)
    batch_sampler = BatchSampler(
        sampler,
        batch_size=batch_size,
        drop_last=False)

    dataloader = DataLoader(
        dataset,
        collate_fn=cox_collate_fn,
        batch_sampler=batch_sampler)
    return dataloader