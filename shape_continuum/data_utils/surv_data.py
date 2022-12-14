from typing import Any, List, Optional

import numpy as np
import torch
from torch.utils.data.dataloader import default_collate


def make_riskset(time: np.ndarray) -> np.ndarray:
    """Compute mask that represents each sample's risk set.

    Parameters
    ----------
    time : np.ndarray, shape=(n_samples,)
        Observed event time sorted in descending order.

    Returns
    -------
    risk_set : np.ndarray, shape=(n_samples, n_samples)
        Boolean matrix where the `i`-th row denotes the
        risk set of the `i`-th instance, i.e. the indices `j`
        for which the observer time `y_j >= y_i`.
    """
    assert time.ndim == 1, "expected 1D array"

    # sort in descending order
    o = np.argsort(-time, kind="mergesort")
    n_samples = len(time)
    risk_set = np.zeros((n_samples, n_samples), dtype=np.uint8)
    for i_org, i_sort in enumerate(o):
        ti = time[i_sort]
        k = i_org
        while k < n_samples and ti == time[o[k]]:
            k += 1
        risk_set[i_sort, o[:k]] = 1
    return risk_set


def cox_collate_fn(
    batch: List[Any], time_index: Optional[int] = -1, data_collate=default_collate
) -> List[torch.Tensor]:
    """Create risk set from batch."""
    transposed_data = list(zip(*batch))
    y_time = np.array(transposed_data[time_index])

    data = []
    for b in transposed_data:
        bt = data_collate(b)
        data.append(bt)

    data.append(torch.from_numpy(make_riskset(y_time)))

    return data
