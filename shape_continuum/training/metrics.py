from abc import ABCMeta, abstractmethod
from typing import Dict

import numpy as np
from sksurv.metrics import concordance_index_censored
from torch import Tensor


class Metric(metaclass=ABCMeta):
    """Base class for metrics."""

    @abstractmethod
    def reset(self) -> None:
        """Reset the metric to its initial state."""

    @abstractmethod
    def update(self, inputs: Dict[str, Tensor], outputs: Dict[str, Tensor]) -> None:
        """Accumulates statistics for the metric.

        Args:
          inputs:
            A Dict with the batch's data obtained from the DataLoader and passed to the model.
          outputs:
            A Dict with the outputs returned by the model for the current batch.
        """

    @abstractmethod
    def values(self) -> Dict[str, float]:
        """Computes and returns the metrics.

        Returns (dict):
          A Dict mapping a metric's name to its value.
        """


class Mean(Metric):
    """Computes the mean of the tensor with the given name.

    Args:
      tensor_name (str):
        Name of tensor to compute the mean for. A tensor of
        the given name must be returned by the model's forward method.
        Only scalar tensors are supported.
    """

    def __init__(self, tensor_name: str) -> None:
        self._tensor_name = tensor_name
        self._value = None

    def values(self) -> Dict[str, float]:
        return {f"{self._tensor_name}/mean": self._value}

    def reset(self) -> None:
        self._value = 0
        self._total = 0

    def update(self, inputs: Dict[str, Tensor], outputs: Dict[str, Tensor]) -> None:
        value = outputs[self._tensor_name].detach().cpu()
        assert value.dim() == 0, "tensor must be scalar"
        self._total += 1
        self._value += (value.item() - self._value) / self._total


class Accuracy(Metric):
    """Calculates how often predictions matches labels.

    Args:
      prediction (str):
        Name of tensor with the predicted value. A tensor of
        the given name must be returned by the model's forward method.
      target (str):
        Name of the tensor witht the  ground truth. A tensor of
        the given name must be returned by the data loader.
    """

    def __init__(self, prediction: str, target: str) -> None:
        self._prediction = prediction
        self._target = target
        self._value = None

    def values(self) -> Dict[str, float]:
        value = self._correct / self._total
        return {"accuracy": value}

    def reset(self) -> None:
        self._correct = 0
        self._total = 0

    def update(self, inputs: Dict[str, Tensor], outputs: Dict[str, Tensor]) -> None:
        target_tensor = inputs[self._target].detach().cpu()

        pred = outputs[self._prediction].detach().cpu()
        class_id = pred.argmax(dim=1)
        self._correct += (class_id == target_tensor).sum().item()
        self._total += pred.size()[0]


class BalancedAccuracy(Metric):
    """Calculates the balanced accuracy.

    It is defined as the average of recall obtained on each class.

    Args:
      n_classes (int):
        Number of classes in the dataset.
      prediction (str):
        Name of tensor with the predicted value. A tensor of
        the given name must be returned by the model's forward method.
      target (str):
        Name of the tensor witht the  ground truth. A tensor of
        the given name must be returned by the data loader.
    """

    def __init__(self, n_classes: int, prediction: str, target: str) -> None:
        self._n_classes = n_classes
        self._prediction = prediction
        self._target = target

    def values(self) -> Dict[str, float]:
        value = np.mean(self._correct / self._total)
        return {"balanced_accuracy": value}

    def reset(self) -> None:
        # per-class counts
        self._correct = np.zeros(self._n_classes, dtype=int)
        self._total = np.zeros(self._n_classes, dtype=int)

    def update(self, inputs: Dict[str, Tensor], outputs: Dict[str, Tensor]) -> None:
        pred = outputs[self._prediction].detach().cpu()
        class_id = pred.argmax(dim=1).numpy()
        target_tensor = inputs[self._target].detach().cpu().numpy().astype(class_id.dtype)

        is_correct = class_id == target_tensor
        classes, counts = np.unique(target_tensor, return_counts=True)
        for i, c in zip(classes, counts):
            self._total[i] += c
            self._correct[i] += is_correct[target_tensor == i].sum()


class ConcordanceIndex(Metric):
    """Computes concordance index across one epoch."""

    METRIC_NAMES = ("cindex", "concordant", "discordant", "tied_risk")

    def __init__(self, prediction: str, target_event: str, target_time: str) -> None:
        self._prediction = prediction
        self._target_event = target_event
        self._target_time = target_time

    def reset(self) -> None:
        """Clear the buffer of collected values."""
        self._data = {"target_time": [], "target_event": [], "prediction": []}

    def is_best(self) -> bool:
        return False

    def _append_time(self, inputs: Dict[str, Tensor]) -> None:
        arr = inputs[self._target_time].detach().cpu().numpy()
        self._data["target_time"].append(arr)

    def _append_event(self, inputs: Dict[str, Tensor]) -> None:
        arr = inputs[self._target_event].squeeze(1).detach().cpu().numpy()
        self._data["target_event"].append(arr)

    def _append_prediction(self, outputs: Dict[str, Tensor]) -> None:
        arr = outputs[self._prediction].squeeze(1).detach().cpu().numpy()
        self._data["prediction"].append(arr)

    def update(self, inputs: Dict[str, Tensor], outputs: Dict[str, Tensor]) -> None:
        """Collect observed time, event indicator and predictions for a batch."""
        self._append_event(inputs)
        self._append_time(inputs)
        self._append_prediction(outputs)

    def values(self) -> Dict[str, float]:
        """Computes the concordance index across collected values.

        Returns:
            metrics (dict):
                Computed metrics.
        """
        data = {}
        for k, v in self._data.items():
            data[k] = np.concatenate(v)

        results = concordance_index_censored(data["target_event"] == 1, data["target_time"], data["prediction"])

        result_data = {}
        for k, v in zip(ConcordanceIndex.METRIC_NAMES, results):
            result_data[f"concordance/{k}"] = v

        return result_data
