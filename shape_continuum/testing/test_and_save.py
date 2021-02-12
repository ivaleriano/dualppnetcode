import pickle
from itertools import chain
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Sequence, Tuple

import pandas as pd
import torch
from torch import Tensor
from tqdm import tqdm

from ..cli import BaseModelFactory
from ..data_utils.adni_hdf import Task
from ..models.base import BaseModel, check_is_unique
from ..training.metrics import Metric
from ..training.train_and_eval import ModelRunner
from ..training.wrappers import DataLoaderWrapper


def concat_tensors_in_dict(data: Dict[Any, Sequence[Tensor]]) -> Dict[Any, Tensor]:
    return {k: torch.cat(v, dim=0) for k, v in data.items()}


class ModelTester(ModelRunner):
    """Execute a model on every batch of data in evaluation mode.

    Args:
      model (BaseModel):
        Instance of model to call.
      data (DataLoaderWrapper):
        Instance of DataLoader to obtain batches from.
        Keys of `data.output_names` must be contained in keys of `model.input_names`
        and `loss.input_names`.
      device (torch.device):
        Optional; Which device to run on.
      progressbar (bool):
        Optional; Whether to display a progess bar.
    """

    def __init__(
        self,
        model: BaseModel,
        data: DataLoaderWrapper,
        device: Optional[torch.device] = None,
        # hooks: Optional[Sequence[Hook]] = None,
        progressbar: bool = True,
    ) -> None:
        super().__init__(
            model=model, data=data, device=device, progressbar=progressbar,
        )
        all_names = list(chain(model.input_names, model.output_names))
        check_is_unique(all_names)

        model_data_intersect = set(model.input_names).intersection(set(data.output_names))
        if len(model_data_intersect) == 0:
            raise ValueError("model inputs and data loader outputs do not agree")

        self._collect_outputs = ("logits",)

    def _step_no_loss(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        outputs = super()._step(batch)

        batch.update(outputs)

        return outputs

    def _step(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        with torch.no_grad():
            return self._step_no_loss(batch)

    def _set_model_state(self):
        self.model = self.model.eval()

    def run(self):
        raise NotImplementedError("run is not implemented. You probably want to use predict or predict_iter.")

    def predict_iter(self) -> Iterator[Tuple[Dict[str, Tensor], Dict[str, Tensor]]]:
        """Execute model for a single batch.

        Yields:
            Two dicts. The first one contains the model's outputs,
            the second one the data loader's outputs that have not
            been consumed by the model, typically the true label.
        """
        self._set_model_state()

        extra_inputs = set(self.data.output_names).difference(set(self.model.input_names))

        pbar = tqdm(self.data, total=len(self.data), disable=not self.progressbar)
        for batch in pbar:
            batch = self._batch_to_device(batch)
            outputs = self._step(batch)
            predictions = {k: outputs[k].detach().cpu() for k in self._collect_outputs}

            input_data = {k: batch[k].detach().cpu() for k in extra_inputs}

            yield predictions, input_data

    def predict_all(self) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
        """Execute model for every batch and concatenate the outputs.

        Returns:
            Two dicts. The first one contains the model's outputs,
            the second one the data loader's outputs that have not
            been consumed by the model, typically the true label.
        """
        pred = {k: [] for k in self._collect_outputs}
        unconsumed_inputs = {}
        for p, ed in self.predict_iter():
            for k, v in p.items():
                pred[k].append(v)
            for k, v in ed.items():
                unconsumed_inputs.setdefault(k, []).append(v)

        pred = concat_tensors_in_dict(pred)
        unconsumed_inputs = concat_tensors_in_dict(unconsumed_inputs)

        return pred, unconsumed_inputs


def evaluate_model(*, metrics: Sequence[Metric], **kwargs) -> Tuple[Dict[str, float], Dict[str, Tensor]]:
    """Obtain predictions from model and evaluate its performance.

    Args:
      metrics (list):
        List of metrics to compute on test data
      model (BaseModel):
        Instance of model to call.
      data (DataLoaderWrapper):
        Instance of DataLoader to obtain batches from.
        Keys of `data.output_names` must be contained in keys of `model.input_names`
        and `loss.input_names`.
      device (torch.device):
        Optional; Which device to run on.
      progressbar (bool):
        Optional; Whether to display a progess bar.

    Returns:
        Two dicts. The first dict contains the computed metrics
        on the entire data. The second dict contains the model's raw
        output for each data point.
    """
    tester = ModelTester(**kwargs)
    predictions, unconsumed_inputs = tester.predict_all()

    metrics_dict = {}
    for m in metrics:
        m.reset()
        m.update(inputs=unconsumed_inputs, outputs=predictions)
        metrics_dict.update(m.values())

    predictions.update(unconsumed_inputs)
    return metrics_dict, predictions


def load_best_model(
    factory: BaseModelFactory, checkpoints_dir: Path, device: Optional[torch.device] = None
) -> torch.nn.Module:
    """Load model from `best' checkpoint in the given dir.

    The checkpoint to restore depends on the value of ``factory.task``.

    Args:
      factory (BaseModelFactory):
        Factory used to create model instance.
      checkpoints_dir (Path):
        Directory to search for checkpoints.
      device (torch.device):
        Optional; The device the model should be associated with.

    Returns:
        torch.nn.Module: The model with weights restored from the `best' checkpoint.
    """
    if factory.task in {Task.BINARY_CLASSIFICATION, Task.MULTI_CLASSIFICATION}:
        metric = "balanced_accuracy"
    elif factory.task == Task.SURVIVAL_ANALYSIS:
        metric = "concordance_cindex"
    else:
        raise ValueError("task={!r} is not supported".format(factory.task))

    best_net_path = checkpoints_dir / f"best_discriminator_{metric}.pth"
    best_discriminator = factory.get_and_init_model()
    best_discriminator.load_state_dict(torch.load(best_net_path, map_location=device))
    best_discriminator = best_discriminator.to(device)

    return best_discriminator


def save_csv(
    csv_dir: Path, params: Dict[str, Any], out_metrics: Dict[str, float], tensors: Dict[str, Tensor]
) -> Dict[str, Any]:
    """Write parameters, metrics and tensors to disk.

    Args:
      csv_dir (Path):
        Path to directory where outputs will be written to. Will be created
        if it does not exist.
      params (dict):
        Meta-information, such as hyper-parameters, that will be saved
        together with metrics in a ``metrics.csv`` file.
      out_metrics (dict):
        Name and value of metrics to save.
      tensors (dict):
        Tensors to save in ``logits.pkl``. Torch tensors will be concerted
        to numpy arrays before pickling.

    Returns:
      dict: Values written to ``metrics.csv`` file.
    """
    csv_dir.mkdir(parents=True, exist_ok=True)
    logits_path = csv_dir / "logits.pkl"
    metrics_path = csv_dir / "metrics.csv"

    arrays = {k: v.numpy() for k, v in tensors.items()}
    with open(logits_path, "wb") as fout:
        pickle.dump(arrays, fout, protocol=pickle.HIGHEST_PROTOCOL)

    assert len(set(params.keys()).intersection(set(out_metrics.keys()))) == 0
    saving_dict = {"logits_dir": str(logits_path.resolve())}
    saving_dict.update(params)
    saving_dict.update(out_metrics)
    saving_dict["Name"] = "{}-{}".format(saving_dict["discriminator_net"], saving_dict["shape"])

    df = pd.DataFrame.from_dict({k: [v] for k, v in saving_dict.items()}).sort_index(axis=1)
    df.to_csv(metrics_path, index=False)
    return saving_dict
