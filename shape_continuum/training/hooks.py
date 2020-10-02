import numbers
from pathlib import Path
from typing import Dict, Optional, Sequence

import torch
from torch import Tensor
from torch.nn import Module
from torch.utils.tensorboard.writer import SummaryWriter

from .metrics import Metric


class Hook:
    """Base class for hooks called by :class:`shape_continuum.training.train_and_eval.ModelRunner`"""

    def on_begin_epoch(self) -> None:
        """Called before each iteration over the data."""

    def on_end_epoch(self) -> None:
        """Called after the data has been fully consumed."""

    def before_step(self, inputs: Dict[str, Tensor]) -> None:
        """Called before the model is evaluated on a batch.

        Args:
          inputs:
            A Dict with the batch's data obtained from the DataLoader and passed to the model.
        """

    def after_step(self, outputs: Dict[str, Tensor]) -> None:
        """Called fater the model is evaluated on a batch.

        Args:
          outputs:
            A Dict with the outputs returned by the model for the current batch.
        """


class CheckpointSaver(Hook):
    """Saves checkpoints every N epochs.

    Args:
      model (Module):
        The model which state should be saved.
      checkpoint_dir (str):
        Base directory for the checkpoint files.
      metrics (list of Metric):
        Instances of metrics to compute. Used for keeping track of best performing model
      save_best(boolean):
        True if you want to save the best performing model for each metric
      save_every_n_epochs (int):
        Optional; Save every N steps.
      max_keep (int):
        Optional; Keep the latest N checkpoints, or all, if None.
    """

    def __init__(
        self,
        model: Module,
        checkpoint_dir: str,
        save_every_n_epochs: int = 1,
        max_keep: Optional[int] = None,
        metrics: Sequence[Metric] = None,
        save_best: bool = False,
    ) -> None:
        self._model = model
        self._checkpoint_dir = Path(checkpoint_dir)
        self._save_every_n_epochs = save_every_n_epochs
        self._max_keep = max_keep
        self._epoch = 0
        self._ckpkt_remove = []
        self._metrics = metrics
        self._save_best = save_best

    def _forward(self, fn_name, *args):
        for m in self._metrics:
            fn = getattr(m, fn_name)
            fn(*args)

    def before_step(self, inputs: Dict[str, Tensor]) -> None:
        self._inputs = inputs

    def after_step(self, outputs: Dict[str, Tensor]) -> None:
        if self._save_best and self._checkpoint_dir:
            self._forward("update", self._inputs, outputs)

    def on_begin_epoch(self) -> None:
        if self._save_best and self._checkpoint_dir:
            self._forward("reset")

    def on_end_epoch(self) -> None:
        self._epoch += 1
        if self._epoch % self._save_every_n_epochs == 0:
            ckpt_path = self._save()
            if self._max_keep is not None:
                self._remove()
                self._ckpkt_remove.append(ckpt_path)
        if self._save_best and self._checkpoint_dir:
            self._save_best_models()

    def _save(self):
        path = self._checkpoint_dir / "discriminator_{:04d}.pth".format(self._epoch)
        torch.save(
            self._model.state_dict(), path,
        )

        return path

    def _save_best_models(self):
        for m in self._metrics:
            if m.is_best():
                for name in m.values().keys():
                    path = self._checkpoint_dir / "best_discriminator_{:}.pth".format(name)
                    torch.save(
                        self._model.state_dict(), path,
                    )

    def _remove(self):
        if len(self._ckpkt_remove) == self._max_keep:
            self._ckpkt_remove[0].unlink()
            self._ckpkt_remove = self._ckpkt_remove[1:]


class TensorBoardLogger(Hook):
    """Logs metrics after every epoch for visualization in TensorBoard.

    Args:
      log_dir (str):
        The path of the directory where to save the log files to be parsed by TensorBoard.
      metrics (list of Metric):
        Instances of metrics to compute and log.
    """

    def __init__(self, log_dir: str, metrics: Sequence[Metric]) -> None:
        self._writer = SummaryWriter(log_dir)
        self._metrics = metrics
        self._epoch = 0

    def _forward(self, fn_name, *args):
        for m in self._metrics:
            fn = getattr(m, fn_name)
            fn(*args)

    def on_begin_epoch(self) -> None:
        self._forward("reset")

    def on_end_epoch(self) -> None:
        self._epoch += 1
        self._write_all()

    def before_step(self, inputs: Dict[str, Tensor]) -> None:
        self._inputs = inputs

    def after_step(self, outputs: Dict[str, Tensor]) -> None:
        self._forward("update", self._inputs, outputs)

    def _write_all(self):
        for m in self._metrics:
            for name, value in m.values().items():
                self._write(name, value)

    def _write(self, name: str, value):
        if isinstance(value, numbers.Number):
            self._writer.add_scalar(name, value, global_step=self._epoch)
        else:
            self._writer.add_histogram(name, value, global_step=self._epoch)
