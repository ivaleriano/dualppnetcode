import argparse
import torch
from torch import Tensor
from torch.nn import Module
from typing import Dict, Optional, Sequence
from pathlib import Path
from ..training.wrappers import DataLoaderWrapper
from ..training.metrics import Metric
from ..training.train_and_eval import ModelRunner
from ..models.base import BaseModel,check_is_unique
from itertools import chain
import tqdm
import os
import pandas as pd







"""
TO DO:
- (done) ModelEvaluator takes loader and model and concatenates all logits
- (done) Save logits in a separate file and keep path of where it is it saved
- (done) Use Metrics methods to compute the metrics
- (done) Output all metrics in a dictionary
- (done) Save csv with:
    - argparser parameters
    - metrics
    - location of logits

- Comment the code

"""


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
      hooks (list of Hook):
        Optional; List of hooks to call during execution.
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
        task: str = "clf",
    ) -> None:
        super().__init__(
            model=model, data=data, device=device, progressbar=progressbar,
        )
        all_names = list(chain(model.input_names, model.output_names))
        check_is_unique(all_names)
        self.task = task

        model_data_intersect = set(model.input_names).intersection(set(data.output_names))
        if len(model_data_intersect) == 0:
            raise ValueError("model inputs and data loader outputs do not agree")



    def _step_no_loss(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        outputs = super()._step(batch)

        batch.update(outputs)

        return outputs

    def _step(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        with torch.no_grad():
            return self._step_no_loss(batch)

    def _run(self) -> Dict[str,Tensor]:
        """Execute model for every batch."""
        self._set_model_state()

        logits = torch.Tensor().cuda()
        targets = torch.LongTensor().cuda()
        target_event = torch.ByteTensor().cuda()
        target_time = torch.DoubleTensor().cuda()
        #pbar = tqdm(self.data, total=len(self.data), disable=not self.progressbar)
        for batch in self.data:
            batch = self._batch_to_device(batch)
            outputs = self._step(batch)
            logits = torch.cat([logits,outputs["logits"]],dim=0)
            if self.task == "clf":
                targets = torch.cat([targets,batch["target"]])
            else:
                target_event = torch.cat([target_event, batch["event"]])
                target_time = torch.cat([target_time, batch["time"]])

            #     target_time = torch.cat([target_time,batch["target_time"]])
            #     target_time = torch.cat([target_time, batch["target_event"]])
        return {"logits":logits,"target":targets,"target_event":target_event,"target_time":target_time}



def evaluate_model(model:Module,DataLoader:DataLoaderWrapper,metrics:Sequence[Metric],task:str="clf") -> Sequence[Dict[str,Tensor]]:
    tester = ModelTester(model=model,data=DataLoader,device=torch.device("cuda"),progressbar=True,task=task)
    in_out_dict = tester._run()
    metrics_dict = {}
    for m in metrics:
        m.reset()
        m.update(inputs=in_out_dict,outputs=in_out_dict)
        for key,value in m.values().items():
            metrics_dict[key] = value
    return metrics_dict,in_out_dict

def save_csv(csv_dir:Path,params:Dict,out_metrics:Dict[str,Tensor],log_targ_metrics:Dict[str,Tensor]):
    os.makedirs(csv_dir,exist_ok=True)
    logits_path = csv_dir / "logits.csv"
    metrics_path = csv_dir / "metrics.csv"

    df = pd.DataFrame([log_targ_metrics])
    df.to_csv(logits_path)
    saving_dict = {**params,**out_metrics}
    saving_dict["logits_dir"] = logits_path
    saving_dict["Name"] = saving_dict["discriminator_net"] + "-" + saving_dict["shape"]
    df = pd.DataFrame([saving_dict])
    df.to_csv(metrics_path,index=False)
    return saving_dict











def create_parser():
    parser = argparse.ArgumentParser("PointNet")
    parser.add_argument("--batchsize", type=int, default=20, help="input batch size")
    parser.add_argument("--workers", type=int, default=4, help="number of data loading workers")
    parser.add_argument("--num_points", type=int, default=1500, help="number of epochs for training")
    parser.add_argument("--task", choices=["clf", "surv"], default="clf", help="classification or survival analysis")
    parser.add_argument("--test_data", type=Path, required=True, help="path to testing dataset")
    parser.add_argument(
        "--discriminator_net", default="pointnet", help="which architecture to use for discriminator",
    )
    parser.add_argument(
        "--shape",
        default="pointcloud",
        help="which shape representation to use "
        "(pointcloud,mesh,mask,vol_with_bg,vol_without_bg",
    )
    parser.add_argument("--num_classes", type=int, default=3, help="number of classes")


    return parser
























