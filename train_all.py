from datetime import datetime
from pathlib import Path

import pandas as pd
import torch

from shape_continuum import cli
from shape_continuum.testing.test_and_save import evaluate_model, load_best_model, save_csv
from shape_continuum.training.hooks import CheckpointSaver, TensorBoardLogger
from shape_continuum.training.optim import ClippedStepLR
from shape_continuum.training.train_and_eval import train_and_evaluate


def main(args=None):
    parser = cli.create_parser()
    args = parser.parse_args(args=args)
    complete_dicts = []

    for shape in ["pointcloud", "mesh", "mask", "vol_without_bg", "vol_with_bg"]:
        if shape == "pointcloud":
            networks = ["pointnet", "pointnet++"]
        elif shape == "mesh":
            networks = ["spiralnet"]
        else:
            networks = ["convnet", "resnet"]

        for net in networks:
            args.shape = shape
            args.discriminator_net = net
            factory = cli.get_factory(args)
            experiment_dir, checkpoints_dir, tb_log_dir = factory.make_directories()

            factory.write_args(experiment_dir / "experiment_args.json")

            trainDataLoader, valDataLoader, testDataLoader = factory.get_data()
            discriminator = factory.get_and_init_model()
            optimizerD = factory.get_optimizer(filter(lambda p: p.requires_grad, discriminator.parameters()))
            loss = factory.get_loss()
            train_metrics = factory.get_metrics()
            train_hooks = []
            eval_hooks = []

            if args.tensorboard:
                if args.tb_comment:
                    comment = input("comment to add to TB visualization: ")
                    tb_log_dir /= comment
                train_hooks.append(TensorBoardLogger(str(tb_log_dir / "train"), train_metrics))
                eval_metrics_tb = factory.get_metrics()
                eval_hooks = [TensorBoardLogger(str(tb_log_dir / "eval"), eval_metrics_tb)]
            eval_metrics_cp = factory.get_metrics()
            eval_hooks.append(
                CheckpointSaver(
                    discriminator, checkpoints_dir, save_every_n_epochs=3, max_keep=5, metrics=eval_metrics_cp
                )
            )

            device = torch.device("cuda")
            train_and_evaluate(
                model=discriminator,
                loss=loss,
                train_data=trainDataLoader,
                optimizer=optimizerD,
                scheduler=ClippedStepLR(optimizerD, step_size=30, gamma=0.5),
                num_epochs=args.epoch,
                eval_data=valDataLoader,
                train_hooks=train_hooks,
                eval_hooks=eval_hooks,
                device=device,
            )

            best_discriminator = load_best_model(factory, checkpoints_dir, device)

            param_dict = factory.get_args()
            param_dict["num_parameters"] = cli.get_number_of_parameters(best_discriminator)

            testMetrics = factory.get_test_metrics()
            metrics_dict, in_out_dict = evaluate_model(
                model=best_discriminator, data=testDataLoader, metrics=testMetrics, device=device
            )
            saving_dict = save_csv(experiment_dir / "csv", param_dict, metrics_dict, in_out_dict)
            complete_dicts.append(saving_dict)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    base_dir = Path(f"experiments_{args.task}_all")
    base_dir.mkdir(exist_ok=True)
    complete_experiment_dir = base_dir / timestamp
    complete_experiment_dir.mkdir(exist_ok=True)
    complete_experiment_csv = complete_experiment_dir / "metrics_and_params.csv"
    df = pd.DataFrame(complete_dicts)
    df.to_csv(complete_experiment_csv)


if __name__ == "__main__":
    main()
