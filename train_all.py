from datetime import datetime
import pandas as pd
from pathlib import Path
import torch

from shape_continuum import cli
from shape_continuum.training.hooks import CheckpointSaver, TensorBoardLogger
from shape_continuum.training.optim import ClippedStepLR
from shape_continuum.training.train_and_eval import train_and_evaluate
from shape_continuum.testing.test_and_save import evaluate_model,save_csv
from shape_continuum.training.metrics import Accuracy,BalancedAccuracy,ConcordanceIndex



def main(args=None):
    parser = cli.create_parser()
    args = parser.parse_args(args=args)
    complete_dicts = []

    for shape in ["pointcloud","mesh","mask","vol_without_bg","vol_with_bg"]:
        if shape == "pointcloud":
            networks = ["pointnet","pointnet++"]
        elif shape == "mesh":
            networks = ["spiralnet"]
        else:
            networks = ["convnet","resnet"]
        for net in networks:
            args.shape = shape
            args.discriminator_net = net
            factory = cli.get_factory(args)
            experiment_dir, checkpoints_dir, tb_log_dir = factory.make_directories()

            factory.write_args(experiment_dir / "experiment_args.json")

            trainDataLoader, valDataLoader, testDataLoader = factory.get_data()
            discriminator = factory.get_and_init_model()
            optimizerD = factory.get_optimizer(discriminator.parameters())
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
                CheckpointSaver(discriminator, checkpoints_dir, save_every_n_epochs=3, max_keep=5,
                                metrics=eval_metrics_cp)
            )

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
                device=torch.device("cuda"),
            )
            if args.task == "clf":
                best_net_path = checkpoints_dir / "best_discriminator_balanced_accuracy.pth"
            else:
                best_net_path = checkpoints_dir / "best_discriminator_concordance_cindex.pth"
            best_discriminator = factory.get_and_init_model()
            best_discriminator.load_state_dict(torch.load(best_net_path))
            best_discriminator.cuda()
            param_dict = vars(args)
            param_dict["num_parameters"] = cli.get_number_of_parameters(best_discriminator)
            if args.task == "clf":
                testMetrics = [Accuracy("logits", "target"), BalancedAccuracy(args.num_classes, "logits", "target")]
            else:
                testMetrics = [ConcordanceIndex("logits", "target_event", "target_time")]
            metrics_dict, in_out_dict = evaluate_model(best_discriminator, testDataLoader, testMetrics,task=args.task)
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
