import torch

from shape_continuum import cli
from shape_continuum.training.hooks import CheckpointSaver, TensorBoardLogger
from shape_continuum.training.optim import ClippedStepLR
from shape_continuum.training.train_and_eval import train_and_evaluate


def main(args=None):
    parser = cli.create_parser()
    args = parser.parse_args(args=args)

    factory = cli.get_factory(args)
    experiment_dir, checkpoints_dir, tb_log_dir = factory.make_directories()

    factory.write_args(experiment_dir / "experiment_args.json")

    trainDataLoader, valDataLoader, testDataLoader = factory.get_data()
    discriminator = factory.get_and_init_model()
    optimizerD = factory.get_optimizer(discriminator.parameters())
    loss = factory.get_loss()
    train_metrics = factory.get_metrics()
    train_hooks = []  # [CheckpointSaver(discriminator, checkpoints_dir, save_every_n_epochs=3, max_keep=5)]
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
        CheckpointSaver(discriminator, checkpoints_dir, save_every_n_epochs=3, max_keep=5, metrics=eval_metrics_cp)
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


if __name__ == "__main__":
    main()
