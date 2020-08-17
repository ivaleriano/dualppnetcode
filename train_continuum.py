import argparse
import datetime
import json
import os

import torch
import torch.nn.parallel
import torch.utils.data
from tensorboardX import SummaryWriter
from tqdm import tqdm

from shape_continuum.data_utils.ADNIDataLoaders import ADNI_base_loader
from shape_continuum.data_utils.surv_data import make_loader
from shape_continuum.models.discriminative_models import DiscModel, SurvModel
from shape_continuum.networks.point_networks import PointNet, PointNet2ClsMsg
from shape_continuum.networks.vol_networks import ResNet, Vol_classifier
from shape_continuum.testing.continuum_tests import eval_clf, eval_surv


def parse_args():
    parser = argparse.ArgumentParser("PointNet")
    parser.add_argument("--batchsize", type=int, default=20, help="input batch size")
    parser.add_argument(
        "--workers", type=int, default=4, help="number of data loading workers"
    )
    parser.add_argument(
        "--epoch", type=int, default=200, help="number of epochs for training"
    )
    parser.add_argument(
        "--num_points", type=int, default=1500, help="number of epochs for training"
    )
    parser.add_argument(
        "--pretrain", type=str, default=None, help="whether use pretrain model"
    )
    parser.add_argument("--gpu", type=str, default="0", help="specify gpu device")
    parser.add_argument(
        "--learning_rate", type=float, default=0.001, help="learning rate for training"
    )
    parser.add_argument("--decay_rate", type=float, default=1e-4, help="weight decay")
    parser.add_argument(
        "--optimizer", type=str, default="Adam", help="type of optimizer"
    )
    parser.add_argument(
        "--task", type=str, default="clf", help="classification or survival analysis"
    )
    parser.add_argument(
        "--train_data",
        type=str,
        default="/home/ignacio/shapeAnalysis/data/ADNI_ALL/CN_AD_balanced_cls/pcs_mesh_mask_vols_train_set_1.csv",
        help="path to training dataset",
    )
    parser.add_argument(
        "--test_data",
        type=str,
        default="/home/ignacio/shapeAnalysis/data/ADNI_ALL/CN_AD_balanced_cls/pcs_mesh_mask_vols_test_set_1.csv",
        help="path to testing dataset",
    )
    parser.add_argument(
        "--discriminator_net",
        type=str,
        default="pointnet",
        help="which architecture to use for discriminator",
    )
    parser.add_argument(
        "--shape",
        type=str,
        default="pointcloud_fsl",
        help="which shape representation to use "
        "(pointcloud_free,pointcloud_fsl,mesh_fsl,vol_mask_free,vol_bb_nobg,vol_bb_wbg",
    )
    parser.add_argument("--num_classes", type=int, default=2, help="number of classes")
    parser.add_argument(
        "--out_csv",
        type=str,
        default="perf_metrics",
        help="output file to save performance metrics",
    )
    parser.add_argument(
        "--experiment_name",
        type=bool,
        default=False,
        help="True if input a particular name for the experiment (default False: current date and time)",
    )
    parser.add_argument(
        "--tb_comment",
        type=bool,
        default=False,
        help="any comment for storing on tensorboard",
    )
    parser.add_argument(
        "--tensorboard",
        type=bool,
        default=True,
        help="visualize training progress on tensorboard",
    )

    return parser.parse_args()


def train_one_epoch(model, trainDataLoader, epoch, writer=None):
    model.train()
    for batch_idx, data in tqdm(
        enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9
    ):

        if model.task == "surv":
            model.setShapes(data)
            model.train_epoch()
            if writer:
                writer.add_scalar(
                    "train/coxph_loss",
                    model.loss_D.cpu().data.numpy(),
                    epoch * len(trainDataLoader) + batch_idx,
                )
        else:
            model.setShapes(data)
            model.train_epoch()
            if writer:
                writer.add_scalar(
                    "train/nll_loss",
                    model.loss_D.cpu().data.numpy(),
                    epoch * len(trainDataLoader) + batch_idx,
                )

    print("training loss: %f" % model.loss_D.cpu().data.numpy())


def get_number_of_parameters(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def main(args):
    args = parse_args()

    """CREATE DIR"""

    experiment_dir = "experiments_%s" % args.task
    if args.experiment_name:
        experiment = input("input a name for your experiment")
    else:
        experiment = "shape_%s_network_%s" % (args.shape, args.discriminator_net)
    experiment_dir = os.path.join(experiment_dir, experiment)
    os.makedirs(experiment_dir, exist_ok=True)

    if args.batchsize == 1:
        args.batch_norm = False
    else:
        args.batch_norm = True

    checkpoints_dir = os.path.join(
        experiment_dir, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    )
    os.makedirs(checkpoints_dir, exist_ok=True)

    with open(os.path.join(checkpoints_dir, "experiment_args.txt"), "w") as f:
        json.dump(args.__dict__, f, indent=2)

    if args.tensorboard:
        if args.tb_comment:
            comment = input("comment to add to TB visualization: ")
        else:
            comment = ""
        writer = SummaryWriter(
            os.path.join(
                experiment_dir,
                "tb_log",
                comment
                + args.shape
                + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M"),
            )
        )
    else:
        writer = None

    if args.task == "surv":
        args.num_classes = 1

    trainDataset = ADNI_base_loader(args.train_data, shape=args.shape, task=args.task)
    valDataset = ADNI_base_loader(args.test_data, shape=args.shape, task=args.task)
    if args.task == "surv":
        trainDataLoader = make_loader(
            trainDataset, batch_size=args.batchsize, shuffle=True
        )
        valDataLoader = make_loader(
            valDataset, batch_size=args.batchsize, shuffle=False
        )
    else:
        trainDataLoader = torch.utils.data.DataLoader(
            trainDataset, batch_size=args.batchsize, shuffle=True, drop_last=True
        )
        valDataLoader = torch.utils.data.DataLoader(
            valDataset, batch_size=args.batchsize, shuffle=True, drop_last=True
        )

    if args.shape == "multi_vol":
        args.in_channels = 2
    else:
        args.in_channels = 1

    disc_ops = args
    if args.shape == "pointcloud_free" or args.shape == "pointcloud_fsl":
        args.discriminator = (
            PointNet(disc_ops)
            if args.discriminator_net == "pointnet"
            else PointNet2ClsMsg(disc_ops)
        )

    elif args.shape == "vol_mask_free":
        args.discriminator = (
            ResNet(disc_ops)
            if args.discriminator_net == "resnet"
            else Vol_classifier(disc_ops)
        )
    else:
        args.discriminator = Vol_classifier(disc_ops)  # TODO: change for mesh classfier

    print("number of parameters: %d" % get_number_of_parameters(args.discriminator))

    if args.pretrain is not None:
        args.discriminator.load_state_dict(torch.load(args.pretrain))
        print("load model %s" % args.pretrain)
    else:
        print("Training from scratch")
    pretrain = args.pretrain
    init_epoch = int(pretrain[-14:-11]) if args.pretrain is not None else 0  # TODO: CHANGE THIS

    if args.optimizer == "SGD":
        args.optimizerD = torch.optim.SGD(
            args.discriminator.parameters(), lr=0.01, momentum=0.9
        )
    elif args.optimizer == "Adam":
        args.optimizerD = torch.optim.Adam(
            args.discriminator.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate,
        )
    args.scheduler = torch.optim.lr_scheduler.StepLR(
        args.optimizerD, step_size=20, gamma=0.5
    )
    LEARNING_RATE_CLIP = 1e-5

    args.discriminator.cuda()
    if args.task == "surv":
        args.loss_method = "coxph"
        model = SurvModel(args)
    else:
        args.loss_method = "nll"
        model = DiscModel(args)
    i_test = 0
    for epoch in range(init_epoch, args.epoch):
        print("EPOCH : %d" % epoch)
        args.scheduler.step()
        lr = max(model.optimizerD.param_groups[0]["lr"], LEARNING_RATE_CLIP)
        print("Learning rate:%f" % lr)
        for param_group in model.optimizerD.param_groups:
            param_group["lr"] = lr
        train_one_epoch(model, trainDataLoader, epoch, writer)

        # TESTING
        if epoch % 3 == 0:
            if args.task == "surv":
                mets = eval_surv(model, valDataLoader, i_test, writer)
            else:
                mets = eval_clf(model, valDataLoader, i_test, writer)
                print("accuracy:" + str(mets["acc"]))
            i_test += 1
            torch.save(
                model.discriminator.state_dict(),
                "%s/discriminator_%.3d.pth" % (checkpoints_dir, epoch),
            )


if __name__ == "__main__":
    args = parse_args()
    main(args)
