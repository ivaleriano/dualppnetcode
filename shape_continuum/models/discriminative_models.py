import torch.nn as nn
import torch.nn.functional as F

from .losses import coxph_loss


class DiscModel(nn.Module):
    def __init__(self, opt, train=True):
        super(DiscModel, self).__init__()
        self.discriminator = opt.discriminator
        self.opt = opt
        self.task = opt.task
        self.shape = opt.shape
        if train:
            self.loss_method = opt.loss_method
            self.optimizerD = opt.optimizerD

    def setShapes(self, data):
        shapes, target = data
        self.shapes, self.target = shapes.cuda(), target[:, 0].cuda()
        if self.shape == "pointcloud_free" or self.shape == "pointcloud_fsl":
            self.bs, C, N = shapes.size()
            if C > N:
                self.shapes = self.shapes.transpose(2, 1)

    def forward(self):
        # self.generator.zero_grad()
        self.outputs = self.discriminator(self.shapes)
        self.pred = self.outputs["pred"]

    def get_disc_loss(self, pred, target):
        if self.loss_method == "nll":
            loss = F.nll_loss(pred, target.long())
        elif self.loss_method == "bce":
            loss = F.binary_cross_entropy(pred, target.long())
        else:
            print("classification loss not defined. Using nll")
            loss = F.nll_loss(pred, target.long())
        return loss

    def get_accuracy(self):
        pred_choice = self.pred.data.max(1)[1]
        correct = pred_choice.eq(self.target.long().data).cpu().sum()
        acc = correct.item() / float(self.pred.size()[0])
        return acc

    def backward_D(self):
        self.loss_D = self.get_disc_loss(self.pred, self.target)
        self.loss_D.backward()

    def train_epoch(self):
        self.forward()
        self.optimizerD.zero_grad()  # set G's gradients to zero
        self.backward_D()  # calculate graidents for G
        self.optimizerD.step()


class SurvModel(DiscModel):
    def setShapes(self, data):
        shapes, event, y_time, riskset = data
        self.shapes = shapes.cuda()
        self.riskset = riskset.cuda()
        self.event = event.cuda()
        if self.shape == "pointcloud_free" or self.shape == "pointcloud_fsl":
            self.bs, C, N = shapes.size()
            if C > N:
                self.shapes = self.shapes.transpose(2, 1)

    def get_disc_loss(self, pred, target_riskset, target_event):
        return coxph_loss(target_event, target_riskset, pred)

    def backward_D(self):
        self.loss_D = self.get_disc_loss(self.pred, self.riskset, self.event)
        self.loss_D.backward()
