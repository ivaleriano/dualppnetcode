import argparse
import torch
import numpy as np
import torch.nn.parallel
import torch.utils.data
from tqdm import tqdm

from sksurv.metrics import concordance_index_censored
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, average_precision_score


def parse_args():
    parser = argparse.ArgumentParser('PointNet')
    parser.add_argument('--batchsize', type=int, default=40, help='input batch size')
    parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')
    parser.add_argument('--lat_features', type=int, default=1024, help='number of latent features')
    parser.add_argument('--num_points', type=int, default=1500, help='number of epochs for training')
    parser.add_argument('--gpu', type=str, default='1', help='specify gpu device')
    parser.add_argument('--task', type=str, default='surv', help='classification or survival analysis')
    parser.add_argument('--dataset', type=str, default='adni', help='dataset to train on')
    parser.add_argument('--data_path_test', type=str, default='../data/adni_surv_mci2ad_bl_balanced_sets/sets/0/test/', help='path to testing dataset')
    parser.add_argument('--discriminator_net', type=str, default='hrpnet',
                        help='which architecture to use for discriminator')
    parser.add_argument('--num_classes', type=int, default=1, help='number of classes')
    parser.add_argument('--radius0', type=float, default=0.1, help='radius of first query ball (h=0) ')
    parser.add_argument('--nsamples0', type=int, default=32, help='number of samples per neighborhood in h=0')
    parser.add_argument('--block_list', default=['crb','crb'], nargs='*',help='list specifying which recalibration block to use in each level (options:none,crb,srb,scrb)')
    parser.add_argument('--out_csv',type=str,default='perf_metrics',help='output file to save performance metrics')

    return parser.parse_args()



def eval_clf(model,testDataLoader,i_test=0,writer=None):

    mean_correct = []
    mean_precision = []
    mean_recall = []
    mean_f1 = []
    mean_auc = []
    mean_a_prec = []

    with torch.no_grad():
        model.eval()
        for j, data in enumerate(testDataLoader, 0):
            shapes, target = data
            target = target[:, 0]
            shapes, target = shapes.cuda(), target.cuda()
            model.setShapes(shapes,target)
            model.forward()
            pred = torch.squeeze(model.pred)
            pred_choice = pred.data.max(1)[1]
            prob = np.exp(pred[:, 1].cpu().data.numpy())
            correct = pred_choice.eq(target.long().data).cpu().sum()
            mean_correct.append(correct.item() / float(shapes.size()[0]))
            pred_choice = pred_choice.cpu().data.numpy()
            target = target.long().cpu().data.numpy()
            prec, rec, f1, _ = precision_recall_fscore_support(target, pred_choice, average='binary')

            auc = roc_auc_score(target,prob)
            avg_prec = average_precision_score(target,prob)
            mean_precision.append(prec)
            mean_recall.append(rec)
            mean_f1.append(f1)
            mean_auc.append(auc)
            mean_a_prec.append(avg_prec)

    metrics = {'acc': np.mean(mean_correct),
                'precision': np.mean(mean_precision),
                'recall': np.mean(mean_recall),
                'f1': np.mean(mean_f1),
               'auc': np.mean(mean_auc),
               'avg_prec': np.mean(mean_a_prec)
               }
    if writer:
        for metric in metrics:
            writer.add_scalar('test/%s' % metric, metrics[metric], i_test)

    return metrics




def eval_surv(model,testDataLoader,i_test=0,writer=None):
    with torch.no_grad():
        model.eval()
        for batch_idx, (x, y_event, y_time, y_riskset) in tqdm(enumerate(testDataLoader, 0),
                                                           total=len(testDataLoader),
                                                           smoothing=0.9):
            shapes, y_event, y_riskset = \
                x.cuda(), y_event.cuda(), y_riskset.cuda()
            model.setShapes(shapes, y_riskset, y_event)
            model.forward()
            pred = torch.squeeze(model.pred).cpu().detach().numpy()
            event = torch.squeeze(y_event).cpu().detach().numpy()
            time = torch.squeeze(y_time).cpu().detach().numpy()
            if batch_idx == 0:
                pred_all = pred
                time_all = time
                event_all = event
            else:
                pred_all = np.concatenate([pred_all, pred], axis=0)
                time_all = np.concatenate([time_all, time], axis=0)
                event_all = np.concatenate([event_all, event], axis=0)
        ret = concordance_index_censored(event_all == 1, time_all, pred_all)
        if writer:
            writer.add_scalar('test/concordance_index', ret[0], i_test)
    return {'cindex': ret[0]}