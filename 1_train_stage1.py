import warnings
warnings.filterwarnings('ignore')
import os
import numpy as np
import argparse
import importlib
import random
# from visdom import Visdom
import network.resnet38_cls


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision import transforms
from tool import pyutils, torchutils
from tool.GenDataset import Stage1_TrainDataset
from tool.infer_fun import infer
from tool.loss import KDLoss, PKT
cudnn.enabled = True

# seed = 666
# random.seed(seed)
# np.random.seed(seed)
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)


class HintLoss(nn.Module):
    """Fitnets: hints for thin deep nets, ICLR 2015"""
    def __init__(self):
        super(HintLoss, self).__init__()
        self.crit = nn.MSELoss()

    def forward(self, f_s, f_t):
        loss = self.crit(f_s, f_t)
        return loss



def compute_acc(pred_labels, gt_labels):
    pred_correct_count = 0
    for pred_label in pred_labels:
        if pred_label in gt_labels:
            pred_correct_count += 1
    union = len(gt_labels) + len(pred_labels) - pred_correct_count
    acc = round(pred_correct_count/union, 4)
    return acc

def adaptive_min_pooling_loss(x):
    # This loss does not affect the highest performance, but change the optimial background score (alpha)
    n,c,h,w = x.size()
    k = h*w//4
    x = torch.max(x, dim=1)[0]
    y = torch.topk(x.view(n,-1), k=k, dim=-1, largest=False)[0]
    y = F.relu(y, inplace=False)
    loss = torch.sum(y)/(k*n)
    return loss

def train_phase(args):
    # viz = Visdom(env=args.env_name)
    model = getattr(importlib.import_module(args.network), 'Net')(args.init_gama, n_class=args.n_class)
    print(vars(args))
    transform_train = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),
                                          transforms.GaussianBlur(3),
                                          transforms.RandomVerticalFlip(p=0.5),
                                          transforms.ColorJitter(brightness=0.5, contrast=0.5),
                                          transforms.ToTensor()])
    train_dataset = Stage1_TrainDataset(data_path=args.trainroot,transform=transform_train, dataset=args.dataset)
    train_data_loader = DataLoader(train_dataset,
                                    batch_size=args.batch_size,
                                    shuffle=True,
                                    num_workers=args.num_workers,
                                    pin_memory=False,
                                    drop_last=True)
    max_step = (len(train_dataset) // args.batch_size) * args.max_epoches
    param_groups = model.get_parameter_groups()
    optimizer = torchutils.PolyOptimizer([
        {'params': param_groups[0], 'lr': args.lr, 'weight_decay': args.wt_dec},
        {'params': param_groups[1], 'lr': 2*args.lr, 'weight_decay': 0},
        {'params': param_groups[2], 'lr': 10*args.lr, 'weight_decay': args.wt_dec},
        {'params': param_groups[3], 'lr': 20*args.lr, 'weight_decay': 0}
    ], lr=args.lr, weight_decay=args.wt_dec, max_step=max_step)
    if args.weights[-7:] == '.params':
        assert args.network == "network.resnet38_cls"
        import network.resnet38d
        weights_dict = network.resnet38d.convert_mxnet_to_torch(args.weights)
        model.load_state_dict(weights_dict, strict=False)
    elif args.weights[-4:] == '.pth':
        weights_dict = torch.load(args.weights)
        model.load_state_dict(weights_dict, strict=False)
    else:
        print('random init')
    model = model.cuda()
    avg_meter = pyutils.AverageMeter(
            'loss',
            'avg_ep_EM',
            'avg_ep_acc')
    timer = pyutils.Timer("Session started: ")
    # cp = network.resnet38_cls.Class_Predictor(4, 1).cuda()
    for ep in range(args.max_epoches):
        model.train()
        args.ep_index = ep
        ep_count = 0
        ep_EM = 0
        ep_acc = 0
        for iter, (filename, data, label) in enumerate(train_data_loader):
            img = data
            # img2 = F.interpolate(img, scale_factor=0.5, mode='bilinear', align_corners=False)
            # print(img, img2)
            label = label.cuda(non_blocking=True)
            # print(label, label2)
            if ep > 1:
                # enable_PDA = 1
                enable_AMM = 1
                # enable_NAEA = 1
                enable_MARS = 1
            else:
                # enable_PDA = 0
                enable_AMM = 0
                # enable_NAEA = 0
                enable_MARS = 0
            x, feature, y, cam1 = model(img.cuda(), enable_PDA=0, enable_AMM=0, enable_NAEA=0, enable_MARS=enable_MARS)
            # x, feature, y, cam1 = model(img.cuda(), enable_PDA=0, enable_AMM=0, enable_NAEA=0)
            # x_pre = x
            # if ep > 0:
            #     x = 0.9 * x + 0.1 * x_pre

            # x, feature, y = model(img.cuda(), enable_PDA=0, enable_AMM=1)
            # cam1 = model.forward_cam(img.cuda())
            # print(cam1)
            # print(feature.shape)
            # x2, feature2, y2, cam2 = model(img.cuda(), enable_PDA=0, enable_AMM=0, enable_NAEA=0, enable_MARS=0)
            # x_pre2 = x2
            # if ep > 0:
            #     x2 = 0.9 * x2 + 0.1 * x_pre2
            # cam2 = model.forward_cam(img.cuda())
            # print(cam2)
            prob = y.cpu().data.numpy()
            # prob2 = y2.cpu().data.numpy()
            # prob = (prob + prob2) / 2
            gt = label.cpu().data.numpy()
            for num, one in enumerate(prob):
                ep_count += 1
                pass_cls = np.where(one > 0.5)[0]
                true_cls = np.where(gt[num] == 1)[0]
                if np.array_equal(pass_cls, true_cls) == True:  # exact match
                    ep_EM += 1
                acc = compute_acc(pass_cls, true_cls)
                ep_acc += acc
            avg_ep_EM = round(ep_EM/ep_count, 4)
            avg_ep_acc = round(ep_acc/ep_count, 4)
            # print(label)
            # print(x)
            loss_cls = F.multilabel_soft_margin_loss(x, label, reduction='none')
            # loss_re, _ = cp(x, label)
            # print(loss_cls)
            # print(loss_cls.mean())
            # condition = loss_cls > (torch.max(loss_cls) - torch.min(loss_cls)) / 2
            # weight_mask = torch.where(condition.cuda(), loss_cls, torch.tensor(0.0).cuda())
            # weight_mask = weight_mask.float() / 255
            # # print(weight_mask)
            # uncertain_mask = weight_mask >= 0.005
            # weight_mask[uncertain_mask == 1] = 0.1
            # weight_mask[uncertain_mask == 0] = 1
            # loss_cls = loss_cls * weight_mask
            # print(loss_cls)
            # loss_cls2 = F.multilabel_soft_margin_loss(x2, label, reduction='none')
            # loss_kd = KDLoss(T=30)(cam1, cam2)
            # uncertain = torch.mean(torch.abs(cam1 - cam2))
            # print(cam1, cam2)
            # print(uncertain)
            # weight = 1 - uncertain
            # loss_ht = HintLoss()(loss_cls, loss_cls2)
            # loss_pkt = PKT()(loss_cls, loss_cls2)
            # loss_cps = torch.mean(torch.abs(cam1 - cam2))
            # print(loss_cls.mean(), loss_cls2.mean(), loss_cps.mean())
            # print(cams.shape)
            # loss = 0.5*loss_cls.mean() +0.5*loss_cls2.mean() + 0.01 * loss_cps
            loss = loss_cls.mean()
            # print(loss)
            avg_meter.add({'loss':loss.item(),
                            'avg_ep_EM':avg_ep_EM,
                            'avg_ep_acc':avg_ep_acc,
                           })
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()
            if (optimizer.global_step)%100 == 0 and (optimizer.global_step)!=0:
                timer.update_progress(optimizer.global_step / max_step)

                print('Epoch:%2d' % (ep),
                      'Iter:%5d/%5d' % (optimizer.global_step, max_step),
                      'Loss:%.4f' % (avg_meter.get('loss')),
                      'avg_ep_EM:%.4f' % (avg_meter.get('avg_ep_EM')),
                      'avg_ep_acc:%.4f' % (avg_meter.get('avg_ep_acc')),
                      'lr: %.4f' % (optimizer.param_groups[0]['lr']), 
                      'Fin:%s' % (timer.str_est_finish()),
                      flush=True)
                # viz.line([avg_meter.pop('loss')],[optimizer.global_step],win='loss',update='append',opts=dict(title='loss'))
                # viz.line([avg_meter.pop('avg_ep_EM')],[optimizer.global_step],win='Acc_exact',update='append',opts=dict(title='Acc_exact'))
                # viz.line([avg_meter.pop('avg_ep_acc')],[optimizer.global_step],win='Acc',update='append',opts=dict(title='Acc'))
        if model.gama > 0.65:
            model.gama = model.gama*0.98
        print('Gama of progressive dropout attention is: ',model.gama)
    torch.save(model.state_dict(), os.path.join(args.save_folder, 'stage1_checkpoint_trained_on_'+args.dataset+'_'+args.model_name+'.pth'))

def test_phase(args):
    model = getattr(importlib.import_module(args.network), 'Net_CAM')(n_class=args.n_class)
    model = model.cuda()
    args.weights = os.path.join(args.save_folder, 'stage1_checkpoint_trained_on_'+args.dataset+'_'+args.model_name+'.pth')
    # args.weights = os.path.join(args.save_folder, 'stage1_checkpoint_trained_on_'+args.dataset+'.pth')
    weights_dict = torch.load(args.weights)
    model.load_state_dict(weights_dict, strict=False)
    model.eval()
    score = infer(model, args.testroot, args.n_class)
    print(score)
    torch.save(model.state_dict(), os.path.join(args.save_folder, 'stage1_checkpoint_trained_on_'+args.dataset+'_'+args.model_name+'.pth'))
    # torch.save(model.state_dict(), os.path.join(args.save_folder, 'stage1_checkpoint_trained_on_'+args.dataset+'.pth'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=20, type=int)
    parser.add_argument("--max_epoches", default=20, type=int)
    parser.add_argument("--network", default="network.resnet38_cls", type=str)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--num_workers", default=1, type=int)
    parser.add_argument("--wt_dec", default=5e-4, type=float)
    parser.add_argument("--session_name", default="Stage 1", type=str)
    parser.add_argument("--env_name", default="PDA", type=str)
    parser.add_argument("--model_name", default='res38d_cca_ar', type=str)
    parser.add_argument("--n_class", default=4, type=int)
    parser.add_argument("--weights", default='/root/autodl-tmp/project/WSSS-Second/init_weights/ilsvrc-cls_rna-a1_cls1000_ep-0001.params', type=str)
    parser.add_argument("--trainroot", default='/root/autodl-tmp/project/WSSS-Second/datasets/BCSS-WSSS/train/', type=str)
    parser.add_argument("--testroot", default='/root/autodl-tmp/project/WSSS-Second/datasets/BCSS-WSSS/test/', type=str)
    parser.add_argument("--save_folder", default='/root/autodl-tmp/project/WSSS-Second/checkpoints/',  type=str)
    parser.add_argument("--init_gama", default=1, type=float)
    parser.add_argument("--dataset", default='bcss', type=str)
    args = parser.parse_args()
    # print(torch.cuda.memory_allocated())
    # print(torch.cuda.memory_reserved())
    torch.cuda.empty_cache()

    train_phase(args)
    test_phase(args)
