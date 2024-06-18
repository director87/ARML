import argparse
import os
import numpy as np
import math
import random
import cv2
from tqdm import tqdm
from PIL import Image
import torch
# torch.autograd.set_detect_anomaly(True)
from tool.GenDataset import make_data_loader
from network.sync_batchnorm.replicate import patch_replication_callback
from network.deeplab import *
from network.pspnet import PSPNet
from tool.loss import SegmentationLosses, STLoss
from tool.lr_scheduler import LR_Scheduler
from tool.saver import Saver
from tool.summaries import TensorboardSummary
from tool.metrics import Evaluator
import ml_collections
import segmentation_models_pytorch as smp
from collections import defaultdict
from scipy import stats
from torchvision import transforms
import timm

# torch.backends.cudnn.benchmark = False

def SWV(outputs_main, outputs_aux1, outputs_aux2, mask):
    n = outputs_main.shape[0]
    loss_main = F.cross_entropy(
        outputs_main, mask.long(), reduction='none').view(n, -1)
    hard_aux1 = torch.argmax(outputs_aux1, dim=1).view(n, -1)
    hard_aux2 = torch.argmax(outputs_aux2, dim=1).view(n, -1)
    loss_select = 0
    for i in range(n):
        aux1_sample = hard_aux1[i]
        aux2_sample = hard_aux2[i]
        loss_sample = loss_main[i]
        agree_aux = (aux1_sample == aux2_sample)
        disagree_aux = (aux1_sample != aux2_sample)
        loss_select += 2*torch.sum(loss_sample[agree_aux]) + (1/2)*torch.sum(loss_sample[disagree_aux])

    return loss_select / (n*loss_main.shape[1])

class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.gama = 1.0
        # Define
        self.saver = Saver(args)
        self.summary = TensorboardSummary('logs')
        self.writer = self.summary.create_summary()
        kwargs = {'num_workers': args.workers, 'pin_memory': False}
        self.train_loader, self.val_loader, self.test_loader = make_data_loader(args, **kwargs)
        self.nclass = args.n_class
        # model = DeepLab(num_classes=self.nclass,
        #                 backbone=args.backbone,
        #                 output_stride=args.out_stride,
        #                 sync_bn=args.sync_bn,
        #                 freeze_bn=args.freeze_bn)
        model = smp.PSPNet(encoder_name='timm-resnest101e', encoder_weights='imagenet', in_channels=3, classes=self.nclass)
        # train_params = [{'params': model.get_1x_lr_params(), 'lr': args.lr},
        #                 {'params': model.get_10x_lr_params(), 'lr': args.lr * 10}]
        lr = args.lr
        optimizer = torch.optim.SGD(params=model.parameters(), lr=lr, momentum=args.momentum,
                                    weight_decay=args.weight_decay)
        self.criterion = SegmentationLosses(weight=None, cuda=args.cuda).build_loss(mode=args.loss_type)
        self.model, self.optimizer = model, optimizer
        self.evaluator = Evaluator(self.nclass)
        self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr,
                                            args.epochs, len(self.train_loader))

        # Using cuda
        if args.cuda:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids)
            patch_replication_callback(self.model)
            self.model = self.model.cuda()
        # Resuming checkpoint
        self.best_pred = 0.0
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'" .format(args.resume))
            checkpoint = torch.load(args.resume)
            # if args.cuda:
            #     W = checkpoint['state_dict']
            #     if not args.ft:
            #         del W['decoder.last_conv.8.weight']
            #         del W['decoder.last_conv.8.bias']
            #     self.model.module.load_state_dict(W, strict=False)
            # else:
            #     self.model.load_state_dict(checkpoint['state_dict'])
            self.model.module.load_state_dict(checkpoint['state_dict'])
            # if args.ft:
            #     self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' ".format(args.resume))

    def training(self, epoch):
        train_loss = 0.0
        self.model.train()
        tbar = tqdm(self.train_loader)
        num_img_tr = len(self.train_loader)
        for i, sample in enumerate(tbar):
            image, target, target_a, target_b = sample['image'], sample['label'], sample['label_a'], sample['label_b']
            if self.args.cuda:
                image, target, target_a, target_b = image.cuda(), target.cuda(), target_a.cuda(), target_b.cuda()
            self.scheduler(self.optimizer, i, epoch, self.best_pred)
            self.optimizer.zero_grad()
            output = self.model(image)
            output2 = self.model(image)
            output3 = self.model(image)
            one = torch.ones((output.shape[0],1,224,224)).cuda()
            one2 = torch.ones((output2.shape[0],1,224,224)).cuda()
            one3 = torch.ones((output3.shape[0],1,224,224)).cuda()
            output = torch.cat([output,(100 * one * (target==4).unsqueeze(dim = 1))],dim = 1)
            output2 = torch.cat([output2,(100 * one2 * (target==4).unsqueeze(dim = 1))],dim = 1)
            output3 = torch.cat([output3,(100 * one3 * (target==4).unsqueeze(dim = 1))],dim = 1)
            loss_o = self.criterion(output, target, self.gama)
            # loss_a = self.criterion(output, target_a, self.gama)
            # loss_b = self.criterion(output, target_b, self.gama)
            loss_v1 = SWV(output, output2, output3, target)
            loss_st1 = STLoss()(output, output2)
            loss_st2 = STLoss()(output, output3)
            loss_st = (loss_st1 + loss_st2) / 2
            loss = 0.8*loss_v1 + 0.2*loss_st
            # loss = 0.6 * loss_o + 0.2 * loss_a + 0.2 * loss_b
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))
            self.writer.add_scalar('train/total_loss_iter', loss.item(), i + num_img_tr * epoch)

        self.writer.add_scalar('train/total_loss_epoch', train_loss, epoch)
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        print('Loss: %.3f' % train_loss)
        # if (epoch + 1) % 3 == 0:
        self.gama = self.gama * 0.98

    def validation(self, epoch):
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.val_loader, desc='\r')
        test_loss = 0.0
        for i, sample in enumerate(tbar):
            image, target = sample[0]['image'], sample[0]['label']
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                output = self.model(image)
            pred = output.data.cpu().numpy()
            target = target.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            ## cls 4 is exclude
            pred[target==4]=4
            self.evaluator.add_batch(target, pred)

        # Fast test during the training
        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        ious = self.evaluator.Intersection_over_Union()
        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
        DSC = self.evaluator.Dice_Similarity_Coefficient()
        self.writer.add_scalar('val/total_loss_epoch', test_loss, epoch)
        self.writer.add_scalar('val/mIoU', mIoU, epoch)
        self.writer.add_scalar('val/Acc', Acc, epoch)
        self.writer.add_scalar('val/Acc_class', Acc_class, epoch)
        self.writer.add_scalar('val/fwIoU', FWIoU, epoch)
        self.writer.add_scalar('val/DSC', DSC, epoch)
        print('Validation:')
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}, DSC: {}".format(Acc, Acc_class, mIoU, FWIoU, DSC))
        print('Loss: %.3f' % test_loss)
        print('IoUs: ', ious)

        if mIoU > self.best_pred:
            self.best_pred = mIoU
            self.saver.save_checkpoint({
                'state_dict': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict()
            }, 'stage2_checkpoint_trained_on_'+self.args.dataset+self.args.backbone+self.args.loss_type+'.pth')
    def load_the_best_checkpoint(self):
        checkpoint = torch.load('checkpoints/stage2_checkpoint_trained_on_'+self.args.dataset+self.args.backbone+self.args.loss_type+'.pth')
        self.model.module.load_state_dict(checkpoint['state_dict'], strict=False)
    def test(self, epoch, Is_GM):
        self.load_the_best_checkpoint()
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.test_loader, desc='\r')
        test_loss = 0.0
        for i, sample in enumerate(tbar):
            image, target = sample[0]['image'], sample[0]['label']
            image_name = sample[-1][0].split('/')[-1].replace('.png', '')
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                output = self.model(image)
                if Is_GM:
                    output = self.model(image)
                    # print(output.shape)
                    _,y_cls = self.model_stage1.forward_cam(image)
                    y_cls = y_cls.cpu().data
                    # y_cls = y_cls.cpu().data
                    # print(y_cls)
                    pred_cls = (y_cls > 0.5)
            pred = output.data.cpu().numpy()
            if Is_GM:
                pred = pred*(pred_cls.unsqueeze(dim=2).unsqueeze(dim=3).numpy())
            target = target.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            ## cls 4 is exclude
            pred[target==4]=4
            self.evaluator.add_batch(target, pred)

        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        ious = self.evaluator.Intersection_over_Union()
        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
        DSC = self.evaluator.Dice_Similarity_Coefficient()
        self.writer.add_scalar('val/total_loss_epoch', test_loss, epoch)
        self.writer.add_scalar('val/mIoU', mIoU, epoch)
        self.writer.add_scalar('val/Acc', Acc, epoch)
        self.writer.add_scalar('val/Acc_class', Acc_class, epoch)
        self.writer.add_scalar('val/fwIoU', FWIoU, epoch)
        self.writer.add_scalar('val/DSC', DSC, epoch)
        print('Test:')
        print('[numImages: %5d]' % (i * self.args.batch_size + image.data.shape[0]))
        print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}, DSC: {}".format(Acc, Acc_class, mIoU, FWIoU, DSC))
        print('Loss: %.3f' % test_loss)
        print('IoUs: ', ious)

def main():
    parser = argparse.ArgumentParser(description="WSSS Stage2")
    parser.add_argument('--backbone', type=str, default='psp101_cca_ar_mix_distonly_', choices=['resnet', 'xception', 'drn', 'mobilenet'])
    parser.add_argument('--out-stride', type=int, default=16)
    parser.add_argument('--Is_GM', type=bool, default=False, help='Enable the Gate mechanism in test phase')
    parser.add_argument('--dataroot', type=str, default='datasets/BCSS-WSSS/')
    parser.add_argument('--dataset', type=str, default='bcss')
    parser.add_argument('--savepath', type=str, default='checkpoints/')
    parser.add_argument('--workers', type=int, default=0, metavar='N')
    parser.add_argument('--sync-bn', type=bool, default=None)
    parser.add_argument('--freeze-bn', type=bool, default=False)
    parser.add_argument('--loss-type', type=str, default='mvce', choices=['ce', 'mvce'])
    parser.add_argument('--n_class', type=int, default=4)
    # training hyper params
    parser.add_argument('--epochs', type=int, default=10, metavar='N')
    parser.add_argument('--batch-size', type=int, default=20, metavar='N')
    # optimizer params
    parser.add_argument('--lr', type=float, default=5e-3, metavar='LR')
    parser.add_argument('--lr-scheduler', type=str, default='poly',choices=['poly', 'step', 'cos'])
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M')
    parser.add_argument('--weight-decay', type=float, default=5e-4, metavar='M')
    parser.add_argument('--nesterov', action='store_true', default=False)
    # cuda, seed and logging
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--gpu-ids', type=str, default='0')
    parser.add_argument('--seed', type=int, default=1, metavar='S')
    # checking point
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--checkname', type=str, default='deeplab-resnet')
    parser.add_argument('--ft', action='store_true', default=False)
    parser.add_argument('--eval-interval', type=int, default=1)
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    if args.sync_bn is None:
        if args.cuda and len(args.gpu_ids) > 1:
            args.sync_bn = True
        else:
            args.sync_bn = False
    print(args)
    trainer = Trainer(args)
    for epoch in range(trainer.args.epochs):
        # pass
        trainer.training(epoch)
        trainer.validation(epoch)
    trainer.test(epoch, args.Is_GM)
    trainer.writer.close()

if __name__ == "__main__":
    torch.cuda.empty_cache()
    main()
