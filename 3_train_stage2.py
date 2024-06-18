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
from network.UNet import UNet
from network.pspnet import PSPNet
from network.UCTransNet import UCTransNet
from network.TransAttUnet import UNet_Attention_Transformer_Multiscale
# from network.TransUNet import VisionTransformer, CONFIGS
from tool.loss import SegmentationLosses, KDLoss, PKT, HintLoss, Correlation, RKDLoss
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

# seed = 3407
# random.seed(seed)
# np.random.seed(seed)
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)

def get_CTranS_config():
    config = ml_collections.ConfigDict()
    config.transformer = ml_collections.ConfigDict()
    config.KV_size = 960  # KV_size = Q1 + Q2 + Q3 + Q4
    config.transformer.num_heads  = 4
    config.transformer.num_layers = 4
    config.expand_ratio           = 8  # MLP channel dimension expand ratio
    config.transformer.embeddings_dropout_rate = 0.1
    config.transformer.attention_dropout_rate = 0.1
    config.transformer.dropout_rate = 0
    config.patch_sizes = [16,8,4,2]
    config.base_channel = 64 # base channel of U-Net
    config.n_classes = 1
    return config

def get_base_config():
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (16, 16)})
    config.hidden_size = 768
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 3072
    config.transformer.num_heads = 12
    config.transformer.num_layers = 12
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1

    config.classifier = 'seg'
    config.patch_size = 16

    config.decoder_channels = (256, 128, 64, 16)
    config.n_classes = 4
    config.activation = 'softmax'
    return config

def get_TFCNs_config():
    config = get_base_config()
    config.patches.grid = (16, 16)
    config.resnet = ml_collections.ConfigDict()
    config.resnet.num_layers = (3, 4, 9)
    config.resnet.width_factor = 1

    config.decoder_channels = (256, 128, 64, 16)
    config.skip_channels = [512, 256, 64, 16]
    config.n_classes = 4
    config.n_skip = 3
    config.activation = 'softmax'

    return config

def voting(outputs_main, outputs_aux1, outputs_aux2, mask, ep):
    n = outputs_main.shape[0]
    e = math.sqrt(ep)
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
        # loss_select += 1*torch.sum(loss_sample[agree_aux]) + (0)*torch.sum(loss_sample[disagree_aux])
        # loss_select += math.exp(-e)*torch.sum(loss_sample[agree_aux]) + (1 / math.exp(-e))*torch.sum(loss_sample[disagree_aux])
        # loss_select += 2*torch.sum(loss_sample[agree_aux]) + (1 / math.pow(2, e))*torch.sum(loss_sample[disagree_aux])

    return loss_select / (n*loss_main.shape[1])

def joint_optimization(outputs_main, outputs_aux1, outputs_aux2, mask, kd_weight, kd_T, vote_weight):
    kd_loss = KDLoss(T=kd_T)
    avg_aux = (outputs_aux1 + outputs_aux2) / 2

    L_kd = kd_loss(outputs_main.permute(0, 2, 3, 1).reshape(-1, 4),
                   avg_aux.permute(0, 2, 3, 1).reshape(-1, 4))
    L_vote = voting(outputs_main, outputs_aux1, outputs_aux2, mask)
    # L_urn = SegmentationLosses(weight=None, cuda=args.cuda).build_loss(mode=args.loss_type)(outputs_main, mask, gama)
    L = vote_weight * L_vote + kd_weight * L_kd
    return L

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
        # model = UNet(n_classes=self.nclass)
        # model = PSPNet(n_classes=self.nclass)
        # config_vit = get_CTranS_config()
        # model = UCTransNet(config=config_vit, n_classes=self.nclass)
        # model = UNet_Attention_Transformer_Multiscale(n_channels=3, n_classes=self.nclass)
        # model = logo(num_classes=self.nclass, img_size=224)
        # model = SwinTransformer(num_classes=self.nclass)
        # config_TFCN = get_TFCNs_config()
        # config_TFCN.patches.grid = (int(224 / 16), int(224 / 16))
        # model = TFCNs(config=config_TFCN, num_classes=self.nclass)
        # model = MTUNet(out_ch=self.nclass)
        # model = ScaleFormer(n_classes=self.nclass)
        # model = ft_unetformer(num_classes=self.nclass, pretrained=True)

        model = smp.PSPNet(encoder_name='timm-resnest101e', encoder_weights='imagenet', in_channels=3, classes=self.nclass)
        # model = smp.Unet(encoder_name='mit_b1', encoder_weights='imagenet', in_channels=3, classes=self.nclass)
        # model = smp.PSPNet(encoder_name='resnet50', encoder_weights='imagenet', in_channels=3, classes=self.nclass)
        # model = smp.DeepLabV3Plus(encoder_name='resnext101_32x8d', encoder_weights='imagenet', in_channels=3, classes=self.nclass)

        # model = DeepLabV2_ResNet101_MSC(n_classes=self.nclass, n_blocks=[3, 4, 23, 3], atrous_rates=[6, 12, 18, 24])
        # config_vit = CONFIGS[args.vit_name]
        # config_vit.n_classes = self.nclass
        # if args.vit_name.find('R50') != -1:
        #     config_vit.patches.grid = (int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))
        # model = VisionTransformer(config=config_vit,  num_classes=config_vit.n_classes)
        # train_params = [{'params': model.get_1x_lr_params(), 'lr': args.lr},
        #                 {'params': model.get_10x_lr_params(), 'lr': args.lr * 10}]
        lr = args.lr
        optimizer = torch.optim.SGD(params=model.parameters(), lr=lr, momentum=args.momentum,
                                    weight_decay=args.weight_decay)
        # optimizer = torch.optim.AdamW(params=model.parameters(), lr=lr)
        self.criterion = SegmentationLosses(weight=None, cuda=args.cuda).build_loss(mode=args.loss_type)
        self.model, self.optimizer = model, optimizer
        self.evaluator = Evaluator(self.nclass)
        self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr,
                                            args.epochs, len(self.train_loader))

        ##  Create ResNet38 and load the weights of Stage 1.
        import importlib
        model_stage1 = getattr(importlib.import_module('network.resnet38_cls'), 'Net_CAM')(n_class=4)
        resume_stage1 = '/root/autodl-tmp/project/WSSS-Second/checkpoints/stage1_checkpoint_trained_on_'+str(args.dataset)+'_res38d_pda'+'.pth'
        # resume_stage1 = 'checkpoints/stage1_checkpoint_trained_on_'+str(args.dataset)+'.pth'
        weights_dict = torch.load(resume_stage1)
        model_stage1.load_state_dict(weights_dict)
        self.model_stage1 = model_stage1.cuda()
        self.model_stage1.eval()

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
            # print(image1.shape, target1.shape)
            # print(target.shape)
            if self.args.cuda:
                image, target, target_a, target_b = image.cuda(), target.cuda(), target_a.cuda(), target_b.cuda()
            self.scheduler(self.optimizer, i, epoch, self.best_pred)

            self.optimizer.zero_grad()

            # print(image.shape)
            output = self.model(image)
            output2 = self.model(image)
            output3 = self.model(image)
            # print(output, output2)
            # print(output)
            # num_samples = 30  # 设置Monte Carlo采样次数
            # predictions = []
            # for _ in range(num_samples):
            #     with torch.no_grad():
            #         uncer_output = self.model(image)  # 替换为你的输入图像
            #         predictions.append(uncer_output)
            # predictions = torch.stack(predictions)
            # uncertainty_map = predictions.std(dim=0)
            # print(uncertainty_map[:, 0, :, :].shape)
            # # uncertainty_map = uncertainty_map.cpu().numpy()
            # # print(uncertainty_map)
            # uncertainty_threshold = 0.5
            # mask = (uncertainty_map < uncertainty_threshold)
            # masked_output = output * mask
            # # print(masked_output.shape)
            # loss_uncer = F.cross_entropy(uncertainty_map, target.long(), reduction='mean')



            # print(uncertainty_map.shape)
            # print(output.shape)
            one = torch.ones((output.shape[0],1,224,224)).cuda()
            one2 = torch.ones((output2.shape[0],1,224,224)).cuda()
            one3 = torch.ones((output3.shape[0],1,224,224)).cuda()
            # one = torch.ones((output[0].shape[0],1,224,224)).cuda()
            # print(output)
            # print((100 * one * (target==4).unsqueeze(dim = 1)).shape)
            # output = torch.cat([output,(100 * one * (target==4).unsqueeze(dim = 1))],dim = 1)
            # output2 = torch.cat([output2,(100 * one2 * (target==4).unsqueeze(dim = 1))],dim = 1)
            # output3 = torch.cat([output3,(100 * one3 * (target==4).unsqueeze(dim = 1))],dim = 1)

            # print(output.shape, target.shape)
            # print(output[:, 0, :, :], output[:, 1, :, :])
            # loss_o = self.criterion(output[:, 0, :, :], target)
            # weight_mask = torch.clamp(output, min=0.0)

            loss_o = self.criterion(output, target, self.gama)
            # loss_o2 = self.criterion(output2, target, self.gama)
            # loss_o3 = self.criterion(output3, target, self.gama)
            # loss_o2 = self.criterion(output2, target, self.gama)
            # loss_f = SegmentationLosses(weight=None, cuda=0).build_loss('focal')(output, target)

            # loss_a = self.criterion(output, target_a, self.gama)
            # loss_b = self.criterion(output, target_b, self.gama)
            loss_v1 = voting(output, output2, output3, target, epoch)
            # loss_v2 = voting(output2, output, output3, target, epoch)
            # loss_v3 = voting(output3, output, output2, target, epoch)
            # loss_v = (loss_v1 + loss_v2 + loss_v3) / 3
            # loss = 0.7*loss_o+0.2*loss_a+0.1*loss_b
            # loss_ht1 = HintLoss()(output.permute(0, 2, 3, 1).reshape(-1, 4), output2.permute(0, 2, 3, 1).reshape(-1, 4))
            # loss_ht2 = HintLoss()(output.permute(0, 2, 3, 1).reshape(-1, 4), output3.permute(0, 2, 3, 1).reshape(-1, 4))
            # loss_ht3 = HintLoss()(output2, output3)
            # loss_ht = (loss_ht1 + loss_ht2) / 2
            # loss_ht = (loss_ht1 + loss_ht2 + loss_ht3) / 3

            loss_kd1 = RKDLoss()(output, output2)
            loss_kd2 = RKDLoss()(output, output3)
            loss_kd = (loss_kd1 + loss_kd2) / 2
            # loss_kl1 = KDLoss(T=30)(output.permute(0, 2, 3, 1).reshape(-1, 4), output2.permute(0, 2, 3, 1).reshape(-1, 4))
            # loss_kl2 = KDLoss(T=30)(output.permute(0, 2, 3, 1).reshape(-1, 4), output3.permute(0, 2, 3, 1).reshape(-1, 4))
            # loss_kl = (loss_kl1 + loss_kl2) / 2
            # print(loss_kd.item())
            # print(loss_v1.item(), loss_kd.item())
            loss = 1*loss_v1 + 0.1*loss_kd
            # loss = 1*loss_o + 0.1*loss_kd
            # loss = 0.5*loss_v + 0.5*loss_cc
            # loss = 0.5*loss_d + 0.5*loss_ht3
            # loss = loss_o
            # loss = 0.6 * loss_o + 0.2 * loss_a + 0.2 * loss_b
            # loss = (loss_o + loss_o2 + loss_o3) / 3 + (loss_kd + loss_kd2 + loss_kd3) / 3
            # print(loss_o, loss_a, loss_b)
            # predicted_probabilities = F.softmax(output, dim=1)
            # print(predicted_probabilities)
            # loss_map = loss_o.detach().cpu()
            # print(loss_map)

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
            # pred = output[0].data.cpu().numpy()
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
        checkpoint = torch.load('/root/autodl-tmp/project/WSSS-Second/checkpoints/stage2_checkpoint_trained_on_'+self.args.dataset+self.args.backbone+self.args.loss_type+'.pth')
        # checkpoint = torch.load('/root/autodl-tmp/project/WSSS-Second/checkpoints/stage2_checkpoint_trained_on_bcsspsp101_mix_a2b0.5_3_mvce.pth')
        self.model.module.load_state_dict(checkpoint['state_dict'], strict=False)
    def test(self, epoch, Is_GM):
        self.load_the_best_checkpoint()
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.test_loader, desc='\r')
        test_loss = 0.0
        for i, sample in enumerate(tbar):
            image, target = sample[0]['image'], sample[0]['label']
            # print(target.shape)
            # print(sample)
            image_name = sample[-1][0].split('/')[-1].replace('.png', '')
            # print(image_name, "----")
            # print(image_name)
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
            # pred = output[0].data.cpu().numpy()
            # print(pred.shape)
            if Is_GM:
                pred = pred*(pred_cls.unsqueeze(dim=2).unsqueeze(dim=3).numpy())
            target = target.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            # print(pred)
            ## cls 4 is exclude
            pred[target==4]=4
            # print(pred[0])
            colored_image = Image.new("RGB", pred[0].shape)
            if self.args.dataset == 'bcss':
                for i in range(pred[0].shape[0]):
                    for j in range(pred[0].shape[1]):
                        if pred[0][i, j] == 0:
                            colored_image.putpixel((i, j), (255, 0, 0))
                        elif pred[0][i, j] == 1:
                            colored_image.putpixel((i, j), (0, 255, 0))
                        elif pred[0][i, j] == 2:
                            colored_image.putpixel((i, j), (0, 0, 255))
                        elif pred[0][i, j] == 3:
                            colored_image.putpixel((i, j), (153, 0, 255))
                        elif pred[0][i, j] == 4:
                            colored_image.putpixel((i, j), (255, 255, 255))
            elif self.args.dataset == 'luad':
                for i in range(pred[0].shape[0]):
                    for j in range(pred[0].shape[1]):
                        if pred[0][i, j] == 0:
                            colored_image.putpixel((i, j), (205, 51, 51))
                        elif pred[0][i, j] == 1:
                            colored_image.putpixel((i, j), (0, 255, 0))
                        elif pred[0][i, j] == 2:
                            colored_image.putpixel((i, j), (65, 105, 225))
                        elif pred[0][i, j] == 3:
                            colored_image.putpixel((i, j), (255, 165, 0))
                        elif pred[0][i, j] == 4:
                            colored_image.putpixel((i, j), (255, 255, 255))
            colored_image = colored_image.rotate(90, expand=True)
            colored_image = colored_image.transpose(Image.FLIP_TOP_BOTTOM)
            # print(colored_image)
            save_path = f'/root/autodl-tmp/project/WSSS-Second/outputs/{self.args.dataset}/{self.args.backbone}{self.args.loss_type}'
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            colored_image.save(save_path + f'/{image_name}.png')
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
    # parser.add_argument('--Is_MV', type=bool, default=True, help='Enable the Monte Carlo Augmentation in test phase')
    parser.add_argument('--dataroot', type=str, default='/root/autodl-tmp/project/WSSS-Second/datasets/BCSS-WSSS/')
    parser.add_argument('--dataset', type=str, default='bcss')
    parser.add_argument('--savepath', type=str, default='/root/autodl-tmp/project/WSSS-Second/checkpoints/')
    parser.add_argument('--workers', type=int, default=1, metavar='N')
    parser.add_argument('--sync-bn', type=bool, default=None)
    parser.add_argument('--freeze-bn', type=bool, default=False)
    parser.add_argument('--loss-type', type=str, default='mvce', choices=['ce', 'focal'])
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
    # parser.add_argument('--resume', type=str, default='/root/autodl-tmp/project/WSSS-Second/init_weights/deeplab-resnet.pth.tar')
    # parser.add_argument('--resume', type=str, default='/root/autodl-tmp/project/WSSS-Second/checkpoints/stage2_checkpoint_trained_on_bcsspsp101_cca_ar_mix_a2b0.5_mvce.pth')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--checkname', type=str, default='deeplab-resnet')
    parser.add_argument('--ft', action='store_true', default=False)
    parser.add_argument('--eval-interval', type=int, default=1)

    parser.add_argument('--vit_patches_size', type=int,
                        default=16, help='vit_patches_size, default is 16')
    parser.add_argument('--vit_name', type=str,
                        default='R50-ViT-B_16', help='select one vit model')
    parser.add_argument('--img_size', type=int,
                        default=224, help='input patch size of network input')
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
