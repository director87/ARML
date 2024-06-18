import os
import torch
import argparse
import importlib
from torch.backends import cudnn
cudnn.enabled = True
from tool.infer_fun import create_pseudo_mask

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default='/root/autodl-tmp/project/WSSS-Second/checkpoints/stage1_checkpoint_trained_on_bcss_res38d_cca_ar.pth', type=str)
    parser.add_argument("--network", default="/root/autodl-tmp/project/WSSS-Second/network.resnet38_cls", type=str)
    parser.add_argument("--dataroot", default="/root/autodl-tmp/project/WSSS-Second/datasets/BCSS-WSSS", type=str)
    parser.add_argument("--dataset", default="bcss", type=str)
    parser.add_argument("--num_workers", default=1, type=int)
    parser.add_argument("--n_class", default=4, type=int)

    args = parser.parse_args()
    print(args)
    if args.dataset == 'luad':
        palette = [0]*15
        palette[0:3] = [205,51,51]
        palette[3:6] = [0,255,0]
        palette[6:9] = [65,105,225]
        palette[9:12] = [255,165,0]
        palette[12:15] = [255, 255, 255]
    elif args.dataset == 'bcss':
        palette = [0]*15
        palette[0:3] = [255, 0, 0]
        palette[3:6] = [0,255,0]
        palette[6:9] = [0,0,255]
        palette[9:12] = [153, 0, 255]
        palette[12:15] = [255, 255, 255]
    PMpath = os.path.join(args.dataroot,'train_PM')
    if not os.path.exists(PMpath):
        os.mkdir(PMpath)
    model = getattr(importlib.import_module("network.resnet38_cls"), 'Net_CAM')(n_class=args.n_class)
    model.load_state_dict(torch.load(args.weights), strict=False)
    model.eval()
    model.cuda()
    #
    # fm = 'b4_5'
    # savepath = os.path.join(PMpath,'PM_'+'res38d_pda'+fm)
    # if not os.path.exists(savepath):
    #     os.mkdir(savepath)
    # create_pseudo_mask(model, args.dataroot, fm, savepath, args.n_class, palette, args.dataset)
    # ##
    # fm = 'b5_2'
    # savepath = os.path.join(PMpath,'PM_'+'res38d_pda'+fm)
    # if not os.path.exists(savepath):
    #     os.mkdir(savepath)
    # create_pseudo_mask(model, args.dataroot, fm, savepath, args.n_class, palette, args.dataset)
    #
    fm = 'bn7'
    savepath = os.path.join(PMpath,'PM_'+'res38d_cca_ar'+fm)
    if not os.path.exists(savepath):
        os.mkdir(savepath)
    create_pseudo_mask(model, args.dataroot, fm, savepath, args.n_class, palette, args.dataset)