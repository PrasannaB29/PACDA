import argparse
import os, sys
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import os.path as osp
import torchvision
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import network_bns as network
import loss
from torch.utils.data import DataLoader
from data_list import ImageList
import random, pdb, math, copy
# from tqdm import tqdm
from loss import CrossEntropyLabelSmooth
import math, pickle
from torch.nn import Softmax
from torchvision.utils import save_image
from select_params import calc_bns_loss_batch_cl, calc_bns_loss_batch, calc_bns_loss_single, entropy_method

def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer

def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer

def image_train(resize_size=256, crop_size=224, alexnet=False):
  if not alexnet:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
  else:
    normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
  return  transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

def image_test(resize_size=256, crop_size=224, alexnet=False):
  if not alexnet:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
  else:
    normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
  return  transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normalize
    ])

def data_load(args): 
    ## prepare data
    dsets = {}
    dset_loaders = {}
    train_bs = args.batch_size
    txt_test = open(args.test_dset_path).readlines()
    dsets["test"] = ImageList(txt_test, transform=image_test())
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs, shuffle=True, num_workers=args.worker, drop_last=False)
    return dset_loaders

def data_load_source(args):
    ## prepare data
    dsets = {}
    dset_loaders = {}
    train_bs = args.batch_size
    txt_src = open(args.s_dset_path).readlines()
    txt_test = open(args.test_dset_path).readlines()

    dsize = len(txt_src)
    tr_size = int(0.8 * dsize)
    # print(dsize, tr_size, dsize - tr_size)
    tr_txt, te_txt = torch.utils.data.random_split(txt_src, [tr_size, dsize - tr_size])

    dsets["source_tr"] = ImageList(tr_txt, transform=image_train())
    dset_loaders["source_tr"] = DataLoader(dsets["source_tr"], batch_size=train_bs, shuffle=True,
                                           num_workers=args.worker, drop_last=False)
    dsets["source_te"] = ImageList(te_txt, transform=image_test())
    dset_loaders["source_te"] = DataLoader(dsets["source_te"], batch_size=train_bs, shuffle=True,
                                           num_workers=args.worker, drop_last=False)
    # dsets["test"] = ImageList(txt_test, transform=image_test())
    # dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs * 2, shuffle=True, num_workers=args.worker,
    #                                   drop_last=False)

    return dset_loaders

def load_mask(args, dir, domain_num, layer='F', param_type='w'):
    mask_name = str(domain_num)+'_mask_'+layer+'_'+param_type+'.pkl'
    mask_file = os.path.join(dir, mask_name)
    with open(mask_file, 'rb') as f:
        mask_dict_numpy = pickle.load(f)
    mask_dict = {}
    for key in mask_dict_numpy:
        mask_dict[key] = torch.from_numpy(mask_dict_numpy[key]).cuda()
    return mask_dict

def cal_acc(loader, netF, netB, netC):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            outputs,_ = netF(inputs)
            outputs = netC(netB(outputs))
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    mean_ent = torch.mean(loss.Entropy(nn.Softmax(dim=1)(all_output))).cpu().data.item()
    return accuracy*100, mean_ent

def test_target_bns(args, num, names, folder, dset_loaders):
    """
    Has been written just for source=0 and target=1. Generalize when doing CL
    """
    args.model_dir = osp.join(args.output, str(args.total_domains-1)+names[args.s][0])
    if num==0:
        args.output_dir = osp.join(args.output, names[args.s][0])
    else:
        args.output_dir = osp.join(args.output, str(num)+names[args.s][0])
    
    ## set base network
    netF = []
    netB = []
    for i in range(args.total_domains):
        netF.append(network.ResBase(res_name=args.net).cuda())
        netB.append(network.feat_bootleneck(type=args.classifier, feature_dim=netF[i].in_features, bottleneck_dim=args.bottleneck).cuda())
    
    netC = network.feat_classifier(type=args.layer, class_num = args.class_num, bottleneck_dim=args.bottleneck).cuda()
    cur_dir = []
    cur_dir.append(osp.join(args.output, names[args.s][0]))
    for i in range(args.total_domains-1):
        cur_dir.append(osp.join(args.output, str(i+1)+names[args.s][0]))

    for i in range(args.total_domains):            
        modelpath = cur_dir[args.total_domains-1] + '/' + str(args.total_domains-1)+'_F_pruned.pt'
        # modelpath = '/data3/prasanna/Workspace/BNS_github/clvision_2/ACPR/1A/1_F_pruned.pt'
        netF[i].load_state_dict(torch.load(modelpath))
        modelpath = cur_dir[args.total_domains-1] + '/' + str(args.total_domains-1)+'_B_pruned.pt'
        # modelpath = '/data3/prasanna/Workspace/BNS_github/clvision_2/ACPR/1A/1_B_pruned.pt'
        netB[i].load_state_dict(torch.load(modelpath))
        netF[i].eval()
        netB[i].eval()
    modelpath = cur_dir[args.total_domains-1] + '/' + str(args.total_domains-1)+'_C_pruned.pt'
    # modelpath = '/data3/prasanna/Workspace/BNS_github/clvision_2/ACPR/A/0_C_pruned.pt'
    netC.load_state_dict(torch.load(modelpath))
    netC.eval()

    # Load masks starts
    # only till 3rd is done because last model doesn't require a mask
    mask_dict_F_w = []
    mask_dict_F_b = []
    mask_dict_B_w = []
    mask_dict_B_b = []
    for i in range(args.total_domains):
        mask_dict_F_w.append(load_mask(args, cur_dir[i], i, 'F', 'w'))
        mask_dict_F_b.append(load_mask(args, cur_dir[i], i, 'F', 'b'))
        mask_dict_B_w.append(load_mask(args, cur_dir[i], i, 'B', 'w'))
        mask_dict_B_b.append(load_mask(args, cur_dir[i], i, 'B', 'b'))
    # Load masks ends

    # obtaining source model using source mask starts
    # only till 3rd is done because last model doesn't require a mask
    for i in range(args.total_domains):
        for k, v in netF[i].named_parameters():
            if (k.endswith("weight")) and ("bn" not in k):
                tmp = k[:-7]  # because ".weight" has 7 characters
                v.data = v.data * mask_dict_F_w[i][tmp]

            elif (k.endswith("bias")) and ("bn" not in k):
                tmp = k[:-5]
                v.data = v.data * mask_dict_F_b[i][tmp]

        for k, v in netB[i].named_parameters():
            if (k.endswith("weight")) and ("bn" not in k):
                tmp = k[:-7]  # because ".weight" has 7 characters
                v.data = v.data * mask_dict_B_w[i][tmp]

            elif (k.endswith("bias")) and ("bn" not in k):
                tmp = k[:-5]
                v.data = v.data * mask_dict_B_b[i][tmp]
    
    for i in range(args.total_domains):
        tmpnum=1
        for n,m in netF[i].named_modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                bn_dir = osp.join(cur_dir[i],str(i)+'_F_BN'+str(tmpnum)+'.pt')
                source_bn = torch.load(bn_dir)
                m.load_state_dict(source_bn)
                tmpnum+=1
        tmpnum=1
        for n,m in netB[i].named_modules():
            if isinstance(m, torch.nn.BatchNorm1d):
                bn_dir = osp.join(cur_dir[i],str(i)+'_B_BN'+str(tmpnum)+'.pt')
                source_bn = torch.load(bn_dir)
                m.load_state_dict(source_bn)
                tmpnum+=1
    # obtaining source model using source mask ends

    del mask_dict_F_w
    del mask_dict_F_b
    del mask_dict_B_w
    del mask_dict_B_b
    
    # Accuracy using domain-id
    if num==args.s:
        loader = dset_loaders['source_te']
    else:
        loader = dset_loaders['test']

    for i in range(len(args.names)):
        did_accuracy,_ = cal_acc(loader, netF[i], netB[i], netC)
        print_lines(args, "Accuracy using parameters of domain {} is {}".format(i, did_accuracy))
    return

def print_lines(args, p):
    print(p)
    args.out_file.write(p+'\n')
    args.out_file.flush()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SHOT')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--s', type=int, default=0, help="source")
    parser.add_argument('--t', type=int, default=1, help="target")
    parser.add_argument('--max_epoch', type=int, default=20, help="max iterations")
    parser.add_argument('--batch_size', type=int, default=64, help="batch_size")
    parser.add_argument('--worker', type=int, default=4, help="number of workers")
    parser.add_argument('--dset', type=str, default='office-home', choices=['VISDA-C', 'office', 'office-home', 'office-caltech'])
    parser.add_argument('--lr', type=float, default=1e-2, help="learning rate")
    parser.add_argument('--net', type=str, default='resnet50', help="vgg16, resnet50, resnet101")
    parser.add_argument('--seed', type=int, default=2020, help="random seed")
    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--epsilon', type=float, default=1e-5)
    parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument('--smooth', type=float, default=0.1)   
    parser.add_argument('--output', type=str, default='san')
    parser.add_argument('--da', type=str, default='uda', choices=['uda', 'pda', 'oda'])
    parser.add_argument('--trte', type=str, default='val', choices=['full', 'val'])
    parser.add_argument('--method', type=str, default="batch-squareddist", choices=["batch-squareddist",
    "single-squareddist", "single-gaussian","single-augmentation","entropy"])
    args = parser.parse_args()

    # print("Current device is {}".format(torch.cuda.current_device()))
    if args.dset == 'office-home':
        # names = ['Art', 'Clipart', 'Product', 'RealWorld']
        # names = ['Clipart', 'Product', 'RealWorld', 'Art']
        # names = ['Clipart', 'Art', 'Product', 'RealWorld']
        # names = ['Product', 'Art', 'Clipart', 'RealWorld']
        # names = ['RealWorld', 'Product', 'Clipart', 'Art']
        names = ['RealWorld', 'Art', 'Clipart', 'Product']
        args.class_num = 65 
    if args.dset == 'office':
        names = ['amazon', 'dslr', 'webcam']
        args.class_num = 31
    if args.dset == 'VISDA-C':
        names = ['train', 'validation']
        args.class_num = 12
    if args.dset == 'office-caltech':
        names = ['amazon', 'caltech', 'dslr', 'webcam']
        args.class_num = 10

    #os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    # torch.backends.cudnn.deterministic = True

    folder = '/data3/prasanna/Datasets/Office_home/'
    #TODO: start - Carefully edit these lines for each run
    args.total_domains = 4

    args.s_dset_path = folder + args.dset + '/' + names[args.s] + '_list.txt'
    args.test_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'

    # dset_loaders = data_load_source(args)
    dset_loaders = torch.load(os.path.join(args.output, 'dset_loaders_source.pt'))

    args.out_file = open(osp.join(args.output, 'Results_allparams_alldomains.txt'), 'w')

    print_lines(args, "RACP - Source : 60% ; Target : each 10%\n")
    #TODO: start - Carefully edit these lines for each run
    args.names = names
    for i in range(len(names)):
        args.t = i
        
        args.test_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'

        args.name = names[args.s][0].upper() + names[args.t][0].upper()

        print_line = "Inference on domain {}".format(names[i])
        print_lines(args, print_line)

        if i!=args.s:
            dset_loaders = data_load(args)
        
        test_target_bns(args, i, names, folder, dset_loaders)
        print_lines(args, "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")