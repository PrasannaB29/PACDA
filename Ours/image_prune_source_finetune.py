import argparse
import os, sys
import os.path as osp
import torchvision
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import network, loss
from torch.utils.data import DataLoader
from data_list import ImageList, ImageList_idx
import random, pdb, math, copy
from loss import CrossEntropyLabelSmooth
import torch.nn.utils.prune as prune
import pickle
import copy

def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer

def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        # param_group['weight_decay'] = 1e-3
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
    txt_src = open(args.s_dset_path).readlines()
    txt_test = open(args.test_dset_path).readlines()

    if args.trte == "val":
        dsize = len(txt_src)
        tr_size = int(0.8 * dsize)
        # print(dsize, tr_size, dsize - tr_size)
        tr_txt, te_txt = torch.utils.data.random_split(txt_src, [tr_size, dsize - tr_size])
    else:
        dsize = len(txt_src)
        tr_size = int(0.8 * dsize)
        _, te_txt = torch.utils.data.random_split(txt_src, [tr_size, dsize - tr_size])
        tr_txt = txt_src

    dsets["source_tr"] = ImageList(tr_txt, transform=image_train())
    dset_loaders["source_tr"] = DataLoader(dsets["source_tr"], batch_size=train_bs, shuffle=True,
                                           num_workers=args.worker, drop_last=False)
    dsets["source_te"] = ImageList(te_txt, transform=image_test())
    dset_loaders["source_te"] = DataLoader(dsets["source_te"], batch_size=train_bs, shuffle=True,
                                           num_workers=args.worker, drop_last=False)
    dsets["test"] = ImageList(txt_test, transform=image_test())
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs * 2, shuffle=True, num_workers=args.worker,
                                      drop_last=False)

    return dset_loaders

def cal_acc(loader, netF, netB, netC):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            outputs = netC(netB(netF(inputs)))
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    mean_ent = torch.mean(loss.Entropy(all_output)).cpu().data.item()

    return accuracy * 100, mean_ent

def save_mask(args, mask_dict, dir, domain_num, layer='F', param_type='w'):
    mask_dict_cpy = copy.deepcopy(mask_dict)
    for key in mask_dict:
        mask_dict_cpy[key] = mask_dict[key].cpu().numpy()
    mask_name = str(domain_num)+'_mask_'+layer+'_'+param_type+'.pkl'
    file = osp.join(dir, mask_name)
    with open(file, 'wb') as f:
        pickle.dump(mask_dict_cpy,f)

def prune_source(args):
    ## set base network  
    netF = network.ResBase(res_name=args.net).cuda()
    netB = network.feat_bootleneck(type=args.classifier, feature_dim=netF.in_features, bottleneck_dim=args.bottleneck).cuda()
    netC = network.feat_classifier(type=args.layer, class_num = args.class_num, bottleneck_dim=args.bottleneck).cuda()

    modelpath = args.output_dir_src + '/' + '0_F.pt'
    netF.load_state_dict(torch.load(modelpath))
    modelpath = args.output_dir_src + '/' + '0_B.pt'
    netB.load_state_dict(torch.load(modelpath))
    modelpath = args.output_dir_src + '/' + '0_C.pt'
    netC.load_state_dict(torch.load(modelpath))
    netC.eval()

    # Pruning of netF starts
    mask_dict_F_w = {}
    mask_dict_F_b = {}
    for name, module in netF.named_modules():
        f=0
        if isinstance(module, torch.nn.Conv2d):
            prune.l1_unstructured(module, name='weight', amount=args.pf_c) # For unstructured pruning
            f=1
        # prune 40% of connections in all linear layers
        elif isinstance(module, torch.nn.BatchNorm2d):
            prune.l1_unstructured(module, name='weight', amount=args.pf_bn)
            prune.l1_unstructured(module, name='bias', amount=args.pf_bn)
            f=2
        if f==1:
            mask_dict_F_w[name] = dict(module.named_buffers())['weight_mask']
            prune.remove(module, 'weight')
        elif f==2:
            mask_dict_F_w[name] = dict(module.named_buffers())['weight_mask']
            mask_dict_F_b[name] = dict(module.named_buffers())['bias_mask']
            prune.remove(module, 'weight')
            prune.remove(module, 'bias')
    # Pruning of netF ends

    # Pruning of netB starts
    mask_dict_B_w = {}
    mask_dict_B_b = {}
    for name, module in netB.named_modules():
        f = 0
        if isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=args.pf_c)
            prune.l1_unstructured(module, name='bias', amount=args.pf_bn)
            f = 1
        elif isinstance(module, torch.nn.BatchNorm1d):
            prune.l1_unstructured(module, name='weight', amount=args.pf_bn)
            prune.l1_unstructured(module, name='bias', amount=args.pf_bn)
            f = 1
        if f == 1:
            mask_dict_B_w[name] = dict(module.named_buffers())['weight_mask']
            mask_dict_B_b[name] = dict(module.named_buffers())['bias_mask']
            prune.remove(module, 'weight')
            prune.remove(module, 'bias')
    # Pruning of netB ends

    # Save mask as pickle file - start
    save_mask(args, mask_dict_F_w, args.output_dir_src, 0, 'F', 'w')
    save_mask(args, mask_dict_F_b, args.output_dir_src, 0, 'F', 'b')
    save_mask(args, mask_dict_B_w, args.output_dir_src, 0, 'B', 'w')
    save_mask(args, mask_dict_B_b, args.output_dir_src, 0, 'B', 'b')

    mask_list = list([mask_dict_F_w, mask_dict_F_b, mask_dict_B_w, mask_dict_B_b])
    return netF, netB, netC, mask_list

def finetune_source(args, netF, netB, netC, mask_list, dset_loaders):
    mask_dict_F_w, mask_dict_F_b, mask_dict_B_w, mask_dict_B_b = mask_list

    for k, v in netC.named_parameters():
        v.requires_grad = False

    param_group = []
    learning_rate = args.lr
    for k, v in netF.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate * 0.1}]
    for k, v in netB.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate}]
    for k, v in netC.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate}]

    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    acc_init = 0
    max_iter = args.max_epoch * len(dset_loaders["source_tr"])
    interval_iter = max_iter // 10
    iter_num = 0

    netF.train()
    netB.train()
    netC.train()

    acc_list = list()

    tmp_before_F_w = 0
    tmp_before_F_b = 0
    tmp_before_B_w = 0
    tmp_before_B_b = 0

    while iter_num < max_iter:
        try:
            inputs_source, labels_source = iter_source.next()
        except:
            iter_source = iter(dset_loaders["source_tr"])
            inputs_source, labels_source = iter_source.next()

        if inputs_source.size(0) == 1:
            continue

        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)

        inputs_source, labels_source = inputs_source.cuda(), labels_source.cuda()
        outputs_source = netC(netB(netF(inputs_source)))
        classifier_loss = CrossEntropyLabelSmooth(num_classes=args.class_num, epsilon=args.smooth)(outputs_source,
                                                                                                   labels_source)
        optimizer.zero_grad()
        classifier_loss.backward()

        # Gradient masking starts
        # for k, v in netF.named_parameters():
        #     if k.endswith("weight"):
        #         print("grad pruned layer = " + k)
        #         # print("Grad before grad_pruning")
        #         # print(v.grad)
        #         tmp = k[:-7]  # because ".weight" has 7 characters
        #         # print("Masked used is ")
        #         # print(mask_dict[tmp])
        #         v.grad = torch.mul(v.grad, mask_dict_F_w[tmp])
        #         with torch.no_grad():
        #             tmp_before_F_w += torch.sum(torch.mul(v.data, 1 - mask_dict_F_w[tmp]))
        #         print("Grad after grad_pruning")
        #         print(tmp_before_F_w)
        #     elif k.endswith("bias"):
        #         tmp = k[:-5]
        #         print("grad pruned layer = " + k)
        #         # print("Grad before grad_pruning")
        #         # print(v.grad)
        #         v.grad = torch.mul(v.grad, mask_dict_F_b[tmp])
        #         with torch.no_grad():
        #             tmp_before_F_b += torch.sum(torch.mul(v.data, 1 - mask_dict_F_b[tmp]))
        #         print("Grad after grad_pruning")
        #         print(tmp_before_F_b)
        #     else:
        #         print(" Should not happen "+k)

        # for k, v in netB.named_parameters():
        #     if k.endswith("weight"):
        #         tmp = k[:-7]  # because ".weight" has 7 characters
        #         print("grad pruned layer = " + k)
        #         # print("Grad before grad_pruning")
        #         # print(v.grad)
        #         v.grad = torch.mul(v.grad, mask_dict_B_w[tmp])
        #         with torch.no_grad():
        #             tmp_before_B_w += torch.sum(torch.mul(v.data, 1 - mask_dict_B_w[tmp]))
        #         print("Grad after grad_pruning")
        #         print(tmp_before_B_w)
        #     elif k.endswith("bias"):
        #         tmp = k[:-5]
        #         print("grad pruned layer = " + k)
        #         # print("Grad before grad_pruning")
        #         # print(v.grad)
        #         v.grad = torch.mul(v.grad, mask_dict_B_b[tmp])
        #         with torch.no_grad():
        #             tmp_before_B_b += torch.sum(torch.mul(v.data, 1 - mask_dict_B_b[tmp]))
        #         print("Grad after grad_pruning")
        #         print(tmp_before_B_b)
        #     else:
        #         print(" Should not happen "+k)

        for k, v in netF.named_parameters():
            if k.endswith("weight"):
                tmp = k[:-7]  # because ".weight" has 7 characters
                v.grad = torch.mul(v.grad, mask_dict_F_w[tmp])
            elif k.endswith("bias"):
                tmp = k[:-5]
                v.grad = torch.mul(v.grad, mask_dict_F_b[tmp])
            else:
                print(" Should not happen "+k)

        for k, v in netB.named_parameters():
            if k.endswith("weight"):
                tmp = k[:-7]  # because ".weight" has 7 characters
                v.grad = torch.mul(v.grad, mask_dict_B_w[tmp])
            elif k.endswith("bias"):
                tmp = k[:-5]
                v.grad = torch.mul(v.grad, mask_dict_B_b[tmp])
            else:
                print(" Should not happen "+k)
        # Gradient masking ends

        optimizer.step()

        netF.eval()
        netB.eval()
        netC.eval()
        
        # acc_s_te, _ = cal_acc(dset_loaders['source_te'], netF, netB, netC)
        # acc_list.append(acc_s_te)

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            acc_s_te, _ = cal_acc(dset_loaders['source_te'], netF, netB, netC)
            log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(args.name_src, iter_num, max_iter, acc_s_te)
            args.out_file.write(log_str + '\n')
            args.out_file.flush()
            print(log_str + '\n')

            if acc_s_te >= acc_init:
                acc_init = acc_s_te
                best_netF = netF.state_dict()
                best_netB = netB.state_dict()
                best_netC = netC.state_dict()
                
                f_bn_list = []
                b_bn_list = []
                for _, module in netF.named_modules():
                    if isinstance(module, torch.nn.BatchNorm2d):
                        f_bn_list.append(module.state_dict())
                for _, module in netB.named_modules():
                    if isinstance(module, torch.nn.BatchNorm1d):
                        b_bn_list.append(module.state_dict())

        netF.train()
        netB.train()
        netC.train()

        print("Iteration num " + str(iter_num) + " done")

    # acc_list_array = np.asarray(acc_list)
    # np.save(osp.join(args.output_dir_src, "pruned_acc_list.npy"))    
    torch.save(best_netF, osp.join(args.output_dir_src, "0_F_pruned.pt"))#CL
    torch.save(best_netB, osp.join(args.output_dir_src, "0_B_pruned.pt"))#CL
    torch.save(best_netC, osp.join(args.output_dir_src, "0_C_pruned.pt"))#CL
    
    for i, bn in enumerate(f_bn_list):
        torch.save(bn, osp.join(args.output_dir_src, "0_F_BN"+str(i+1)+".pt"))
    for i, bn in enumerate(b_bn_list):
        torch.save(bn, osp.join(args.output_dir_src, "0_B_BN"+str(i+1)+".pt"))

    # tmpnum=1
    # for _, module in netF.named_modules():
    #     if isinstance(module, torch.nn.BatchNorm2d):
    #         torch.save(module.state_dict(), osp.join(args.output_dir_src, "0_BN"+str(tmpnum)+".pt"))
    #         tmpnum+=1
    return netF, netB, netC

def test_target(args, dset_loaders, domain_num):
    ## set base network
    netF = network.ResBase(res_name=args.net).cuda()
    netB = network.feat_bootleneck(type=args.classifier, feature_dim=netF.in_features,
                                   bottleneck_dim=args.bottleneck).cuda()
    netC = network.feat_classifier(type=args.layer, class_num=args.class_num, bottleneck_dim=args.bottleneck).cuda()

    args.modelpath = args.output_dir_src + '/0_F_pruned.pt'
    netF.load_state_dict(torch.load(args.modelpath))
    args.modelpath = args.output_dir_src + '/0_B_pruned.pt'
    netB.load_state_dict(torch.load(args.modelpath))
    args.modelpath = args.output_dir_src + '/0_C_pruned.pt'
    netC.load_state_dict(torch.load(args.modelpath))
    netF.eval()
    netB.eval()
    netC.eval()

    if domain_num == args.s:
        loader = dset_loaders['source_te']
    else:
        loader = dset_loaders['test']
    acc, _ = cal_acc(loader, netF, netB, netC)
    log_str = '\nTraining: {}, Task: {}, Accuracy = {:.2f}%'.format(args.trte, args.name, acc)

    args.out_file.write(log_str)
    args.out_file.flush()
    print(log_str)

def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CL')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--s', type=int, default=0, help="source")
    parser.add_argument('--t', type=int, default=1, help="target")
    parser.add_argument('--max_epoch', type=int, default=10, help="max iterations")
    parser.add_argument('--batch_size', type=int, default=64, help="batch_size")
    parser.add_argument('--worker', type=int, default=4, help="number of workers")
    parser.add_argument('--dset', type=str, default='office-home', choices=['VISDA-C', 'office', 'office-home', 'office-caltech'])
    parser.add_argument('--lr', type=float, default=1e-2, help="learning rate")
    parser.add_argument('--pf_c', type=float, default=1e-2, help="pruning fraction conv")
    parser.add_argument('--pf_bn', type=float, default=0, help="pruning fraction BN")
    parser.add_argument('--net', type=str, default='resnet50', help="alexnet, vgg16, resnet50, res101")
    parser.add_argument('--seed', type=int, default=2020, help="random seed")

    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--epsilon', type=float, default=1e-5)
    parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument('--smooth', type=float, default=0.1)
    parser.add_argument('--output', type=str, default='san')
    parser.add_argument('--output_src', type=str, default='san')
    parser.add_argument('--da', type=str, default='uda', choices=['uda', 'pda'])
    parser.add_argument('--trte', type=str, default='val', choices=['full', 'val'])
    args = parser.parse_args()

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
        
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    # torch.backends.cudnn.deterministic = True

    folder = '/data3/prasanna/Datasets/Office_home/'
    args.s_dset_path = folder + args.dset + '/' + names[args.s] + '_list.txt'
    args.test_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'

    # dset_loaders_source = data_load(args)
    # torch.save(dset_loaders_source, os.path.join(args.output, 'dset_loaders_source.pt'))
    # sys.exit()
    dset_loaders = torch.load(os.path.join(args.output, 'dset_loaders_source.pt'))

    args.output_dir_src = osp.join(args.output_src, names[args.s][0].upper())
    args.name_src = names[args.s][0].upper()
    args.out_file = open(osp.join(args.output_dir_src, 'log_pruned.txt'), 'w')
    args.out_file.write(print_args(args) + '\n')
    args.out_file.flush()

    netF, netB, netC, mask_list = prune_source(args)
    finetune_source(args, netF, netB, netC, mask_list, dset_loaders)

    args.out_file = open(osp.join(args.output_dir_src, 'log_test_pruned.txt'), 'w')
    for i in range(len(names)):
        args.t = i
        args.name = names[args.s][0].upper() + names[args.t][0].upper()

        args.s_dset_path = folder + args.dset + '/' + names[args.s] + '_list.txt'
        args.test_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'

        if i != args.s:
            dset_loaders = data_load(args)

        test_target(args, dset_loaders, i)
