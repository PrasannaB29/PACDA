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
from scipy.spatial.distance import cdist
import torch.nn.utils.prune as prune
import pickle

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
    txt_tar = open(args.t_dset_path).readlines()
    txt_test = open(args.test_dset_path).readlines()

    dsets["target"] = ImageList_idx(txt_tar, transform=image_train())
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, shuffle=True, num_workers=args.worker, drop_last=False)
    dsets["test"] = ImageList_idx(txt_test, transform=image_test())
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs*3, shuffle=False, num_workers=args.worker, drop_last=False)

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
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    mean_ent = torch.mean(loss.Entropy(nn.Softmax(dim=1)(all_output))).cpu().data.item()

    return accuracy*100, mean_ent

def load_mask(args, dir, domain_num, layer='F', param_type='w'):
    mask_name = str(domain_num)+'_mask_'+layer+'_'+param_type+'.pkl'
    mask_file = os.path.join(dir, mask_name)
    with open(mask_file, 'rb') as f:
        mask_dict_numpy = pickle.load(f)
    mask_dict = {}
    for key in mask_dict_numpy:
        mask_dict[key] = 1 - torch.from_numpy(mask_dict_numpy[key]).cuda()
    return mask_dict

def train_target(args, num):

    """
    BN running stats reset before starting target train
    """
    # Load source mask starts
    mask_dict_F_w = load_mask(args, args.output_dir_src, args.s, 'F', 'w')
    mask_dict_F_b = load_mask(args, args.output_dir_src, args.s, 'F', 'b')
    mask_dict_B_w = load_mask(args, args.output_dir_src, args.s, 'B', 'w')
    mask_dict_B_b = load_mask(args, args.output_dir_src, args.s, 'B', 'b')
    # Load source mask ends

    dset_loaders = data_load(args)

    ## set base network
    netF = network.ResBase(res_name=args.net).cuda()
    netB = network.feat_bootleneck(type=args.classifier, feature_dim=netF.in_features, bottleneck_dim=args.bottleneck).cuda()
    netC = network.feat_classifier(type=args.layer, class_num = args.class_num, bottleneck_dim=args.bottleneck).cuda()

    modelpath = args.output_dir_src + '/' + str(num) + '_F_pruned.pt'
    netF.load_state_dict(torch.load(modelpath))
    modelpath = args.output_dir_src + '/' + str(num) + '_B_pruned.pt'
    netB.load_state_dict(torch.load(modelpath))
    modelpath = args.output_dir_src + '/' + str(num) + '_C_pruned.pt'
    netC.load_state_dict(torch.load(modelpath))
    netC.eval()

    # Set BN running stats to ImageNet Resnet50 init - start
    netF_dummy = network.ResBase(res_name=args.net).cuda()
    netB_dummy = network.feat_bootleneck(type=args.classifier, feature_dim=netF.in_features,
                                   bottleneck_dim=args.bottleneck).cuda()
    for (name1, module1),(name2, module2) in zip(netF.named_modules(), netF_dummy.named_modules()):
        if isinstance(module1, torch.nn.BatchNorm2d):
            module1.load_state_dict(module2.state_dict())

    for (name1, module1),(name2, module2) in zip(netB.named_modules(), netB_dummy.named_modules()):
        if isinstance(module1, torch.nn.BatchNorm1d):
            module1.load_state_dict(module2.state_dict())
    del netF_dummy
    del netB_dummy
    # Set BN running stats to ImageNet Resnet50 init - end

    # Reset BN running stats - start
    # for name, module in netF.named_modules():
    #     if isinstance(module, torch.nn.BatchNorm2d):
    #         module.reset_running_stats()
    # for name, module in netB.named_modules():
    #     if isinstance(module, torch.nn.BatchNorm1d):
    #         module.reset_running_stats()
    # Reset BN running stats - end

    # Optimizer starts
    for k, v in netC.named_parameters():
        v.requires_grad = False
    param_group = []
    for k, v in netF.named_parameters():
        if args.lr_decay1 > 0:
            param_group += [{'params': v, 'lr': args.lr * args.lr_decay1}]
        else:
            v.requires_grad = False
    for k, v in netB.named_parameters():
        if args.lr_decay2 > 0:
            param_group += [{'params': v, 'lr': args.lr * args.lr_decay2}]
        else:
            v.requires_grad = False

    """
    # Setting BN modules to eval, to freeze running stats
    for name, module in netF.named_modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.eval()
            module.track_running_stats=False
    for name, module in netB.named_modules():
        if isinstance(module, torch.nn.BatchNorm1d):
            module.eval()
            module.track_running_stats = False
    """

    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)
    # Optimizer ends

    # Training starts
    max_iter = args.max_epoch * len(dset_loaders["target"])
    interval_iter = max_iter // args.interval
    iter_num = 0

    acc_list = list()

    while iter_num < max_iter:
        try:
            inputs_test, _, tar_idx = iter_test.next()
        except:
            iter_test = iter(dset_loaders["target"])
            inputs_test, _, tar_idx = iter_test.next()

        if inputs_test.size(0) == 1:
            continue

        if iter_num % interval_iter == 0 and args.cls_par > 0:
            netF.eval()
            netB.eval()
            mem_label = obtain_label(dset_loaders['test'], netF, netB, netC, args)
            mem_label = torch.from_numpy(mem_label).cuda()

            """
            # Setting all modules except BN to train, to freeze running stats
            for name, module in netF.named_modules():
                if not isinstance(module, torch.nn.BatchNorm2d):
                    module.train()
            for name, module in netB.named_modules():
                if not isinstance(module, torch.nn.BatchNorm1d):
                    module.train()
            """
            netF.train()
            netB.train()

        inputs_test = inputs_test.cuda()

        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)

        features_test = netB(netF(inputs_test))
        outputs_test = netC(features_test)

        if args.cls_par > 0:
            pred = mem_label[tar_idx]
            classifier_loss = nn.CrossEntropyLoss()(outputs_test, pred)
            classifier_loss *= args.cls_par
            if iter_num < interval_iter and args.dset == "VISDA-C":
                classifier_loss *= 0
        else:
            classifier_loss = torch.tensor(0.0).cuda()

        if args.ent:
            softmax_out = nn.Softmax(dim=1)(outputs_test)
            entropy_loss = torch.mean(loss.Entropy(softmax_out))
            if args.gent:
                msoftmax = softmax_out.mean(dim=0)
                gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + args.epsilon))
                entropy_loss -= gentropy_loss
            im_loss = entropy_loss * args.ent_par
            classifier_loss += im_loss
        optimizer.zero_grad()
        classifier_loss.backward()

        with torch.no_grad():
            for k, v in netF.named_parameters():
                if (k.endswith("weight")) and ("bn" not in k):
                    tmp = k[:-7]
                    v.grad = torch.mul(v.grad, mask_dict_F_w[tmp])
                    
                elif (k.endswith("bias")) and ("bn" not in k):
                    tmp = k[:-5]
                    v.grad = torch.mul(v.grad, mask_dict_F_b[tmp])
                # else:
                #     print(k)

            for k, v in netB.named_parameters():
                if (k.endswith("weight")) and ("bn" not in k):
                    tmp = k[:-7]                    
                    v.grad = torch.mul(v.grad, mask_dict_B_w[tmp])

                elif (k.endswith("bias")) and ("bn" not in k):
                    tmp = k[:-5]                    
                    v.grad = torch.mul(v.grad, mask_dict_B_b[tmp])

                # else:
                #     print(k)
        ## Gradient Pruning method ends

        optimizer.step()

        netF.eval()
        netB.eval()
        # acc_s_te, _ = cal_acc(dset_loaders['test'], netF, netB, netC)
        # acc_list.append(acc_s_te)

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            acc_s_te, _ = cal_acc(dset_loaders['test'], netF, netB, netC)
            log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(args.name, iter_num, max_iter, acc_s_te)
            args.out_file.write(log_str + '\n')
            args.out_file.flush()
            print(log_str+'\n')

        """
        # Setting BN modules to eval, to freeze running stats
        for name, module in netF.named_modules():
            if not isinstance(module, torch.nn.BatchNorm2d):
                module.train()
        for name, module in netB.named_modules():
            if not isinstance(module, torch.nn.BatchNorm1d):
                module.train()
        """
        netF.train()
        netB.train()

        print("Iteration "+str(iter_num)+" done")

    # acc_list_array = np.asarray(acc_list)
    # np.save(osp.join(args.output_dir, "acc_list.npy"))    
    if args.issave:
        torch.save(netF.state_dict(), osp.join(args.output_dir, str(num+1) + "_F.pt"))#CL
        torch.save(netB.state_dict(), osp.join(args.output_dir, str(num+1) + "_B.pt"))#CL
        torch.save(netC.state_dict(), osp.join(args.output_dir, str(num+1) + "_C.pt"))#CL
        
    return netF, netB, netC

def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s

def obtain_label(loader, netF, netB, netC, args):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            feas = netB(netF(inputs))
            outputs = netC(feas)
            if start_test:
                all_fea = feas.float().cpu()
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    all_output = nn.Softmax(dim=1)(all_output)
    ent = torch.sum(-all_output * torch.log(all_output + args.epsilon), dim=1)
    unknown_weight = 1 - ent / np.log(args.class_num)
    _, predict = torch.max(all_output, 1)

    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    if args.distance == 'cosine':
        all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
        all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()

    all_fea = all_fea.float().cpu().numpy()
    K = all_output.size(1)
    aff = all_output.float().cpu().numpy()
    initc = aff.transpose().dot(all_fea)
    initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
    cls_count = np.eye(K)[predict].sum(axis=0)
    labelset = np.where(cls_count>args.threshold)
    labelset = labelset[0]
    # print(labelset)

    dd = cdist(all_fea, initc[labelset], args.distance)
    pred_label = dd.argmin(axis=1)
    pred_label = labelset[pred_label]

    for round in range(1):
        aff = np.eye(K)[pred_label]
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
        dd = cdist(all_fea, initc[labelset], args.distance)
        pred_label = dd.argmin(axis=1)
        pred_label = labelset[pred_label]

    acc = np.sum(pred_label == all_label.float().numpy()) / len(all_fea)
    log_str = 'Accuracy = {:.2f}% -> {:.2f}%'.format(accuracy * 100, acc * 100)

    ## Prune testing starts
    # Uncomment below lines once prune testing is done
    # args.out_file.write(log_str + '\n')
    # args.out_file.flush()
    # print(log_str+'\n')
    ## Prune testing ends

    return pred_label.astype('int')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SHOT')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--s', type=int, default=0, help="source")
    parser.add_argument('--t', type=int, default=1, help="target")
    parser.add_argument('--max_epoch', type=int, default=15, help="max iterations")
    parser.add_argument('--interval', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=64, help="batch_size")
    parser.add_argument('--worker', type=int, default=4, help="number of workers")
    parser.add_argument('--dset', type=str, default='office-home', choices=['VISDA-C', 'office', 'office-home', 'office-caltech'])
    parser.add_argument('--lr', type=float, default=1e-2, help="learning rate")
    parser.add_argument('--net', type=str, default='resnet50', help="alexnet, vgg16, resnet50, res101")
    parser.add_argument('--seed', type=int, default=2020, help="random seed")
    parser.add_argument('--pf_c', type=float, default=1e-2, help="pruning fraction conv")
    parser.add_argument('--pf_bn', type=float, default=0, help="pruning fraction BN")
 
    parser.add_argument('--gent', type=bool, default=True)
    parser.add_argument('--ent', type=bool, default=True)
    parser.add_argument('--threshold', type=int, default=0)
    parser.add_argument('--cls_par', type=float, default=0.3)
    parser.add_argument('--ent_par', type=float, default=1.0)
    parser.add_argument('--lr_decay1', type=float, default=0.1)
    parser.add_argument('--lr_decay2', type=float, default=1.0)

    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--epsilon', type=float, default=1e-5)
    parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument('--distance', type=str, default='cosine', choices=["euclidean", "cosine"])  
    parser.add_argument('--output', type=str, default='san')
    parser.add_argument('--output_src', type=str, default='san')
    parser.add_argument('--da', type=str, default='uda', choices=['uda', 'pda'])
    parser.add_argument('--issave', type=bool, default=True)
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

    #TODO: start - Carefully set below params for each run
    num = 0 # num = 0 is source, num = i is ith target
    args.s = 0
    args.t = 1

    # 1st choice of output_dir_src used for only 1st target domain. For other cases use second line
    args.output_dir_src = osp.join(args.output_src, names[0][0].upper())
    # args.output_dir_src = osp.join(args.output_src, str(num)+names[0][0].upper())
    #TODO: end - Carefully set below params for each run

    folder = '/data3/prasanna/Datasets/Office_home/'
    # args.s_dset_path = folder + args.dset + '/' + names[args.s] + '_list.txt'
    args.t_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'
    args.test_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'

    args.output_dir = osp.join(args.output, str(num+1)+names[0][0].upper())#CL
    args.name = str(num+1)+names[0][0].upper()#CL

    if not osp.exists(args.output_dir):
        os.mkdir(args.output_dir)

    args.savename = 'par_' + str(args.cls_par)

    args.out_file = open(osp.join(args.output_dir, 'log_' + args.savename + '.txt'), 'w')
    args.out_file.write(print_args(args)+'\n')
    args.out_file.flush()
    netF, netB, netC = train_target(args, num)

    # dset_loaders_source = torch.load(os.path.join(args.output, 'dset_loaders_source.pt'))
    # acc_s_te, _ = cal_acc(dset_loaders_source['source_te'], netF, netB, netC)
    # print("Accuracy on source model is {}".format(acc_s_te))


    


