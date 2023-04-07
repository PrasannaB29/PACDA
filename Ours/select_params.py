import argparse
import os, sys
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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
# from scipy.spatial.distance import cdist
# from sklearn.metrics import confusion_matrix
# from sklearn.cluster import KMeans
import math, pickle
from torch.nn import Softmax
from torchvision.utils import save_image

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
    
# method 1 - Univariate Gaussian using whole batch data
def method_1(sample_bn_source, sample_bn_target, netF_s, netF, args):

    source_mean = netF_s.bn1.state_dict()['running_mean']
    source_var = netF_s.bn1.state_dict()['running_var']
    source_std = torch.sqrt(source_var)
    target_mean = netF.bn1.state_dict()['running_mean']
    target_var = netF.bn1.state_dict()['running_var']   
    target_std = torch.sqrt(target_var)

    sample_source_mean = torch.mean(sample_bn_source,dim=[0,2,3])
    sample_target_mean = torch.mean(sample_bn_target,dim=[0,2,3])

    source_score=0
    for sample, mean, std in zip(sample_source_mean, source_mean,source_std):
        diff_s = sample-mean
        num_s = torch.exp(-0.5*((diff_s/std)**2))
        denom_s = 1/(std*torch.sqrt(2*torch.tensor(math.pi)))
        source_score+= num_s/denom_s

    target_score=0
    for sample, mean, std in zip(sample_target_mean, target_mean,target_std):
        diff_t = sample-mean
        num_t = torch.exp(-0.5*((diff_t/std)**2))
        denom_t = 1/(std*torch.sqrt(2*torch.tensor(math.pi)))
        target_score+= num_t/denom_t
    
    return source_score, target_score

# method 2 - using whole batch but distance between mean and variance
def method_2(sample_bn_source, sample_bn_target, netF_s, netF, args):
    source_mean = netF_s.bn1.state_dict()['running_mean']
    source_var = netF_s.bn1.state_dict()['running_var']
    target_mean = netF.bn1.state_dict()['running_mean']
    target_var = netF.bn1.state_dict()['running_var']   

    sample_source_mean = torch.mean(sample_bn_source,dim=[0,2,3])
    sample_target_mean = torch.mean(sample_bn_target,dim=[0,2,3])
    sample_source_var = torch.var(sample_bn_source,dim=[0,2,3])
    sample_target_var = torch.var(sample_bn_target,dim=[0,2,3])

    source_score_mean=0
    source_score_var=0
    for sample_mean, mean, sample_var, var in zip(sample_source_mean, source_mean, sample_source_var, source_var):
        diff_mean = (sample_mean - mean)**2
        diff_var = (sample_var - var)**2
        source_score_mean += diff_mean
        source_score_var += diff_var
    source_score = torch.sqrt(source_score_mean) + torch.sqrt(source_score_var)

    target_score_mean=0
    target_score_var=0
    for sample_mean, mean, sample_var, var in zip(sample_target_mean, target_mean, sample_target_var, target_var):
        diff_mean = (sample_mean - mean)**2
        diff_var = (sample_var - var)**2
        target_score_mean += diff_mean
        target_score_var += diff_var
    target_score = torch.sqrt(target_score_mean) + torch.sqrt(target_score_var)

    return source_score, target_score    

# method 2 CL - For CL, using whole batch but distance between mean and variance
def method_2_cl(sample_bn, netF, args):
    means = []
    vars = []
    sample_means = []
    sample_vars = []
    for i in range(len(netF)):
        means.append(netF[i].bn1.state_dict()['running_mean'])
        vars.append(netF[i].bn1.state_dict()['running_var'])
        sample_means.append(torch.mean(sample_bn[i],dim=[0,2,3]))
        sample_vars.append(torch.var(sample_bn[i],dim=[0,2,3]))

    scores = []
    for i in range(len(netF)):
        score_mean = 0
        score_var = 0
        for sample_mean, mean, sample_var, var in zip(sample_means[i], means[i], sample_vars[i], vars[i]):
            score_mean += (sample_mean - mean)**2
            score_var += (sample_var - var)**2
        scores.append(torch.sqrt(score_mean) + torch.sqrt(score_var))
    scores = torch.as_tensor(scores)
    # print("Num of scores is {}".format(scores.shape))

    return torch.argmin(scores)    

# method 3 - Using difference for each sample in batch
def method_3(sample_bn_source, sample_bn_target, netF_s, netF, args):
    source_mean = netF_s.bn1.state_dict()['running_mean']
    source_var = netF_s.bn1.state_dict()['running_var']
    target_mean = netF.bn1.state_dict()['running_mean']
    target_var = netF.bn1.state_dict()['running_var']

    source_batch_scores = list()
    for j,fmaps in enumerate(sample_bn_source):
        sample_score_mean=0
        sample_score_var=0
        for i,fmap in enumerate(fmaps):
            sample_mean = torch.mean(fmap)
            sample_var = torch.var(fmap)
            diff_mean = (sample_mean-source_mean[i])**2
            diff_var = (sample_var-source_var[i])**2
            sample_score_mean += diff_mean
            sample_score_var += diff_var
        sample_score = torch.sqrt(sample_score_mean) + torch.sqrt(sample_score_var)
        source_batch_scores.append(sample_score)
    source_batch_scores=torch.stack(source_batch_scores)

    target_batch_scores = list()
    for j,fmaps in enumerate(sample_bn_target):
        sample_score_mean=0
        sample_score_var=0
        for i,fmap in enumerate(fmaps):
            sample_mean = torch.mean(fmap)
            sample_var = torch.var(fmap)
            diff_mean = (sample_mean-target_mean[i])**2
            diff_var = (sample_var-target_var[i])**2
            sample_score_mean += diff_mean
            sample_score_var += diff_var
        sample_score = torch.sqrt(sample_score_mean) + torch.sqrt(sample_score_var)
        target_batch_scores.append(sample_score)
    target_batch_scores=torch.stack(target_batch_scores)

    return source_batch_scores, target_batch_scores

# method 4 - Using Gaussian probability for each value in featuremaps for each sample in batch
def method_4(sample_bn_source, sample_bn_target, netF_s, netF, args):
    source_mean = netF_s.bn1.state_dict()['running_mean']
    source_var = netF_s.bn1.state_dict()['running_var']
    source_std = torch.sqrt(source_var)
    target_mean = netF.bn1.state_dict()['running_mean']
    target_var = netF.bn1.state_dict()['running_var']
    target_std = torch.sqrt(target_var)
    epsilon = 1e-5

    source_batch_scores = list()
    for j,fmaps in enumerate(sample_bn_source):
        source_fmap_score=list()
        for i,fmap in enumerate(fmaps):
            flat_fmap = torch.flatten(fmap)
            mean_s = source_mean[i]
            std_s = source_std[i]
            source_score=0
            diff_map_s = flat_fmap - mean_s
            num_s = torch.exp(-0.5*((diff_map_s/std_s)**2))
            denom_s = 1/(std_s*torch.sqrt(2*torch.tensor(math.pi))+epsilon)
            source_score = torch.sum(num_s/denom_s)
            
            # for val in flat_fmap:
            #     diff_s = val-mean_s
            #     num_s = torch.exp(-0.5*((diff_s/std_s)**2))
            #     denom_s = 1/(std_s*torch.sqrt(2*torch.tensor(math.pi))+epsilon)
            #     source_score += num_s/denom_s

            source_score = source_score/(flat_fmap.shape[0])

            source_fmap_score.append(source_score)
        source_fmap_score = torch.stack(source_fmap_score)
        source_batch_scores.append(source_fmap_score)
    source_batch_scores=torch.stack(source_batch_scores)
    # print("source_batch_scores.shape")
    # print(source_batch_scores.shape)
    
    target_batch_scores = list()
    for j,fmaps in enumerate(sample_bn_target):
        target_fmap_score=list()
        for i,fmap in enumerate(fmaps):
            flat_fmap = torch.flatten(fmap)
            mean_t = target_mean[i]
            std_t = source_std[i]
            target_score=0
            diff_map_t = flat_fmap - mean_t
            num_t = torch.exp(-0.5*((diff_map_t/std_t)**2))
            denom_t = 1/(std_t*torch.sqrt(2*torch.tensor(math.pi))+epsilon)
            target_score = torch.sum(num_t/denom_t)

            target_score = target_score/(flat_fmap.shape[0])

            target_fmap_score.append(target_score)
        target_fmap_score = torch.stack(target_fmap_score)
        target_batch_scores.append(target_fmap_score)
    target_batch_scores=torch.stack(target_batch_scores)
    # print("target_batch_scores.shape")
    # print(target_batch_scores.shape)

    s_cnt=0
    t_cnt=0
    s_greater_cnt_list=list()
    t_greater_cnt_list=list()
    # fmap mapped using sum of scores method
    for s,t in zip(source_batch_scores, target_batch_scores):
        s_score = torch.sum(s)
        t_score = torch.sum(t)
        if s_score>t_score:
            s_cnt+=1
        else:
            t_cnt+=1
        s_greater_cnt_list.append(s_score)
        t_greater_cnt_list.append(t_score)
    # fmap mapped using voting method
    # for s,t in zip(source_batch_scores, target_batch_scores):
    #     s_greater_count = torch.count_nonzero((s>t)*1)
    #     t_greater_count = s.shape[0]-s_greater_count
    #     s_greater_cnt_list.append(s_greater_count)
    #     t_greater_cnt_list.append(t_greater_count)
    #     if s_greater_count>t_greater_count:
    #         s_cnt += 1
    #     else:
    #         t_cnt += 1
    s_greater_cnt_list = torch.stack(s_greater_cnt_list)
    t_greater_cnt_list = torch.stack(t_greater_cnt_list)
    for s,t in zip(s_greater_cnt_list,t_greater_cnt_list):
        # print("Source greater for {} and Target greater for {}".format(s,t))
        print("Source score is {} and Target score is {}".format(s,t))
    print("Num of samples classified as source is {}".format(s_cnt))
    print("Num of samples classified as target is {}".format(t_cnt))
    
    return

def generate_transforms(img):

    transformed_imgs = list()
    transformed_imgs.append(img)
    transformed_imgs.append(transforms.ColorJitter(brightness=0.3, hue=0.3)(img))
    transformed_imgs.append(transforms.RandomHorizontalFlip(p=1)(img))
    transformed_imgs.append(transforms.RandomVerticalFlip(p=1)(img))
    transformed_imgs.append(transforms.RandomRotation(degrees=[30,60])(img))
    transformed_imgs.append(transforms.RandomRotation(degrees=[300,330])(img))
    transformed_imgs.append(transforms.GaussianBlur(kernel_size=5)(img))
    transformed_imgs.append(transforms.RandomResizedCrop(size=224)(img))
    # rot_img_3 = transforms.RandomRotation(degrees=[120,150])(img)
    # rot_img_4 = transforms.RandomRotation(degrees=[210,240])(img)
    transformed_imgs = torch.stack(transformed_imgs)
    return transformed_imgs


# method 5 - Using augmentations of single sample
def method_5(input_i, sample_bn_source, sample_bn_target, netF_s, netF, args):

    source_mean = netF_s.bn1.state_dict()['running_mean']
    source_var = netF_s.bn1.state_dict()['running_var']
    target_mean = netF.bn1.state_dict()['running_mean']
    target_var = netF.bn1.state_dict()['running_var']

    source_batch_scores = list()
    target_batch_scores = list()

    # Obtain augmentations - starts
    for img in input_i:
        # save_image(img, 'normalized_test_image.png')
        # print("normalized Image saved")
        transformed_imgs = generate_transforms(img)
        _, sample_bn_source = netF_s(transformed_imgs)
        _, sample_bn_target = netF(transformed_imgs)
        
        sample_source_mean = torch.mean(sample_bn_source,dim=[0,2,3])
        sample_target_mean = torch.mean(sample_bn_target,dim=[0,2,3])
        sample_source_var = torch.var(sample_bn_source,dim=[0,2,3])
        sample_target_var = torch.var(sample_bn_target,dim=[0,2,3])

        source_score_mean=0
        source_score_var=0
        for sample_mean, mean, sample_var, var in zip(sample_source_mean, source_mean, sample_source_var, source_var):
            diff_mean = (sample_mean - mean)**2
            diff_var = (sample_var - var)**2
            source_score_mean += diff_mean
            source_score_var += diff_var
        source_score = torch.sqrt(source_score_mean) + torch.sqrt(source_score_var)
        source_batch_scores.append(source_score)

        target_score_mean=0
        target_score_var=0
        for sample_mean, mean, sample_var, var in zip(sample_target_mean, target_mean, sample_target_var, target_var):
            diff_mean = (sample_mean - mean)**2
            diff_var = (sample_var - var)**2
            target_score_mean += diff_mean
            target_score_var += diff_var
        target_score = torch.sqrt(target_score_mean) + torch.sqrt(target_score_var)
        target_batch_scores.append(target_score)
        
    source_batch_scores=torch.stack(source_batch_scores)
    target_batch_scores=torch.stack(target_batch_scores)

    s_cnt=0
    t_cnt=0
    for s,t in zip(source_batch_scores, target_batch_scores):
        if s<t:
            s_cnt+=1
        else:
            t_cnt+=1

    for s,t in zip(source_batch_scores,target_batch_scores):
        print("Source score is {} and Target score is {}".format(s,t))
    print("Num of samples classified as source is {}".format(s_cnt))
    print("Num of samples classified as target is {}".format(t_cnt))
    
    return source_batch_scores, target_batch_scores

def calc_bns_loss_batch(args, netF_s, netB_s, netF, netB, netC, names, folder):

    args.test_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'
    dset_loaders = data_load(args)
    
    max_iter = len(dset_loaders["test"])
    print("Total iterations = {}".format(max_iter))
    # interval_iter = max_iter // 10
    iter_num = 0

    batch_scores = list()
    start_test = True
    correct_sum=0
    tot_sum=0
    with torch.no_grad():
        while iter_num < max_iter:
            # print("{}th itrn started".format(iter_num))
            if (iter_num%10==0) or (iter_num==max_iter-1):
                print("{}th itrn started".format(iter_num))
            try:
                inputs, labels = iter_var.next()
            except:
                iter_var = iter(dset_loaders["test"])
                inputs, labels = iter_var.next()

            if inputs.size(0) == 1:
                continue
            inputs, labels = inputs.cuda(), labels.cuda()

            s_outputs, sample_bn_source = netF_s(inputs)
            outputs, sample_bn_target = netF(inputs)
            source_batch_score, target_batch_score = method_2(sample_bn_source, sample_bn_target, netF_s, netF, args)

            del sample_bn_source
            del sample_bn_target        

            # print("source error = {} and target error = {}".format(source_batch_score, target_batch_score))
            if source_batch_score>target_batch_score:
                # print("Batch {} belongs to target".format(iter_num))
                batch_scores.append(1)
                outputs = netC(netB(outputs))

            else:
                # print("Batch {} belongs to source".format(iter_num))
                batch_scores.append(0)
                outputs = netC(netB_s(s_outputs))
                # del s_outputs

            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

            # all_output = outputs.float().cpu()
            # all_label = labels.float().cpu()
            iter_num += 1
            del inputs
            del labels
            del outputs
            del s_outputs

            # _, predict = torch.max(all_output, 1)
            # correct_sum += torch.sum(torch.squeeze(predict).float() == all_label).item()
            # tot_sum += float(all_label.size()[0])

            # del all_output
            # del all_label
    all_label = all_label.cpu()
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    # accuracy = correct_sum / tot_sum
    return accuracy*100, batch_scores

def calc_bns_loss_batch_cl(args, netF, netB, netC, names, folder, loader):

    max_iter = len(loader)
    print("Total iterations = {}".format(max_iter))
    # interval_iter = max_iter // 10
    iter_num = 0

    batch_scores = list()
    start_test = True
    with torch.no_grad():
        while iter_num < max_iter:
            # print("{}th itrn started".format(iter_num))
            if (iter_num%10==0) or (iter_num==max_iter-1):
                print("{}th itrn started".format(iter_num))
            try:
                inputs, labels = iter_var.next()
            except:
                iter_var = iter(loader)
                inputs, labels = iter_var.next()

            if inputs.size(0) == 1:
                continue
            inputs, labels = inputs.cuda(), labels.cuda()

            outputs = []
            sample_bn = []
            for i in range(len(netF)):
                tmp_outputs, tmp_sample_bn = netF[i](inputs)
                outputs.append(tmp_outputs)
                sample_bn.append(tmp_sample_bn)
            chosen_domain = method_2_cl(sample_bn, netF, args)

            del sample_bn        

            batch_scores.append(chosen_domain)
            outputs = netC(netB[chosen_domain](outputs[chosen_domain]))
            # print("source error = {} and target error = {}".format(source_batch_score, target_batch_score))

            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

            iter_num += 1
            del inputs
            del labels
            del outputs
            del tmp_outputs
            del tmp_sample_bn

    all_label = all_label.cpu()
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    return accuracy*100, batch_scores

def calc_bns_loss_single(args, netF_s, netF, names, folder):
    # Inference on source data
    folder = '/data3/prasanna/Datasets/Office_home/'

    args.t = 0
    args.test_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'
    tmp_loader = data_load(args)
    iter_tmp = iter(tmp_loader["test"])
    tmp_data = iter_tmp.next()
    
    input_i = tmp_data[0].cuda()
    _, sample_bn_source = netF_s(input_i)
    _, sample_bn_target = netF(input_i)

    # Modify below line to use different single sample methods
    print("Inference on source data")
    if args.method=="single-gaussian":
        method_4(sample_bn_source, sample_bn_target, netF_s, netF, args)
    elif args.method=="single-squareddist":
        source_batch_scores, target_batch_scores = method_3(sample_bn_source, sample_bn_target, netF_s, netF, args)
    elif args.method=="single-augmentation":
        source_batch_scores, target_batch_scores = method_5(input_i, sample_bn_source, sample_bn_target, netF_s, netF, args)
    # cnt=0
    
    # for s,t in zip(source_batch_scores, target_batch_scores):
    #     print("Source {} | Target {}".format(s, t))
    #     if s<t:
    #         cnt+=1
    # print("Correct outputs for source data are {}".format(cnt))

    # Inference on target data
    args.t = 1 
    args.test_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'
    tmp_loader = data_load(args)
    iter_tmp = iter(tmp_loader["test"])
    tmp_data = iter_tmp.next()
    
    input_i = tmp_data[0].cuda()
    _, sample_bn_source = netF_s(input_i)
    _, sample_bn_target = netF(input_i)

    print("Inference on target data")

    if args.method=="single-gaussian":
        method_4(sample_bn_source, sample_bn_target, netF_s, netF, args)
    elif args.method=="single-squareddist":
        source_batch_scores, target_batch_scores = method_3(sample_bn_source, sample_bn_target, netF_s, netF, args)
    elif args.method=="single-augmentation":
        source_batch_scores, target_batch_scores = method_5(input_i, sample_bn_source, sample_bn_target, netF_s, netF, args)
    # cnt=0
    
    # for s,t in zip(source_batch_scores, target_batch_scores):
    #     print("Source {} | Target {}".format(s, t))
    #     if t<s:
    #         cnt+=1
    # print("Correct outputs for target data are {}".format(cnt))
    return

def entropy_method(args, netF_s, netF, netB, netB_s, netC, names, folder):
    # Inference on source data
    folder = '/data3/prasanna/Datasets/Office_home/'
    epsilon = 1e-5

    args.t = 0
    args.test_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'
    tmp_loader = data_load(args)
    iter_tmp = iter(tmp_loader["test"])
    tmp_data = iter_tmp.next()
    
    input_i = tmp_data[0].cuda()
    features_s,_ = netF_s(input_i)
    features_t,_ = netF(input_i)
    source_outputs = netC(netB_s(features_s))
    target_outputs = netC(netB(features_t))

    # Source Entropy calculation
    source_logits = nn.Softmax(dim=1)(source_outputs)
    source_entropy = -source_logits * torch.log(source_logits + epsilon)
    source_entropy = torch.sum(source_entropy, dim=1)  

    # Target Entropy calculation
    target_logits = nn.Softmax(dim=1)(target_outputs)
    target_entropy = -target_logits * torch.log(target_logits + epsilon)
    target_entropy = torch.sum(target_entropy, dim=1)

    cnt=0
    print("Inference on source data")
    for s,t in zip(source_entropy, target_entropy):
        print("Source {} | Target {}".format(s, t))
        if s<t:
            cnt+=1
    print("Correct outputs for source data are {}".format(cnt))


    # Inference on target data
    args.t = 1
    args.test_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'
    tmp_loader = data_load(args)
    iter_tmp = iter(tmp_loader["test"])
    tmp_data = iter_tmp.next()
    
    input_i = tmp_data[0].cuda()
    features_s,_ = netF_s(input_i)
    features_t,_ = netF(input_i)
    source_outputs = netC(netB_s(features_s))
    target_outputs = netC(netB(features_t))

    # Source Entropy calculation
    source_logits = nn.Softmax(dim=1)(source_outputs)
    source_entropy = -source_logits * torch.log(source_logits + epsilon)
    source_entropy = torch.sum(source_entropy, dim=1)  

    # Target Entropy calculation
    target_logits = nn.Softmax(dim=1)(target_outputs)
    target_entropy = -target_logits * torch.log(target_logits + epsilon)
    target_entropy = torch.sum(target_entropy, dim=1)

    cnt=0
    print("Inference on target data")
    for s,t in zip(source_entropy, target_entropy):
        print("Source {} | Target {}".format(s, t))
        if t<s:
            cnt+=1
    print("Correct outputs for target data are {}".format(cnt))
    return
