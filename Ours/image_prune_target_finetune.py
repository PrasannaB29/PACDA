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
# from tqdm import tqdm
from scipy.spatial.distance import cdist
# from sklearn.metrics import confusion_matrix
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

    if not args.da == 'uda':
        label_map_s = {}
        for i in range(len(args.src_classes)):
            label_map_s[args.src_classes[i]] = i

        new_tar = []
        for i in range(len(txt_tar)):
            rec = txt_tar[i]
            reci = rec.strip().split(' ')
            if int(reci[1]) in args.tar_classes:
                if int(reci[1]) in args.src_classes:
                    line = reci[0] + ' ' + str(label_map_s[int(reci[1])]) + '\n'   
                    new_tar.append(line)
                else:
                    line = reci[0] + ' ' + str(len(label_map_s)) + '\n'   
                    new_tar.append(line)
        txt_tar = new_tar.copy()
        txt_test = txt_tar.copy()

    dsets["target"] = ImageList_idx(txt_tar, transform=image_train())
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, shuffle=True, num_workers=args.worker, drop_last=False)
    dsets["test"] = ImageList_idx(txt_test, transform=image_test())
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs*3, shuffle=False, num_workers=args.worker, drop_last=False)

    return dset_loaders

def cal_acc(loader, netF, netB, netC, flag=False):
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
    mask_file = osp.join(dir, mask_name)
    with open(mask_file, 'rb') as f:
        mask_dict_numpy = pickle.load(f)
    mask_dict = {}
    tmp_mask_dict = {}
    for key in mask_dict_numpy:
        tmp_mask_dict[key] = torch.from_numpy(1e9 * mask_dict_numpy[key]).cuda()
        mask_dict[key] = 1-torch.from_numpy(mask_dict_numpy[key]).cuda()
    return mask_dict, tmp_mask_dict

def save_mask(args, mask_dict, dir, domain_num, layer='F', param_type='w'):
    mask_dict_cpy = copy.deepcopy(mask_dict)
    for key in mask_dict:
        mask_dict_cpy[key] = mask_dict[key].cpu().numpy()
    mask_name = str(domain_num)+'_mask_'+layer+'_'+param_type+'.pkl'
    file = osp.join(dir, mask_name)
    with open(file, 'wb') as f:
        pickle.dump(mask_dict_cpy,f)

def prune_finetune_target(args, num):
    # Load source mask starts
    mask_prev_F_w, tmp_mask_F_w = load_mask(args, args.output_dir_src, num, 'F', 'w')
    mask_prev_F_b, tmp_mask_F_b = load_mask(args, args.output_dir_src, num, 'F', 'b')
    mask_prev_B_w, tmp_mask_B_w = load_mask(args, args.output_dir_src, num, 'B', 'w')
    mask_prev_B_b, tmp_mask_B_b = load_mask(args, args.output_dir_src, num, 'B', 'b')
    # Load mask ends

    dset_loaders = data_load(args)
    ## set base network
    netF = network.ResBase(res_name=args.net).cuda()
    netB = network.feat_bootleneck(type=args.classifier, feature_dim=netF.in_features, bottleneck_dim=args.bottleneck).cuda()
    netC = network.feat_classifier(type=args.layer, class_num = args.class_num, bottleneck_dim=args.bottleneck).cuda()

    modelpath = args.output_dir + '/' + str(num+1) + '_F.pt'
    netF.load_state_dict(torch.load(modelpath))
    modelpath = args.output_dir + '/' + str(num+1) + '_B.pt'
    netB.load_state_dict(torch.load(modelpath))
    modelpath = args.output_dir + '/' + str(num+1) + '_C.pt'
    netC.load_state_dict(torch.load(modelpath))
    netC.eval()

    # Temporary model used for target pruning

    temp_f = network.ResBase(res_name=args.net).cuda()
    temp_b = network.feat_bootleneck(type=args.classifier, feature_dim=netF.in_features, bottleneck_dim=args.bottleneck).cuda()

    modelpath = args.output_dir + '/' + str(num+1) + '_F.pt'
    temp_f.load_state_dict(torch.load(modelpath))
    modelpath = args.output_dir + '/' + str(num+1) + '_B.pt'
    temp_b.load_state_dict(torch.load(modelpath))
    temp_f.eval()
    temp_b.eval()

    # Setting source params in temp_f and temp_b to 1e9
    with torch.no_grad():
        for k, v in temp_f.named_parameters():
            if (k.endswith("weight")) and ("bn" not in k):
                tmp = k[:-7]  # because ".weight" has 7 characters
                v.data = torch.mul(v.data, mask_prev_F_w[tmp]) + tmp_mask_F_w[tmp]

            elif (k.endswith("bias")) and ("bn" not in k):
                tmp = k[:-5]
                v.data = torch.mul(v.data, mask_prev_F_b[tmp]) + tmp_mask_F_b[tmp]

        for k, v in temp_b.named_parameters():
            if (k.endswith("weight")) and ("bn" not in k):
                tmp = k[:-7]  # because ".weight" has 7 characters
                v.data = torch.mul(v.data, mask_prev_B_w[tmp]) + tmp_mask_B_w[tmp]

            elif (k.endswith("bias")) and ("bn" not in k):
                tmp = k[:-5]
                v.data = torch.mul(v.data, mask_prev_B_b[tmp]) + tmp_mask_B_b[tmp]

    # Pruning of temp_f starts
    mask_target_F_w = {}
    mask_target_F_b = {}
    with torch.no_grad():
        for name, module in temp_f.named_modules():
            f = 0
            if isinstance(module, torch.nn.Conv2d):
                prune.l1_unstructured(module, name='weight', amount=args.pf_c)
                # prune.l1_unstructured(module, name='bias', amount=args.pf_bn) # Omitted Because Conv2d in resnet doesn't have bias
                f = 1
            # prune 40% of connections in all linear layers
            elif isinstance(module, torch.nn.BatchNorm2d):
                # print("2 " + name)
                prune.l1_unstructured(module, name='weight', amount=args.pf_bn)
                prune.l1_unstructured(module, name='bias', amount=args.pf_bn)
                f = 2
            if f == 1:
                mask_target_F_w[name] = dict(module.named_buffers())['weight_mask']
                prune.remove(module, 'weight')
            elif f == 2:
                mask_target_F_w[name] = dict(module.named_buffers())['weight_mask']
                mask_target_F_b[name] = dict(module.named_buffers())['bias_mask']
                prune.remove(module, 'weight')
                prune.remove(module, 'bias')
    # Pruning of temp_f ends

    # Pruning of temp_b starts
    mask_target_B_w = {}
    mask_target_B_b = {}
    with torch.no_grad():
        for name, module in temp_b.named_modules():
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
                mask_target_B_w[name] = dict(module.named_buffers())['weight_mask']
                mask_target_B_b[name] = dict(module.named_buffers())['bias_mask']
                prune.remove(module, 'weight')
                prune.remove(module, 'bias')
    # Pruning of temp_b ends

    # # Pruning sanity check starts
    # f_w_sum=0
    # for key in mask_target_F_w:
    #     f_w_sum += torch.sum(torch.mul((1-mask_target_F_w[key]),(1-mask_prev_F_w[key])))
    # print(f_w_sum)
    # b_w_sum = 0
    # for key in mask_target_B_w:
    #     b_w_sum += torch.sum(torch.mul((1 - mask_target_B_w[key]), (1-mask_prev_B_w[key])))
    # print(b_w_sum)
    # f_b_sum = 0
    # for key in mask_target_F_b:
    #     f_b_sum += torch.sum(torch.mul((1 - mask_target_F_b[key]), (1-mask_prev_F_b[key])))
    # print(f_b_sum)
    # b_b_sum = 0
    # for key in mask_target_B_b:
    #     b_b_sum += torch.sum(torch.mul((1 - mask_target_B_b[key]), (1-mask_prev_B_b[key])))
    # print(b_b_sum)
    # return
    # # Pruning sanity check ends

    # Mask for current target finetuning
    mask_cur_F_w = {}
    mask_cur_F_b = {}
    mask_cur_B_w = {}
    mask_cur_B_b = {}
    for key in mask_prev_F_w:
        mask_cur_F_w[key] = torch.mul(mask_prev_F_w[key], mask_target_F_w[key])
    for key in mask_prev_F_b:
        mask_cur_F_b[key] = torch.mul(mask_prev_F_b[key], mask_target_F_b[key])
    for key in mask_prev_B_w:
        mask_cur_B_w[key] = torch.mul(mask_prev_B_w[key], mask_target_B_w[key])
    for key in mask_prev_B_b:
        mask_cur_B_b[key] = torch.mul(mask_prev_B_b[key], mask_target_B_b[key])

    # Save masks as pickle file - start
    # mask_target_F_w is the mask used during eval. mask_cur_F_w is only for target_finetune hence need not be saved
    save_mask(args, mask_target_F_w, args.output_dir, num+1, 'F', 'w')
    save_mask(args, mask_target_F_b, args.output_dir, num+1, 'F', 'b')
    save_mask(args, mask_target_B_w, args.output_dir, num+1, 'B', 'w')
    save_mask(args, mask_target_B_b, args.output_dir, num+1, 'B', 'b')
    # Save mask as pickle file - end

    # Deleting temporary model
    del temp_f
    del temp_b

    # Setting target model weights to pruned weights - starts
    with torch.no_grad():
        for k, v in netF.named_parameters():
            if (k.endswith("weight")) and ("bn" not in k):
                tmp = k[:-7]  # because ".weight" has 7 characters
                v.data = torch.mul(v.data, mask_target_F_w[tmp])

            elif (k.endswith("bias")) and ("bn" not in k):
                tmp = k[:-5]
                v.data = torch.mul(v.data, mask_target_F_b[tmp])

        for k, v in netB.named_parameters():
            if (k.endswith("weight")) and ("bn" not in k):
                tmp = k[:-7]  # because ".weight" has 7 characters
                v.data = torch.mul(v.data, mask_target_B_w[tmp])

            elif (k.endswith("bias")) and ("bn" not in k):
                tmp = k[:-5]
                v.data = torch.mul(v.data, mask_target_B_b[tmp])
    # Setting target model weights to pruned weights - ends

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

        # Gradient Pruning method starts
        # tmp_before_F_w = 0
        # tmp_before_F_b = 0
        # tmp_before_B_w = 0
        # tmp_before_B_b = 0
        with torch.no_grad():
            for k, v in netF.named_parameters():
                if (k.endswith("weight")) and ("bn" not in k):
                    # print("grad pruned layer = " + k)
                    # print("Grad before grad_pruning")
                    # print(v.grad)
                    tmp = k[:-7]
                    # print("Masked used is ")
                    # print(mask_dict_F_w[tmp])
                    v.grad = torch.mul(v.grad, mask_cur_F_w[tmp])
                    # with torch.no_grad():
                    #     tmp_before_F_w += torch.sum(torch.mul(v.data, 1-mask_dict_F_w[tmp]))


                    # print("Grad after grad_pruning")
                    # print(v.grad)
                elif (k.endswith("bias")) and ("bn" not in k):
                    # print(k)
                    tmp = k[:-5]
                    # with torch.no_grad():
                    #     tmp_before_F_b += torch.sum(torch.mul(v.data, 1-mask_dict_F_b[tmp]))
                    v.grad = torch.mul(v.grad, mask_cur_F_b[tmp])

            for k, v in netB.named_parameters():
                if (k.endswith("weight")) and ("bn" not in k):
                    tmp = k[:-7]
                    # with torch.no_grad():
                    #     tmp_before_B_w += torch.sum(torch.mul(v.data, 1-mask_dict_B_w[tmp]))
                    v.grad = torch.mul(v.grad, mask_cur_B_w[tmp])

                elif (k.endswith("bias")) and ("bn" not in k):
                    tmp = k[:-5]
                    # with torch.no_grad():
                    #     tmp_before_B_b += torch.sum(torch.mul(v.data, 1-mask_dict_B_b[tmp]))
                    v.grad = torch.mul(v.grad, mask_cur_B_b[tmp])
        ## Gradient Pruning method ends

        optimizer.step()
        # tmp_after_F_w = 0
        # tmp_after_F_b = 0
        # tmp_after_B_w = 0
        # tmp_after_B_b = 0
        # with torch.no_grad():
        #     for k, v in netF.named_parameters():
        #         if k.endswith("weight"):
        #             # print("grad pruned layer = " + k)
        #             # print("Grad before grad_pruning")
        #             # print(v.grad)
        #             tmp = k[:-7]
        #             with torch.no_grad():
        #                 tmp_after_F_w += torch.sum(torch.mul(v.data, 1 - mask_dict_F_w[tmp]))
        #         elif k.endswith("bias"):
        #             tmp = k[:-5]
        #             with torch.no_grad():
        #                 tmp_after_F_b += torch.sum(torch.mul(v.data, 1 - mask_dict_F_b[tmp]))
        #             # print("Masked used is ")
        #             # print(mask_dict[tmp])
        #     for k, v in netB.named_parameters():
        #         if k.endswith("weight"):
        #             tmp = k[:-7]
        #             with torch.no_grad():
        #                 tmp_after_B_w += torch.sum(torch.mul(v.data, 1-mask_dict_B_w[tmp]))
        #         elif k.endswith("bias"):
        #             tmp = k[:-5]
        #             with torch.no_grad():
        #                 tmp_after_B_b += torch.sum(torch.mul(v.data, 1-mask_dict_B_b[tmp]))
        # print("tmp_before_F_w= " + str(tmp_before_F_w))
        # print("tmp_after_F_w= " + str(tmp_after_F_w))
        # print("tmp_before_F_b= " + str(tmp_before_F_b))
        # print("tmp_after_F_b= " + str(tmp_after_F_b))
        # print("tmp_before_B_w= " + str(tmp_before_B_w))
        # print("tmp_after_B_w= " + str(tmp_after_B_w))
        # print("tmp_before_B_b= " + str(tmp_before_B_b))
        # print("tmp_after_B_b= " + str(tmp_after_B_b))

        netF.eval()
        netB.eval()
        
        # acc_s_te, _ = cal_acc(dset_loaders['test'], netF, netB, netC, False)
        # acc_list.append(acc_s_te)

        if iter_num == 0 or iter_num % interval_iter == 0 or iter_num == max_iter:
            acc_s_te, _ = cal_acc(dset_loaders['test'], netF, netB, netC, False)
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
    # np.save(osp.join(args.output_dir, "prune_acc_list.npy"))
    if args.issave:
        torch.save(netF.state_dict(), osp.join(args.output_dir, str(num+1) + "_F_pruned.pt"))#CL
        torch.save(netB.state_dict(), osp.join(args.output_dir, str(num+1) + "_B_pruned.pt"))#CL
        torch.save(netC.state_dict(), osp.join(args.output_dir, str(num+1) + "_C_pruned.pt"))#CL
    
    tmpnum=1
    for name, module in netF.named_modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            torch.save(module.state_dict(), osp.join(args.output_dir, str(num+1)+"_F_BN"+str(tmpnum)+".pt"))
            tmpnum+=1
    tmpnum=1
    for name, module in netB.named_modules():
        if isinstance(module, torch.nn.BatchNorm1d):
            torch.save(module.state_dict(), osp.join(args.output_dir, str(num+1)+"_B_BN"+str(tmpnum)+".pt"))
            tmpnum+=1

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
    parser.add_argument('--max_epoch', type=int, default=5, help="max iterations")
    parser.add_argument('--interval', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=64, help="batch_size")
    parser.add_argument('--worker', type=int, default=4, help="number of workers")
    parser.add_argument('--dset', type=str, default='office-home', choices=['VISDA-C', 'office', 'office-home', 'office-caltech'])
    parser.add_argument('--lr', type=float, default=1e-2, help="learning rate")
    parser.add_argument('--net', type=str, default='resnet50', help="alexnet, vgg16, resnet50, res101")
    parser.add_argument('--seed', type=int, default=2020, help="random seed")
    # parser.add_argument('--pf_src_c', type=float, default=1e-2, help="pruning fraction conv")
    # parser.add_argument('--pf_src_bn', type=float, default=0, help="pruning fraction BN")
    parser.add_argument('--pf_c', type=float, default=0.1, help="pruning fraction conv")
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

    folder = '/data3/prasanna/Datasets/Office_home/'

    # TODO: start - carefully edit lines for each run
    num = 0
    args.s = 0
    args.t = 1

    args.output_dir_src = osp.join(args.output_src, names[0][0].upper())
    # args.output_dir_src = osp.join(args.output_src, str(num)+names[0][0].upper())
    # TODO: end - carefully edit lines for each run

    args.s_dset_path = folder + args.dset + '/' + names[args.s] + '_list.txt'
    args.t_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'
    args.test_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'

    args.output_dir = osp.join(args.output, str(num+1)+names[0][0].upper())#CL
    args.name = str(num+1)+names[0][0].upper()#CL

    args.savename = 'par_' + str(args.cls_par)
    args.out_file = open(osp.join(args.output_dir, 'log_pruned.txt'), 'w')
    args.out_file.write(print_args(args)+'\n')
    args.out_file.flush()
    prune_finetune_target(args, num)