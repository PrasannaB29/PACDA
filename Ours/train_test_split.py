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
from data_list import ImageList
import random, pdb, math, copy
from loss import CrossEntropyLabelSmooth

def main(args):
    folder = '/data3/prasanna/Datasets/Office_home/'
    if args.dset == 'office-home':
        names = ['Art', 'Clipart', 'Product', 'RealWorld']
        # names = ['Clipart', 'Product', 'RealWorld', 'Art']
        # names = ['Clipart', 'Art', 'Product', 'RealWorld']
        # names = ['Product', 'Art', 'Clipart', 'RealWorld']
        # names = ['RealWorld', 'Product', 'Clipart', 'Art']
        # names = ['RealWorld', 'Art', 'Clipart', 'Product']
        args.class_num = 65 

    args.s_dset_path = folder + args.dset + '/' + names[args.s] + '_list.txt'

    txt_src = open(args.s_dset_path).readlines()

    dsize = len(txt_src)
    tr_size = int(0.8*dsize)
    # print(dsize, tr_size, dsize - tr_size)
    tr_txt, te_txt = torch.utils.data.random_split(txt_src, [tr_size, dsize - tr_size])
    print(str(tr_txt))
    print("hello")
    s_tr_file = folder + args.dset + '/' + names[args.s] +'_train_list.txt'
    s_te_file = folder + args.dset + '/' + names[args.s] +'_test_list.txt'

    with open(s_tr_file, 'w') as f:
        f.write(tr_txt)
    
    with open(s_te_file, 'w') as f:
        f.write(te_txt)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SHOT')
    parser.add_argument('--s', type=int, default=0, help="source")
    parser.add_argument('--t', type=int, default=1, help="target")
    parser.add_argument('--dset', type=str, default='office-home', choices=['VISDA-C', 'office', 'office-home', 'office-caltech'])
    parser.add_argument('--seed', type=int, default=2020, help="random seed")
    parser.add_argument('--da', type=str, default='uda', choices=['uda', 'pda', 'oda'])
    parser.add_argument('--trte', type=str, default='val', choices=['full', 'val'])
    args = parser.parse_args()
    main(args)