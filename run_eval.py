import os
import argparse
import torch
import numpy as np
from sys import exit

from evaluation.evaluation import load_bin, test, load_from_list, formatted_print
from models import get_model

parser = argparse.ArgumentParser()
parser.add_argument('--set_names', default='', type=str)
parser.add_argument('--weights', type=str)
parser.add_argument('--arch', default='r50', type=str)
parser.add_argument('--batch_size', default=256, type=int)
args = parser.parse_args()

# load model
model = get_model(args.arch).to('cuda')
ckpt = torch.load(args.weights)
model.load_state_dict(ckpt)





acc_ = []
std_ = []
print(' --------------------------')
args.set_names = args.set_names.split(',')
for set in args.set_names:
    # load dataset
    pth = os.path.join('validation_sets', set)
    if args.path.endswith('.bin'):
        dataset = load_bin(args.path)
    else:
        assert os.path.isdir(args.path), f'{args.path} should be .bin or webface42m root'
        dataset = load_from_list(args.path)
        exit()
    _,_,acc,std,_ = test(dataset, model, args.batch_size, nfolds=10)
    acc, std = acc*100, std*100
    acc_.append(acc)
    std_.append(std)
    formatted_print(names[i], acc, std)
# print average over all eval sets
print('\n --------------------------')
formatted_print('Avg', np.mean(np.array(acc_)), np.mean(np.array(std_)))
