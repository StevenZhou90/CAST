import os
import argparse
import torch
import numpy as np
from sys import exit

from evaluation.evaluation import load_bin, test, load_CC11_list, formatted_print
from models import get_model

parser = argparse.ArgumentParser()
parser.add_argument('--path', default='data/cc11.bin', type=str)
parser.add_argument('--weights', type=str)
parser.add_argument('--arch', default='r50', type=str)
parser.add_argument('--batch_size', default=256, type=int)
args = parser.parse_args()



# load dataset
if args.path.endswith('.bin'):
    dataset = load_bin(args.path)
    print(type(dataset[0]))
else:
    assert os.path.isdir(args.path), f'{args.path} should be .bin or webface42m root'
    dataset = load_CC11_list(args.path)
    # exit()


# load model
model = get_model(args.arch).to('cuda')
ckpt = torch.load(args.weights)
model.load_state_dict(ckpt)


# results formatting
names = ['Black', 'Caucasian', 'E. Asian', 'Latinx', 'M.E.', 'Young',
        'Female', 'Male', 'G&FH', 'L-p2p', 'Random']


# run eval
sub_benchmarks = 11
acc_ = []
std_ = []
dataset, issame_list = dataset

if isinstance(dataset, torch.Tensor):
    sub_benchmark_size = int(dataset.shape[0]/sub_benchmarks)
    for i in range(sub_benchmarks):
        sub_benchmark = dataset[i*sub_benchmark_size: (i+1)*sub_benchmark_size]
        issame = issame_list[int(i*(sub_benchmark_size/2)): int((i+1)*(sub_benchmark_size/2))]
        _,_,acc,std,_ = test((sub_benchmark, issame), model, args.batch_size, nfolds=10)
        acc, std = acc*100, std*100
        acc_.append(acc)
        std_.append(std)
        if i == 0:
            formatted_print(names[i], acc, std, sep=True)
        else:
            formatted_print(names[i], acc, std)
    # print average over sub-benchmarks
    print()
    formatted_print('Avg', np.mean(np.array(acc_)), np.mean(np.array(std_)), sep=True)
