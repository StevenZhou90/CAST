import datetime
import os
import pickle
import numpy as np
import sklearn
import torch
import cv2
from torchvision import transforms
from PIL import Image
from scipy import interpolate
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
import argparse
from models import get_model
import mxnet as mx
import math
from tqdm import tqdm


def run_evaluation(args):
    # dataloader = get_dataloader(args)
    network = get_model('r50')


""" Class loads multiple validation_sets """
class MultiValidationSet(torch.utils.data.Dataset):
    def __init__(self, wf42_root, set_names, transforms):
        self.transforms = transforms
        self.set_names = set_names
        self.pairs = []
        self.imglist = []
        for name in set_names:
            list_pth = os.path.join('validation_sets', name+'.list')
            with open(list_pth, 'r') as f:
                for line in f:
                    line = line.strip()
                    line = line.split()
                    img1 = os.path.join(wf42_root, line[0])
                    img2 = os.path.join(wf42_root, line[1])
                    self.imglist.append(img1)
                    self.imglist.append(img2)
                    self.pairs.append(int(line[2]))

    def __getitem__(self, idx):
        img = Image.open(self.imglist[idx])
        img = self.transforms(img)
        return img

    def __len__(self):
        return len(self.imglist)


def get_dataloader(args):
    transform = transforms.Compose([
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
             ])
    dataset = MultiValidationSet(args.data_root, args.set_names, transform)
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=args.batch_size)
    return dataloader


# results formatting
def formatted_print(row_name, acc, std, sep=False):
    if sep:
        print(' --------------------------------')
    print(f'| {row_name.ljust(17)}   {acc:.2f}+-{std:.2f}|')
    print(' --------------------------------')

class LFold:
    def __init__(self, n_splits=2, shuffle=False):
        self.n_splits = n_splits
        if self.n_splits > 1:
            self.k_fold = KFold(n_splits=n_splits, shuffle=shuffle)

    def split(self, indices):
        if self.n_splits > 1:
            return self.k_fold.split(indices)
        else:
            return [(indices, indices)]



def calculate_roc(thresholds,
                  embeddings1,
                  embeddings2,
                  actual_issame,
                  nrof_folds=10,
                  pca=0):
    assert (embeddings1.shape[0] == embeddings2.shape[0])
    assert (embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = LFold(n_splits=nrof_folds, shuffle=False)

    tprs = np.zeros((nrof_folds, nrof_thresholds))
    fprs = np.zeros((nrof_folds, nrof_thresholds))
    accuracy = np.zeros((nrof_folds))
    indices = np.arange(nrof_pairs)

    if pca == 0:
        diff = np.subtract(embeddings1, embeddings2)
        dist = np.sum(np.square(diff), 1)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        if pca > 0:
            print('doing pca on', fold_idx)
            embed1_train = embeddings1[train_set]
            embed2_train = embeddings2[train_set]
            _embed_train = np.concatenate((embed1_train, embed2_train), axis=0)
            pca_model = PCA(n_components=pca)
            pca_model.fit(_embed_train)
            embed1 = pca_model.transform(embeddings1)
            embed2 = pca_model.transform(embeddings2)
            embed1 = sklearn.preprocessing.normalize(embed1)
            embed2 = sklearn.preprocessing.normalize(embed2)
            diff = np.subtract(embed1, embed2)
            dist = np.sum(np.square(diff), 1)

        # Find the best threshold for the fold
        acc_train = np.zeros((nrof_thresholds))
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, acc_train[threshold_idx] = calculate_accuracy(
                threshold, dist[train_set], actual_issame[train_set])
        best_threshold_index = np.argmax(acc_train)
        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx, threshold_idx], fprs[fold_idx, threshold_idx], _ = calculate_accuracy(
                threshold, dist[test_set],
                actual_issame[test_set])
        _, _, accuracy[fold_idx] = calculate_accuracy(
            thresholds[best_threshold_index], dist[test_set],
            actual_issame[test_set])

    tpr = np.mean(tprs, 0)
    fpr = np.mean(fprs, 0)
    return tpr, fpr, accuracy


def calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(
        np.logical_and(np.logical_not(predict_issame),
                       np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    acc = float(tp + tn) / dist.size
    return tpr, fpr, acc


def distance_(embeddings0, embeddings1):
    # Distance based on cosine similarity
    dot = np.sum(np.multiply(embeddings0, embeddings1), axis=1)
    norm = np.linalg.norm(embeddings0, axis=1) * np.linalg.norm(embeddings1, axis=1)
    # shaving
    similarity = np.clip(dot / norm, -1., 1.)
    dist = np.arccos(similarity) / math.pi
    return dist


def calculate_val(thresholds,
                  embeddings1,
                  embeddings2,
                  actual_issame,
                  far_target,
                  nrof_folds=10):
    assert (embeddings1.shape[0] == embeddings2.shape[0])
    assert (embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = LFold(n_splits=nrof_folds, shuffle=False)

    val = np.zeros(nrof_folds)
    far = np.zeros(nrof_folds)

    # diff = np.subtract(embeddings1, embeddings2)
    # dist = np.sum(np.square(diff), 1)
    dist = distance_(embeddings1, embeddings2)
    indices = np.arange(nrof_pairs)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):

        # Find the threshold that gives FAR = far_target
        far_train = np.zeros(nrof_thresholds)
        for threshold_idx, threshold in enumerate(thresholds):
            _, far_train[threshold_idx] = calculate_val_far(
                threshold, dist[train_set], actual_issame[train_set])
        if np.max(far_train) >= far_target:
            f = interpolate.interp1d(far_train, thresholds, kind='slinear')
            threshold = f(far_target)
        else:
            threshold = 0.0

        val[fold_idx], far[fold_idx] = calculate_val_far(
            threshold, dist[test_set], actual_issame[test_set])

    val_mean = np.mean(val)
    far_mean = np.mean(far)
    val_std = np.std(val)
    return val_mean, val_std, far_mean


def calculate_val_far(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    true_accept = np.sum(np.logical_and(predict_issame, actual_issame))
    false_accept = np.sum(
        np.logical_and(predict_issame, np.logical_not(actual_issame)))
    n_same = np.sum(actual_issame)
    n_diff = np.sum(np.logical_not(actual_issame))
    # print(true_accept, false_accept)
    # print(n_same, n_diff)
    val = float(true_accept) / float(n_same)
    far = float(false_accept) / float(n_diff)
    return val, far


def evaluate(embeddings, actual_issame, nrof_folds=10, pca=0):
    # Calculate evaluation metrics
    thresholds = np.arange(0, 4, 0.01)
    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]
    tpr, fpr, accuracy = calculate_roc(thresholds,
                                       embeddings1,
                                       embeddings2,
                                       np.asarray(actual_issame),
                                       nrof_folds=nrof_folds,
                                       pca=pca)
    thresholds = np.arange(0, 4, 0.001)
    val, val_std, far = calculate_val(thresholds,
                                      embeddings1,
                                      embeddings2,
                                      np.asarray(actual_issame),
                                      1e-3,
                                      nrof_folds=nrof_folds)
    return tpr, fpr, accuracy, val, val_std, far



@torch.no_grad()
def test(data_set, backbone, batch_size, nfolds=10):
    data_list = data_set[0]
    issame_list = data_set[1]
    time_consumed = 0.0
    data = data_list
    embeddings = None
    ba = 0
    while ba < data.shape[0]:
        bb = min(ba + batch_size, data.shape[0])
        count = bb - ba
        _data = data[bb - batch_size: bb]
        time0 = datetime.datetime.now()
        img = ((_data / 255) - 0.5) / 0.5
        img = img.to('cuda')
        net_out: torch.Tensor = backbone(img)
        _embeddings = net_out.detach().cpu().numpy()
        time_now = datetime.datetime.now()
        diff = time_now - time0
        time_consumed += diff.total_seconds()
        if embeddings is None:
            embeddings = np.zeros((data.shape[0], _embeddings.shape[1]))
        embeddings[ba:bb, :] = _embeddings[(batch_size - count):, :]
        ba = bb

    _xnorm = 0.0
    _xnorm_cnt = 0
    embed = embeddings
    for i in range(embed.shape[0]):
        _em = embed[i]
        _norm = np.linalg.norm(_em)
        _xnorm += _norm
        _xnorm_cnt += 1
    _xnorm /= _xnorm_cnt

    embeddings = sklearn.preprocessing.normalize(embeddings)
    acc1 = 0.0
    std1 = 0.0
    embeddings = sklearn.preprocessing.normalize(embeddings)
    _, _, accuracy, val, val_std, far = evaluate(embeddings, issame_list, nrof_folds=nfolds)
    acc2, std2 = np.mean(accuracy), np.std(accuracy)
    return acc1, std1, acc2, std2, _xnorm

@torch.no_grad()
def load_bin(path, image_size=(112, 112)):
    try:
        with open(path, 'rb') as f:
            bins, issame_list = pickle.load(f)  # py2
    except UnicodeDecodeError as e:
        with open(path, 'rb') as f:
            bins, issame_list = pickle.load(f, encoding='bytes')  # py3
    data = torch.empty((len(issame_list) * 2, 3, image_size[0], image_size[1]))
    print('loading images...')
    for idx in tqdm(range(len(issame_list) * 2)):
        _bin = bins[idx]
        img = mx.image.imdecode(_bin)
        if img.shape[1] != image_size[0]:
            img = mx.image.resize_short(img, image_size[0])
        img = np.transpose(img, axes=(2, 0, 1))
        data[idx][:] = torch.from_numpy(img.asnumpy())
    print(data.shape)
    return data, issame_list


@torch.no_grad()
def load_from_list(data_root, list_dir):
    list_files = [os.path.join(list_dir, x) for x in os.listdir(list_dir)]
    list_files = list(filter(lambda x: os.path.basename(x).endswith('.list') and
                            os.path.basename(x).split('.')[0].isdigit(), list_files))
    return read_paths(list_files, data_root)

@torch.no_grad()
def load_CC11_list(data_root):
    list_dir = 'validation_sets/CC11'
    list_files = sub_dirs = ['black', 'caucasian', 'east_asian', 'latinx',
                            'middle_eastern', 'young', 'female', 'male',
                            'glasses_facial_hair', 'low_p2p', 'random']
    temp_list_files = []
    for d in list_files:
        if os.path.isdir(os.path.join(list_dir, d)):
            temp_list_files += [os.path.join(list_dir, d, x) for x in os.listdir(os.path.join(list_dir,d))]
    list_files = temp_list_files
    list_files = list(filter(lambda x: os.path.basename(x).endswith('.list') and
                            os.path.basename(x).split('.')[0].isdigit(), list_files))
    # from pprint import pprint
    # pprint(list_files)
    return read_paths(list_files, data_root)


def read_paths(list_files, data_root):
    bins = []
    issame_list = []
    for l_file in list_files:
        with open(l_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                issame_list.append(bool(int(line.strip().split(' ')[-1])))
    data = torch.empty((len(issame_list) * 2, 3, 112, 112))
    idx = 0
    print('loading images...')
    for l_file in tqdm(list_files):
        for line in open(l_file, 'r'):
            line = line.strip().split()
            assert len(line) == 3
            path1 = os.path.join(data_root, line[0])
            path2 = os.path.join(data_root, line[1])
            img1 = cv2.imread(path1)
            img2 = cv2.imread(path2)
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
            img1 = np.transpose(img1, axes=(2, 0, 1))
            img2 = np.transpose(img2, axes=(2, 0, 1))
            data[idx][:] = torch.from_numpy(img1)
            data[idx+1][:] = torch.from_numpy(img2)
            idx +=2
    return data, issame_list




    #                 with open(output, 'wb') as f:
    # pickle.dump((bins, issame_list), f, protocol=4)
