import datetime
import os
import pickle
import numpy as np
import sklearn
import torch
from torchvision import transforms
from PIL import Image
from scipy import interpolate
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
import argparse
from models import get_model
from time import time
from sys import exit
import mxnet as mx
from mxnet import ndarray as nd
import math


def run_evaluation(args, set_list=None, save_pth='save_pth',
                   data_size=20000, num_sets=1):
    print('init dataloader..')
    dataloader = get_dataloader(args, set_list)
    print('dataloader batches', len(dataloader))
    print('loading model..')
    network = get_model('r100').cuda()
    model_name = 'msv3_r100_af.pt'
    # ckpt = torch.load('/home/wrobbins/Data/models/wf4m_r100_af.pt')
    # ckpt = torch.load('/home/wrobbins/Data/models/glint_r100_cos.pt')
    ckpt = torch.load('/home/wrobbins/Data/models/'+model_name)
    # ckpt = torch.load('/scratch/wrobbins/models/abcd/wf4m_r50_af/model.pt')

    network.load_state_dict(ckpt)
    network = torch.nn.DataParallel(network)

    print('MODEL:', model_name)
    multi_test(dataloader, network, args.batch_size, set_list,
               save_name=save_pth, data_size=data_size, num_sets=num_sets)



""" Class loads multiple validation_sets """
class MultiValidationSet(torch.utils.data.Dataset):
    def __init__(self, wf42_root, set_names, transforms):
        self.transforms = transforms
        self.set_names = set_names
        self.pairs = []
        self.imglist = []
        wf42_root = '/scratch/wrobbins/data/webface42'
        for name in set_names:
            # list_pth = os.path.join('validation_sets', name+'.list')
            list_pth = name
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
        # img = np.ones((112, 112, 3), dtype=np.float32)
        img = self.transforms(img)
        return img

    def __len__(self):
        return len(self.imglist)

# class Synthetic(torch.utils.data.Dataset):
#     def __init__(self):
#         super(Synthetic, self).__init__()
#         img = np.random.randint(0, 255, size=(112, 112, 3), dtype=np.int32)
#         img = np.transpose(img, (2, 0, 1))
#         img = torch.from_numpy(img).squeeze(0).float()
#         img = ((img / 255) - 0.5) / 0.5
#         self.img = img
#         self.label = 1
#
#     def __getitem__(self, index):
#         return self.img
#
#     def __len__(self):
#         return 100000


def get_dataloader(args, set_list):
    transform = transforms.Compose([
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
             ])
    if not set_list:
        raise
        set_list = args.set_name
    dataset = MultiValidationSet(args.data_root, set_list, transform)
    # dataset = Synthetic()
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=args.batch_size)
    return dataloader



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
    return similarity, dist


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
    # similarity = dist
    similarity, dist = distance_(embeddings1, embeddings2)
    # print(np.mean(dist))
    # print(np.mean(dist2))
    # print(np.mean(similarity))
    # exit()
    indices = np.arange(nrof_pairs)


    # for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
    #
    #     # Find the threshold that gives FAR = far_target
    far_train = np.zeros(nrof_thresholds)
    for threshold_idx, threshold in enumerate(thresholds):
        _, far_train[threshold_idx] = calculate_val_far(
            threshold, dist, actual_issame)
    # if np.max(far_train) >= far_target:
    #     f = interpolate.interp1d(far_train, thresholds, kind='slinear')
    #     threshold = f(far_target)
    # else:
    #     threshold = 0.0

    fold_idx = 0
    val[fold_idx], far[fold_idx] = calculate_val_far(
        threshold, dist, actual_issame)

    val_mean = np.mean(val)
    far_mean = np.mean(far)
    val_std = np.std(val)
    return val_mean, val_std, far_mean, similarity


def calculate_val_far(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    true_accept = np.sum(np.logical_and(predict_issame, actual_issame))
    false_accept = np.sum(
        np.logical_and(predict_issame, np.logical_not(actual_issame)))
    n_same = np.sum(actual_issame)
    n_diff = np.sum(np.logical_not(actual_issame))
    # print(n_diff, n_same, 'diff same')
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
    val, val_std, far, dist = calculate_val(thresholds,
                                      embeddings1,
                                      embeddings2,
                                      np.asarray(actual_issame),
                                      1e-3,
                                      nrof_folds=nrof_folds)
    return tpr, fpr, accuracy, val, val_std, far, dist



@torch.no_grad()
def test(data_set, backbone, batch_size, nfolds=1):
    print('testing verification..')
    data_list = data_set[0]
    issame_list = data_set[1]
    embeddings_list = []
    time_consumed = 0.0
    for i in range(len(data_list)):
        data = data_list[i]
        embeddings = None
        ba = 0
        while ba < data.shape[0]:
            bb = min(ba + batch_size, data.shape[0])
            count = bb - ba
            _data = data[bb - batch_size: bb]
            time0 = datetime.datetime.now()
            img = ((_data / 255) - 0.5) / 0.5
            img = torch.tensor(img, device='cuda')
            net_out: torch.Tensor = backbone(img)
            _embeddings = net_out.detach().cpu().numpy()
            time_now = datetime.datetime.now()
            diff = time_now - time0
            time_consumed += diff.total_seconds()
            if embeddings is None:
                embeddings = np.zeros((data.shape[0], _embeddings.shape[1]))
            embeddings[ba:bb, :] = _embeddings[(batch_size - count):, :]
            ba = bb
        embeddings_list.append(embeddings)

    _xnorm = 0.0
    _xnorm_cnt = 0
    for embed in embeddings_list:
        for i in range(embed.shape[0]):
            _em = embed[i]
            _norm = np.linalg.norm(_em)
            _xnorm += _norm
            _xnorm_cnt += 1
    _xnorm /= _xnorm_cnt

    embeddings = embeddings_list[0].copy()
    embeddings = sklearn.preprocessing.normalize(embeddings)
    acc1 = 0.0
    std1 = 0.0
    embeddings = embeddings_list[0] + embeddings_list[1]
    embeddings = sklearn.preprocessing.normalize(embeddings)


    print(embeddings.shape)
    print(len(issame_list))
    print(issame_list)
    print('infer time', time_consumed)
    _, _, accuracy, val, val_std, far = evaluate(embeddings, issame_list, nrof_folds=nfolds)
    acc2, std2 = np.mean(accuracy), np.std(accuracy)
    print(acc2)
    return acc1, std1, acc2, std2, _xnorm, embeddings_list

@torch.no_grad()
def multi_test(dataloader, backbone, batch_size, test_sets, num_sets=1, nfolds=1,
               data_size=40000, save_name='race_gender'):
    print('num sets var', num_sets)
    print('testing verification..')
    # data_list = data_set[0]
    # issame_list = data_set[1]
    embeddings_list = []
    time_consumed = 0.0

    s = time()
    for idx, img in enumerate(dataloader):
        img = img.cuda()
        net_out: torch.Tensor = backbone(img)
        _embeddings = net_out.detach().cpu().numpy()
        embeddings_list.append(_embeddings)

        if idx % 10 == 0:
            with open('run.log', 'a') as f:
                temp = time()
                rate = 10*batch_size / (temp-s)
                s = temp
                f.write(f'rate {rate} @ {idx}/{len(dataloader)}\n')
                print(f'rate {rate} @ {idx}/{len(dataloader)}')

    full_list = np.vstack(embeddings_list)
    print(full_list.shape)
    full_pairs = dataloader.dataset.pairs
    datasize=2000
    evaluate_all_sets(full_list, full_pairs, test_sets, num_sets, nfolds, datasize, save_name)


def evaluate_all_sets(full_list, full_pairs, test_sets, num_sets=1, nfolds=1, data_size=20000, save_name='race_gender'):
    full_pairs = list(map(bool,full_pairs))

    assert full_list.shape[0] == data_size * len(test_sets)
    curr_set = None
    results = []
    avg_results = {}
    # acc,std,norm x num_sets x tests
    for i in range(len(test_sets)):
        if curr_set == None or i % num_sets == 0:
            if i > 0:
                scores = np.array([x[1] for x in results[num_sets*-1:]])
                scores_acc = np.array([x[2] for x in results[num_sets*-1:]])
                avg = np.mean(scores)*100
                std = np.std(scores)*100
                avg_acc = np.mean(scores_acc)*100
                std_acc = np.std(scores_acc)*100
                name = test_sets[i-1].replace('CC11/', '').replace('.list','')[:-2]
                name2 = name
                name = os.path.basename(name).ljust(30, ' ')
                # print(f'{name} AVG tar@far {avg:3f}+/-{std:.3f} acc {avg_acc:3f}+/-{std_acc:.3f}')
                print(f'{name} AVG acc {avg_acc:3f}+/-{std_acc:.3f}')
                avg_results[name2] = (avg_acc, std_acc)
            curr_set = test_sets[i]

        assert os.path.dirname(curr_set) == os.path.dirname(test_sets[i])
        embedding_list = full_list[data_size*i:data_size*(i+1),:]
        list_size = int(data_size / 2)
        issame_list = full_pairs[list_size*i:list_size*(i+1)]

        _xnorm = 0.0
        _xnorm_cnt = 0
        for j in range(embedding_list.shape[0]):
            # for j in range(embed.shape[0]):
            _em = embedding_list[j]
            _norm = np.linalg.norm(_em)
            _xnorm += _norm
            _xnorm_cnt += 1
        _xnorm /= _xnorm_cnt

        # embeddings = embeddings_list[0].copy()
        embeddings = embedding_list
        embeddings = sklearn.preprocessing.normalize(embeddings)
        # embeddings = embeddings_list[0] + embeddings_list[1]
        # embeddings = sklearn.preprocessing.normalize(embeddings)

        # print(embeddings.shape)
        tpr, fpr, accuracy, val, val_std, far, dist = evaluate(embeddings, issame_list, nrof_folds=nfolds)
        acc2, std2 = np.mean(accuracy), np.std(accuracy)
        # print('tpr', tpr)
        # print('fpr', fpr)
        mask = fpr < 1e-3
        tar = np.max(tpr[mask])
        # print('acc2', acc2, 'tar@far', tar)


        # print(test_sets[i].replace('validation_sets/', '').replace('.list','').ljust(40, ' '), 'tar@far', tar, 'best acc', acc2)
        with open('run.log', 'a') as f:
            f.write(f'{test_sets[i]} acc {tar} norm {_xnorm}\n')
        results.append((test_sets[i], tar, acc2, std2, _xnorm))
        # return acc1, std1, acc2, std2, _xnorm, embeddings_list

    scores = np.array([x[1] for x in results[num_sets*-1:]])
    scores_acc = np.array([x[2] for x in results[num_sets*-1:]])
    avg = np.mean(scores)*100
    std = np.std(scores)*100
    avg_acc = np.mean(scores_acc)*100
    std_acc = np.std(scores_acc)*100
    name = test_sets[-1].replace('validation_sets/', '').replace('.list','')[:-2]
    name2 = test_sets[i-1].replace('CC11/', '').replace('.list','')[:-2]
    avg_results[name2] = (avg_acc, std_acc)
    name = os.path.basename(name).ljust(25, ' ')
    # print(f'{name} AVG tar@far {avg:3f}+/-{std:.3f}  acc {avg_acc:3f}+/-{std_acc:.3f}')
    print(f'{name} AVG acc {avg_acc:3f}+/-{std_acc:.3f}')



    overall_acc = np.array([x[2] for x in results])
    avg_overall = np.mean(overall_acc)*100
    std_overall = np.std(overall_acc)*100
    print(f'overall AVG acc {avg_overall:3f}+/-{std_overall:.3f}')

    order = ['black', 'caucasian', 'east_asian', 'latinx', 'middle_eastern',
             'young', 'female', 'male', 'glasses_facial_hair', 'low_p2p', 'random']
    # order = ['young', 'male']

    print(avg_results.keys())
    print('& ', end='')
    for idx, name in enumerate(order):
        acc2, std2 = avg_results[name]
        print(f'{acc2:.2f}\pmf{{{std2:.2f}}} &', end=' ')
        if idx == 5:
            print('\\\\')
            print('& ', end='')
    print(f'{avg_overall:.2f}\pmf{{{std_overall:.2f}}} \\\\')


    os.makedirs('results', exist_ok=True)
    with open(os.path.join('results', save_name+'.p'), 'wb') as f:
        pickle.dump(results, f)
    return results


def get_distance(embeddings, pairs):
    pairs = list(map(bool,  pairs))
    embeddings = sklearn.preprocessing.normalize(embeddings)
    tpr, fpr, accuracy, val, val_std, far, dist = evaluate(embeddings, pairs, nrof_folds=1)
    mask = fpr < 1e-3
    tar = np.max(tpr[mask])
    print(f'tar@far {tar:.5f} acc {accuracy[0]:.5f}')
    return dist


@torch.no_grad()
def load_bin(path, image_size):
    try:
        with open(path, 'rb') as f:
            bins, issame_list = pickle.load(f)  # py2
    except UnicodeDecodeError as e:
        with open(path, 'rb') as f:
            bins, issame_list = pickle.load(f, encoding='bytes')  # py3
    data_list = []
    for flip in [0, 1]:
        data = torch.empty((len(issame_list) * 2, 3, image_size[0], image_size[1]))
        data_list.append(data)
    for idx in range(len(issame_list) * 2):
        _bin = bins[idx]
        img = mx.image.imdecode(_bin)
        if img.shape[1] != image_size[0]:
            img = mx.image.resize_short(img, image_size[0])
        img = nd.transpose(img, axes=(2, 0, 1))
        for flip in [0, 1]:
            if flip == 1:
                img = mx.ndarray.flip(data=img, axis=2)
            data_list[flip][idx][:] = torch.from_numpy(img.asnumpy())
        if idx % 1000 == 0:
            print('loading bin', idx)
    print(data_list[0].shape)
    return data_list, issame_list