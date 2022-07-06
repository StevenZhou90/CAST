import numpy as np
import os
from random import randint
from tqdm import tqdm
from time import time
from sys import exit

def filter_train_set(current_mask):
    train_idx_mask = np.load('../data/webface12m_idxs.npy')
    train_idx_mask = ~train_idx_mask
    print('train mask trues', np.sum(train_idx_mask))
    return current_mask * train_idx_mask

# make new directory: validation_sets/set#
def get_save_path():
    existing = os.listdir('../validation_sets')
    existing.remove('README.md')
    if len(existing):
        current_set = max([int(x[-1]) for x in existing])
    else:
        current_set = 0
    print('current set', current_set)
    current_set += 1
    dir_name = os.path.join('..', 'validation_sets', 'set'+str(current_set))
    os.makedirs(dir_name, exist_ok=True)
    return dir_name

""" write to validation_sets/set#/description.txt
    call for each attribute filtered """
def write_description(dir_name, desc):
    with open(os.path.join(dir_name, 'description.txt'), 'a') as f:
        f.write(desc + '\n')

def load_bins():
    attributes = None
    # print('loading attributes..')
    # s = time()
    # attributes = np.load('../data/totalAttrList.npy')
    # print(attributes.shape)
    # print(time()-s, 'seconds to load attributes.')
    paths = np.load('../data/ids.npy')
    assert paths.shape[0] == 41315927
    if os.path.exists('../data/data_index.npy'):
        print('loading data index..')
        s = time()
        data_index = np.load('../data/data_index.npy')
        print(time()-s, 'seconds to load index.')
    else:
        print('making data index..')
        s = time()
        data_index = make_rank(attributes)
        print(time()-s, 'seconds to build index.')
        np.save('../data/data_index', data_index)
        exit()
    cols = read_columns()
    print(cols, '\n')
    assert len(cols) == data_index.shape[1], f'columns names are len {len(cols)} and values are len {data_index.shape[1]}'
    return attributes, paths, data_index

def make_rank(arr):
    attr_list = []
    for i in tqdm(range(0,arr.shape[1])):
        atr_rank = arr[:,i]
        ind_list = np.argsort(atr_rank)

        attr_list.append(ind_list)
    rank_list = np.stack(attr_list, axis=1)
    return rank_list


def write_list_file(pth, pair_list, non_pair_list, paths):
    with open(pth, 'w') as f:
        for matches in pair_list:
            f.write(paths[matches[0]] + " " + paths[matches[1]] + " 1" + '\n')
        for non in non_pair_list:
            f.write(paths[non[0]] + " " + paths[non[1]] + " 0" + '\n')

def write_sets(save_path, sets, paths):
    for i in range(len(sets)):
        match, non_match = sets[i]
        pth = os.path.join(save_path, str(i)+'.list')
        write_list_file(pth, match, non_match, paths)



def read_columns():
    with open('../data/columns.csv', 'r') as f:
        line = f.readline()
    line = line.strip()
    cols = line.split(',')
    return cols

""" percentile to webface42 index """
def p2i(per):
    return int(41315927*per)

def get_col_idx(col):
    cols = read_columns()
    col_idx = cols.index(col)
    return col_idx

def get_id_indexes(paths):
    if os.path.exists('../data/id_indices.npy'):
        print('loading id indices..')
        id_indexes = np.load('../data/id_indices.npy')
    else:
        print('making ids..')
        id_indexes = [0]
        prev = paths[0]
        for i in tqdm(range(1, paths.shape[0])):
            if os.path.dirname(prev) == os.path.dirname(paths[i]):
                continue
            else:    # start of new id
                id_indexes.append(i)
                prev = paths[i]
        assert len(id_indexes) == 2003910, 'identity count not correct'
        id_indexes = np.array(id_indexes)
        np.save('../data/id_indices', id_indexes)
    return id_indexes


def make_pairs_list(mask, paths, replacement=False, pairs=10000, num_sets=10):
    available_indexes = np.where(mask)[0]
    print(available_indexes)
    print(available_indexes.shape)
    id_indexes = get_id_indexes(paths)

    id_list = []
    curr_id = 0
    print('making ids x imgs..')
    for i in tqdm(range(0, available_indexes.shape[0])):
        # print('avail', available_indexes[i], 'curr id index', id_indexes[curr_id], 'curr id', curr_id)
        if curr_id == id_indexes.shape[0]-1:
            id_list[-1].append(available_indexes[i])
            continue
        elif available_indexes[i] < id_indexes[curr_id+1]:
            if len(id_list) == 0:
                id_list.append([])
            id_list[-1].append(available_indexes[i])
        else:
            while available_indexes[i] >= id_indexes[curr_id+1]:
                curr_id += 1
                if curr_id == id_indexes.shape[0]-1:
                    break
            id_list.append([])
            id_list[-1].append(available_indexes[i])


    sets = []
    for set in range(num_sets):
        print(f'Starting set {set}/{num_sets}')
        print('==================================================')
        print('total available ids', len(id_list))
        print('avg imgs/id', np.mean(np.array([len(x) for x in id_list])))
        print('max imgs/id', np.max(np.array([len(x) for x in id_list])))
        print('ids imgs>=2', sum([len(x)>=2 for x in id_list]))
        print('==================================================')
        if not replacement:
            # make matching pairs
            print('making matching pairs..')
            # bar = tqdm(total=pairs)
            matches = []
            while len(matches) < pairs:
                id = randint(0, len(id_list)-1)
                if len(id_list[id]) >= 2:
                    img1, img2 = randint(0, len(id_list[id])-1), randint(0, len(id_list[id])-1)
                    while img1 == img2:
                        img1, img2 = randint(0, len(id_list[id])-1), randint(0, len(id_list[id])-1)
                    matches.append((id_list[id][img1], id_list[id][img2]))
                    if img2 > img1:
                        del id_list[id][img2]
                        del id_list[id][img1]
                    else:
                        del id_list[id][img1]
                        del id_list[id][img2]
                    if len(id_list[id]) == 0:
                        del id_list[id]
                    # bar.update(1)

            # make non matches
            print('making non-matching pairs..')
            # bar = tqdm(total=pairs)
            non_matches = []
            while len(non_matches) < pairs:
                id1 = randint(0, len(id_list)-1)
                id2 = randint(0, len(id_list)-1)
                while id1 == id2:
                    id1 = randint(0, len(id_list)-1)
                    id2 = randint(0, len(id_list)-1)
                if len(id_list[id1]) and len(id_list[id2]):
                    img1, img2 = randint(0, len(id_list[id1])-1), randint(0, len(id_list[id2])-1)
                    non_matches.append((id_list[id1][img1], id_list[id2][img2]))
                    del id_list[id1][img1]
                    del id_list[id2][img2]
                    if id1 > id2:
                        if len(id_list[id1]) == 0:
                            del id_list[id1]
                        if len(id_list[id2]) == 0:
                            del id_list[id2]
                    else:
                        if len(id_list[id2]) == 0:
                            del id_list[id2]
                        if len(id_list[id1]) == 0:
                            del id_list[id1]
                    # bar.update(1)

        sets.append((matches, non_matches))
    return sets


def filter_attribute(mask, col, attributes, data_index):
    col, lower_q, upper_q = col
    low_idx, upper_idx = p2i(lower_q), p2i(upper_q)
    col_idx = get_col_idx(col)
    indexes = data_index[low_idx:upper_idx, col_idx]
    attribute_mask = np.zeros(data_index.shape[0], dtype=bool)
    attribute_mask[indexes] = True
    mask = mask * attribute_mask
    return mask


# pass list of tuples
def make_subset(attr_tuples, attributes, data_index, paths, save_path=False):
    if not save_path:
        save_path = get_save_path()
    else:
        os.makedirs(save_path, exist_ok=True)
    mask = np.ones(data_index.shape[0], dtype=bool)
    print('filtering attributes..')
    mask = filter_train_set(mask)
    for col in attr_tuples:
        mask = filter_attribute(mask, col, attributes, data_index)
        col, lower_q, upper_q = col
        write_description(save_path, f'attribute: {col} lower_quartile: {lower_q} upper_quartile: {upper_q}')

    sets = make_pairs_list(mask, paths)
    write_sets(save_path, sets, paths)



if __name__ == '__main__':
    attributes, paths, data_index = load_bins()
    assert data_index.shape[0] == 41315927
    qual_columns = ['nima','brisque','pzq2piq','sdd-fiqa','cr-fiqa']
    for col in qual_columns:
        for i in range(10):
            lower_q = i/10
            upper_q = (i+1)/10
            tup = [(col, lower_q, upper_q)]
            save_path = os.path.join('..', 'validation_sets', col+str(lower_q)+'-'+str(upper_q))
            make_subset(tup, attributes, data_index, paths, save_path=save_path)
