import numpy as np
import os
from numpy.random import default_rng
import random

class SubsetClass():
    def __init__(self, filtered_paths, num_of_val_sets, num_of_matches, num_of_non_matches, file_name, replacement_bool, inputs):
        self.paths = filtered_paths
        self.tracked_paths = self.paths
        self.ids = self.make_id_array()
        self.num_of_val_sets = num_of_val_sets
        self.num_of_matches = num_of_matches #per set
        self.num_of_non_matches = num_of_non_matches #per set
        self.replacement_bool = replacement_bool # Currently only does by replacemnet 
        self.total_matches = self.num_of_val_sets * self.num_of_matches
        self.total_non_matches = self.num_of_val_sets * self.num_of_non_matches
        self.file_name = file_name
        self.input = inputs
        self.dir = os.path.join('validation_sets/', self.file_name)

    '''Create list with where ids change index'''
    def make_id_array(self):
        id_change = [0]
        prev = self.paths[0]

        for i in range(1, self.paths.shape[0]):
            if os.path.dirname(prev) == os.path.dirname(self.paths[i]):
                continue
            else:    # start of new id
                id_change.append(i)
                prev = self.paths[i]
        return id_change

    def make_dir(self):
        if not os.path.exists(self.dir):
            os.mkdir(self.dir)
        else: 
            print('Path already exists') 
        with open(self.dir + '/' + 'description' + '.txt', 'w') as f:
            f.write(str(self.input))

    '''Make two lists one with matches and one with non matches to later pull validation sets from'''
    def draw_matches(self):
        pair_list = []
        non_pair_list = []
        path_copy = self.paths

        while len(pair_list)<self.total_matches and len(path_copy)>1:
            rand_idx = random.randint(0, len(path_copy))
            while(rand_idx+1<len(path_copy)-1 and len(pair_list)<self.total_matches and os.path.dirname(path_copy[rand_idx])==os.path.dirname(path_copy[rand_idx+1])):
                pair_list.append([path_copy[rand_idx], path_copy[rand_idx+1]])
                path_copy = np.delete(path_copy, rand_idx+1)
                path_copy = np.delete(path_copy, rand_idx)
                rand_idx+=2

        while len(non_pair_list)<self.total_non_matches and len(self.paths)>1:
            rand_idx = np.random.randint(len(self.paths), size=2)
            if(max(rand_idx)<len(self.paths) and self.paths[rand_idx[0]]!=self.paths[rand_idx[1]]):
                non_pair_list.append(self.paths[rand_idx])
                self.paths = np.delete(self.paths, max(rand_idx))
                self.paths = np.delete(self.paths, min(rand_idx))

        return np.array(pair_list), np.array(non_pair_list)

    '''write paths to .list file'''
    def write_to_file(self, pair_list, non_pair_list, set_num):
        '''change to allow for file name parameter'''
        with open(self.dir + '/' + str(set_num) +'.list', 'w') as f:
            for matches in pair_list:
                f.write(matches[0] + " " + matches[1] + " 1" + '\n')
            for non in non_pair_list:
                f.write(non[0] + " " + non[1] + " 0" + '\n')

    '''Call to write validation sets'''
    def write_val_sets(self):
        self.make_dir()

        pair_list, non_pair_list = self.draw_matches()

        for val_set in range(0, self.num_of_val_sets):
            temp_pair_list_idx = default_rng().choice(len(pair_list), self.num_of_matches, replace=False)
            temp_pair_list = pair_list[temp_pair_list_idx]

            temp_non_pair_list_idx = default_rng().choice(len(pair_list), self.num_of_matches, replace=False)
            temp_non_pair_list = non_pair_list[temp_non_pair_list_idx]
            self.write_to_file(temp_pair_list, temp_non_pair_list, val_set)
