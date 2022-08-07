import numpy as np
import os
from numpy.random import default_rng
'''need to make draw by replacement option'''
'''need to drop all files into one dir check wes code'''
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
        unused_ids = self.ids
        match_count = 0
        non_match_count = 0
        pair_list = []
        non_pair_list = []

        while match_count < self.total_matches and len(unused_ids)!=0:
            rand_select = np.random.choice(self.ids)

            if(os.path.dirname(self.paths[rand_select])==os.path.dirname(self.paths[rand_select+1])):
                pair_list.append([self.paths[rand_select], self.paths[rand_select+1]])

                unused_ids = np.delete(unused_ids, np.where(unused_ids==self.paths[rand_select]))
                unused_ids = np.delete(unused_ids, np.where(unused_ids==self.paths[rand_select+1]))

                rand_select+=2

                match_count+=1
            self.ids = np.delete(self.ids, np.where(rand_select==self.ids))
        
        while non_match_count < self.total_non_matches and len(self.paths):
            rand_select = np.random.choice(self.paths, 2)
            if(os.path.dirname(rand_select[0])!=os.path.dirname(rand_select[1])):
                non_pair_list.append(rand_select)
                self.paths = np.delete(self.paths, np.where(self.paths==rand_select[0]))
                self.paths = np.delete(self.paths, np.where(self.paths==rand_select[1]))
                non_match_count+=1

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

        for i in range(0, self.num_of_val_sets):
            temp_pair_list_idx = default_rng().choice(len(pair_list), self.num_of_matches, replace=False)
            temp_pair_list = pair_list[temp_pair_list_idx]

            temp_non_pair_list_idx = default_rng().choice(len(pair_list), self.num_of_matches, replace=False)
            temp_non_pair_list = non_pair_list[temp_non_pair_list_idx]
            print(i)
            self.write_to_file(temp_pair_list, temp_non_pair_list, i)
