import numpy as np
import csv

class filter_class():
    def __init__(self, attr_paths, path_paths, columns):
        self.attrArr = np.load(attr_paths)
        self.current_mask = np.ones(1000, dtype=bool)
        self.paths = np.load(path_paths)
        self.columns = columns
        self.columns_list = self.csv_to_list()

    def csv_to_list(self):
        column_list = []
        with open(self.columns) as f:
            array = csv.reader(f)
            for row in array:
                for i in row:
                    column_list.append(i)
        return column_list

    def attr_name_to_idx(self, attr_name):
        return self.columns_list.index(attr_name)

    '''Function that returns single column of index over values'''
    def make_rank(self, arr, idx):
        atr_rank = arr[:,idx]
        ind_list = np.argsort(atr_rank)
        return ind_list

    '''Function that create index over values array for whole array'''
    '''Delete if above accopmlishes task'''
    def make_rank_old(self, arr):
        attr_list = []
        for i in range(0,arr.shape[1]):
            atr_rank = arr[:,i]
            ind_list = np.argsort(atr_rank)

            attr_list.append(ind_list)
        rank_list = np.stack(attr_list, axis=1)
        return rank_list

    '''Function that creates mask based on bounds given'''
    '''Need to change atr into index, for now just ask for index'''
    '''style is if the sort is by attribute value or by percentile rank'''
    '''if by percentile rank give percentile as 0-100'''
    def make_range_mask(self, attribute_name, style, lower_range, upper_range): 
        idx = self.attr_name_to_idx(attribute_name)
        ranked_list = self.make_rank(self.attrArr, idx)

        if(style=='abs'): 
            lower_mask = self.attrArr[:,idx] > lower_range  
            upper_mask = self.attrArr[:,idx] < upper_range
            mask = lower_mask*upper_mask

        if(style=='rank'):
            upper_bound_rank_idx = round((self.attrArr.shape[0]*upper_range/100))-1
            lower_bound_rank_idx = round((self.attrArr.shape[0]*lower_range/100))

            upper_bound_idx = ranked_list[upper_bound_rank_idx]
            lower_bound_idx = ranked_list[lower_bound_rank_idx]

            upper_bound = self.attrArr[upper_bound_idx, idx]
            lower_bound = self.attrArr[lower_bound_idx, idx]
            
            lower_mask = self.attrArr[:,idx] > lower_bound  
            upper_mask = self.attrArr[:,idx] < upper_bound
            mask = lower_mask*upper_mask

        return mask

    '''Function to create mask that filters by specified trait in specified column'''
    def make_classification_mask(self, attribute_name, filter_class):
        idx = self.attr_name_to_idx(attribute_name)
        mask = self.attrArr[:, idx]==filter_class
        return mask

    '''Function that adds new mask to the current mask that will be used to create new subset'''
    def add_mask(self, tup):
        assert len(tup)==2 or len(tup)==4
        if len(tup)==2:
            mask = self.make_classification_mask(*tup)
        if len(tup)==4:
            mask = self.make_range_mask(*tup)
        self.current_mask = self.current_mask * mask
        return self.current_mask

    def make_new_paths(self):
        return self.paths[self.current_mask]
