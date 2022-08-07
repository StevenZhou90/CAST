from dataset_builder.subset_tool import SubsetClass
from dataset_builder.data_splitter_class import FilterClass 
import os

'''Add the path to your attribute dataset here: Make sure it is a npy file'''
attrARR_path = './data/attrArr1000.npy'

'''Add the path to your list of image paths here: Make sure it is a npy file'''
paths_path = './data/paths1000.npy'

'''Add the path to a csv with all attribute column names'''
columns = './data/columns.csv'

filter_tool = FilterClass(attrARR_path, paths_path, columns)

'''Examples of tuple inputs'''
tup1 = ('Heavy_Makeup', 'abs', 0, 100)
tup2 = ('vitor_gender', 0)
tup_list = [tup1, tup2]

new_paths = filter_tool.run(tup_list)

subset_tool = SubsetClass(new_paths, 10, 10000, 10000, 'heavyMakeupAndGender', False, tup_list)
subset_tool.write_val_sets()
