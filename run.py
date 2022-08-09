from dataset_builder.subset_tool import SubsetClass
from dataset_builder.data_splitter_class import FilterClass 
import os

'''Add the path to your attribute dataset here: Make sure it is a npy file'''
attrARR_path = './data/attrArr1000.npy'

'''Add the path to your list of image paths here: Make sure it is a npy file'''
paths_path = './data/paths1000.npy'

'''Add the path to a csv with all attribute column names'''
columns = './data/columns.csv'

'''OPTIONAL: Add indexes of training samples'''
train_set = 'WebFace12M'

filter_tool = FilterClass(attrARR_path, paths_path, columns, train_set)

'''Examples of tuple inputs'''
attr1 = ('Heavy_Makeup', 'abs', 0, 100)
attr2 = ('vitor_gender', 0)
attr_list = [attr1, attr2]

new_paths = filter_tool.run(attr_list)
print(len(new_paths))

subset_tool = SubsetClass(new_paths, 10, 10, 10, 'loser face', False, attr_list)
subset_tool.write_val_sets()
