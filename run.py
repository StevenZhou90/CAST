from dataset_builder.filter import FilterClass 
from dataset_builder.subset_tool import SubsetClass

'''Add the path to your attribute dataset here: Make sure it is a npy file'''
attrARR_path = './data/attrArr1000.npy'

'''Add the path to your list of image paths here: Make sure it is a npy file'''
paths_path = './data/paths1000.npy'

'''Add the path to a csv with all attribute column names'''
columns_path = './data/columns.csv'

'''OPTIONAL: Add indexes of training samples'''
train_set_path = 'WebFace12M'

filter_tool = FilterClass(attrARR_path, paths_path, columns=columns_path, train_set=None)

'''Examples of tuple inputs'''
attr1 = ('Heavy_Makeup', 'rank', 50, 100)
attr2 = ('vitor_gender', 0)
attr_list = [attr1, attr2]

new_paths = filter_tool.run(attr_list)

'''Create the subset tool and specify parameters for validation sets'''
subset_tool = SubsetClass(new_paths, num_of_val_sets=10, num_of_matches=10000, num_of_non_matches=10000, file_name='Output File Name', replacement_bool=False, inputs=attr_list)

'''Create the output file'''
subset_tool.write_val_sets()
