from dataset_builder.subset_tool import subset_class #To be imported
from dataset_builder.data_splitter import filter_class 

'''Add the path to your attribute dataset here: Make sure it is a npy file'''
attrARR_path = './data/attrArr1000.npy'
'''Add the path to your list of image paths here: Make sure it is a npy file'''
paths_path = './data/paths1000.npy'
'''Add the path to a csv with all attribute column names'''
columns = './data/columns.csv'

filter_tool = filter_class(attrARR_path, paths_path, columns)

'''Examples of tuple inputs'''
tup1 = ('Heavy_Makeup', 'abs', 0, 100)
tup2 = ('vitor_gender', 0)
tup_list = [tup1, tup2]

'''example on how tuples will be added'''
for tup in tup_list:
  filter_tool.add_mask(tup)
  
new_paths = filter_tool.make_new_paths()

subset_tool = SubsetClass(new_paths, 3, 3, 3, 'test_file_name', False)

subset_tool.write_val_sets()
