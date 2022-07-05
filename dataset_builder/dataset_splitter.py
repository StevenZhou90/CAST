import numpy as np
import os


def filter_train_set(current_mask):
    train_idx_mask = np.load('../data/webface12m_idxs.npy')
    return current_mask * train_idx_mask

# make new directory: validation_sets/set#
def save_path():
    existing = os.listdir('../validation_sets')
    current_set = max([int(x[-1]) for x in existing])
    current_set += 1
    dir_name = os.path.join('..', 'validation_set', 'set'+str(current_set))
    os.makedir(dir_name, exist_ok=True)
    return dir_name

""" write to validation_sets/set#/description.txt
    call for each attribute filtered """
def write_description(dir_name, attribute_desc):
    with open(os.path.join(dir_name, 'description.txt', 'a')) as f:
        f.write(desc + '\n')

newArr = np.load('totalAttrList.npy')
paths = np.load('ids.npy')

r1 = int(input('Enter which category you would like to filter by - (1:Face Attributes, 2:Age, 3:Race or Gender, 4:Image Quality, 5:Done): \n'))

def make_rank(arr):
    attr_list = []
    for i in range(0,arr.shape[1]):
        atr_rank = arr[:,i]
        ind_list = np.argsort(atr_rank)

        attr_list.append(ind_list)
    rank_list = np.stack(attr_list, axis=1)
    return rank_list

while r1!=5:
    if(r1==1):
        ranks = make_rank(newArr)
        atr = int(input('Enter the integer for which attribute you would like to filter by - (0:5_o_Clock_Shadow, 1:Arched_Eyebrows, 2:Attractive, 3:Bags_Under_Eyes, 4:Bald, 5:Bangs, 6:Big_Lips, 7:Big_Nose, 8:Black_Hair, 9:Blond_Hair, 10:Blurry, 11:Brown_Hair, 12:Bushy_Eyebrows, 13:Chubby, 14:Double_Chin, 15:Eyeglasses, 16:Goatee, 17:Gray_Hair, 18:Heavy_Makeup, 19:High_Cheekbones, 20:Male, 21:Mouth_Slightly_Open, 22:Mustache, 23:Narrow_Eyes, 24:No_Beard, 25:Oval_Face, 26:Pale_Skin, 27:Pointy_Nose, 28:Receding_Hairline, 29:Rosy_Cheeks, 30:Sideburns, 31:Smiling, 32:Straight_Hair, 33:Wavy_Hair, 34:Wearing_Earrings, 35:Wearing_Hat, 36:Wearing_Lipstick, 37:Wearing_Necklace, 38:Wearing_Necktie, 39:Young): \n'))
        section = int(input('Enter the integer corresponding to which section of the dataset you would like to extract - (1:top, 2:middle, 3:bottom): \n'))
        percent = float(input("Enter the percent of values you would like to extract from the previously stated section (0-1): \n"))

        if(section==1):
            upper = newArr.shape[0]
            bottom = round(newArr.shape[0]-(percent*newArr.shape[0]))
            idx=ranks[bottom:upper,atr]
        elif(section==2):
            upper = round((newArr.shape[0]/2)+(newArr.shape[0]*percent/2))
            bottom = round((newArr.shape[0]/2)-(newArr.shape[0]*percent/2))
            idx=ranks[bottom:upper,atr]
        elif(section==3):
            upper = round((percent*newArr.shape[0]))
            bottom = 0
            idx=ranks[bottom:upper,atr]

        mask = np.zeros(newArr.shape[0], dtype=bool)
        mask[idx] = True
        newArr = newArr[mask]
        paths = paths[mask]
        print(newArr.shape)

    elif(r1==2):
        method = int(input("Enter the integer corresponding to the method you would like to filter by - (1:FairFace (regression), 2:Vitor(Classification)): \n"))

        if(method==1):
            age_range = int(input("Enter the integer corresponding to the age range you would like to rank by - (1:0-2, 2:3-9, 3:10-19, 4:20-29, 5:30-39, 6:40-49, 7:50-59, 8:60-69, 9:70+): \n"))
            section = int(input('Enter the integer corresponding to which section of the dataset you would like to extract - (1:top, 2:middle, 3:bottom): \n'))
            percent = float(input("Enter the percent of values you would like to extract from the previously stated section (0-1): \n"))
            ranks = make_rank(newArr)
            atr = age_range + 51
            if(section==1):
                upper = newArr.shape[0]
                bottom = round(newArr.shape[0]-(percent*newArr.shape[0]))
                idx=ranks[bottom:upper,atr]
            elif(section==2):
                upper = round((newArr.shape[0]/2)+(newArr.shape[0]*percent/2))
                bottom = round((newArr.shape[0]/2)-(newArr.shape[0]*percent/2))
                idx=ranks[bottom:upper,atr]
            elif(section==3):
                upper = round((percent*newArr.shape[0]))
                bottom = 0
                idx=ranks[bottom:upper,atr]

            mask = np.zeros(newArr.shape[0], dtype=bool)
            mask[idx] = True
            newArr = newArr[mask]
            paths = paths[mask]

        elif(method==2):
            age = newArr[:,42]
            lower = int(input("Enter the bottom range age you would like: \n"))
            upper = int(input("Enter the upper range age you would like: \n"))
            mask = (age>lower) & (upper>age)

            newArr = newArr[mask]
            paths = paths[mask]

    elif(r1==3):
        method = int(input("Enter the integer corresponding to the method you would like to filter by - (1:FairFace (regression), 2:Vitor(Classification)): \n"))
        attr = int(input("Enter the integer corresponding to the category you would like to filter by - (1:race, 2:gender): \n"))

        if(method==1):
            if(attr==1):
                race = int(input("Enter the corresponding integer to which race you would like to rank by - (0:Caucasian, 1:African-American, 2:Latino, 3:East Asian, 4:Southeast Asian, 5:Indian, 6:Middle Eastern): \n"))
                section = int(input('Enter the integer corresponding to which section of the dataset you would like to extract - (1:top, 2:middle, 3:bottom): \n'))
                percent = float(input("Enter the percent of values you would like to extract from the previously stated section (0-1): \n"))
                ranks = make_rank(newArr)
                atr = age_range + 43
                if(section==1):
                    upper = newArr.shape[0]
                    bottom = round(newArr.shape[0]-(percent*newArr.shape[0]))
                    idx=ranks[bottom:upper,atr]
                elif(section==2):
                    upper = round((newArr.shape[0]/2)+(newArr.shape[0]*percent/2))
                    bottom = round((newArr.shape[0]/2)-(newArr.shape[0]*percent/2))
                    idx=ranks[bottom:upper,atr]
                elif(section==3):
                    upper = round((percent*newArr.shape[0]))
                    bottom = 0
                    idx=ranks[bottom:upper,atr]
                mask = np.zeros(newArr.shape[0], dtype=bool)
                mask[idx] = True
                newArr = newArr[mask]
                paths = paths[mask]

            elif(attr==2):
                gender = int(input("Enter the corresponding integer to which gender you would like to rank by - (0:Male, 1:Female): \n"))
                section = int(input('Enter the integer corresponding to which section of the dataset you would like to extract - (1:top, 2:middle, 3:bottom): \n'))
                percent = float(input("Enter the percent of values you would like to extract from the previously stated section (0-1): \n"))
                ranks = make_rank(newArr)
                atr = gender + 50
                if(section==1):
                    upper = newArr.shape[0]
                    bottom = round(newArr.shape[0]-(percent*newArr.shape[0]))
                    idx=ranks[bottom:upper,atr]
                elif(section==2):
                    upper = round((newArr.shape[0]/2)+(newArr.shape[0]*percent/2))
                    bottom = round((newArr.shape[0]/2)-(newArr.shape[0]*percent/2))
                    idx=ranks[bottom:upper,atr]
                elif(section==3):
                    upper = round((percent*newArr.shape[0]))
                    bottom = 0
                    idx=ranks[bottom:upper,atr]
                mask = np.zeros(newArr.shape[0], dtype=bool)
                mask[idx] = True
                newArr = newArr[mask]
                paths = paths[mask]

        if(method==2):

            if attr==1:
                temp_arr = newArr[:,40]
                mask_bool = int(input("Enter the corresponding integer to which race you would like to rank by - (0:Caucasian, 1:African-American, 2:Latino, 3:Indian, 4:Other): \n"))
            elif attr==2:
                temp_arr = newArr[:,41]
                mask_bool = int(input("Enter the corresponding integer to which gender you would like to filter for - (0:Female, 1:Male): \n"))
            mask = temp_arr==mask_bool

            newArr = newArr[mask]
            paths = paths[mask]

    elif(r1==4):
        attr = int(input("Enter the integer corresponding to the category you would like to filter by - (1:nima, 2:brisque, 3:paq2piq, 4:sdd-fiqa): \n"))
        if attr==1:
            temp_arr = newArr[:,60+attr]
        elif attr==2:
            temp_arr = newArr[:,60+attr]
        elif attr==3:
            temp_arr = newArr[:,60+attr]
        elif attr==4:
            temp_arr = newArr[:,60+attr]

        ranks = make_rank(newArr)
        section = int(input('Enter the integer corresponding to which section of the dataset you would like to extract - (1:top, 2:middle, 3:bottom): \n'))
        percent = float(input("Enter the percent of values you would like to extract from the previously stated section (0-1): \n"))

        if(section==1):
            upper = temp_arr.shape[0]
            bottom = round(temp_arr.shape[0]-(percent*temp_arr.shape[0]))
            idx=ranks[bottom:upper,50+attr]
        elif(section==2):
            upper = round((temp_arr.shape[0]/2)+(temp_arr.shape[0]*percent/2))
            bottom = round((temp_arr.shape[0]/2)-(temp_arr.shape[0]*percent/2))
            idx=ranks[bottom:upper,50+attr]
        elif(section==3):
            upper = round((percent*temp_arr.shape[0]))
            bottom = 0
            idx=ranks[bottom:upper,50+attr]

        mask = np.zeros(newArr.shape[0], dtype=bool)
        mask[idx] = True
        newArr = newArr[mask]
        paths = paths[mask]
    print(newArr.shape)
    r1 = int(input('Enter which category you would like to filter by - (1:Face Attributes, 2:Age, 3:Race or Gender, 4:Image Quality, 5:Done): \n'))

id_change = [0]
prev = paths[0]

for i in range(1, paths.shape[0]):
    if os.path.dirname(prev) == os.path.dirname(paths[i]):
        continue
    else:    # start of new id
        id_change.append(i)
        prev = paths[i]

used_paths = np.ones(len(paths), dtype = bool)

print("There are " + str(len(id_change)) + " identities and " + str(len(paths)) + "images.")

num_of_matches = int(input("Enter the number of matches you would like in the validation set: \n"))
num_of_non_matches = int(input("Enter the number of non-matches you would like in the validation set: \n"))

non_pair_list = []
pair_list = []

while len(pair_list)<=num_of_matches & len(id_change)!=0:
    temp_id = np.random.choice(id_change)
    id_change = np.delete(id_change, np.where(temp_id == id_change)[0])
    dup = True
    while dup:
        if((os.path.dirname(paths[temp_id])==os.path.dirname(paths[temp_id+1])) & ((temp_id+1)<len(paths))):
            pair_list.append([paths[temp_id], paths[temp_id+1]])
            used_paths[temp_id] = False
            used_paths[temp_id+1] = False

            temp_id+=2
        else:
            dup = False

non_pair_paths = paths[used_paths]

while len(pair_list)<=num_of_non_matches | len(non_pair_paths)!=0:
    temp_path = np.random.choice(non_pair_paths, size=2, replace=False)

    non_pair_paths = np.delete(non_pair_paths, np.where(non_pair_paths==temp_path[0])[0])
    non_pair_paths = np.delete(non_pair_paths, np.where(non_pair_paths==temp_path[1])[0])
    if(temp_path[0]!=temp_path[1]):
        non_pair_list.append([temp_path[0], temp_path[1]])

with open('val.list', 'w') as f:
    num_match = 0
    for matches in pair_list:
        f.write(matches[0] + " " + matches[1] + " 1" + '\n')
        num_match+=1
        if(num_match >= num_of_matches):
            break;

    num_non = 0
    for non in non_pair_list:
        f.write(non[0] + " " + non[1] + " 0" + '\n')
        num_non+=1
        if(num_non >= num_of_non_matches):
            break;
