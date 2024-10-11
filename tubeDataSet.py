import numpy as np
from torch.utils.data import Dataset
import glob
import cv2 as cv
import torch

def get_file_path(data_path):
    all_file = []

    data_path_1 = data_path + 'exception_1/'
    files = glob.glob(data_path_1+'*_*_*_T3_*.bmp')
    for file_T3 in files:
        file_base = file_T3.split('T3')
        file_T5 = glob.glob(file_base[0]+'T5_*.bmp')
        if len(file_T5) != 1:
            continue
        file_T5 = file_T5[0]
        all_file.append((file_T3, file_T5))

    data_path_2 = data_path + 'normal_1/'
    files = glob.glob(data_path_2+'*_*_*_T3_*.bmp')
    for file_T3 in files:
        file_base = file_T3.split('T3')
        file_T5 = glob.glob(file_base[0]+'T5_*.bmp')
        if len(file_T5) != 1:
            continue
        file_T5 = file_T5[0]
        all_file.append((file_T3, file_T5))
    
    return all_file


def get_keyinfo(lable_file):
    if  'normal' in lable_file:
       classes_1 = 1
       classes_2 = 0
    if  'exception' in lable_file:
       classes_1 = 0
       classes_2 = 1
    key_info = {1:classes_1,
                2:classes_2
                }
    return key_info


def preprocess(file_white, file_blue, box, target_size): 

    pic_white = file_white
    pic_white = pic_white.transpose((2,0,1)) # 使用np.transpose，将图像的维度顺序从(H,W,C)变为(C,H,W)
    
    pic_blue = file_blue
    pic_blue = pic_blue.transpose((2,0,1))

    return np.concatenate([pic_white, pic_blue], axis=0)


class tubeDataset(Dataset):
    def __init__(self, data_path, box, target_size):
        self.all_file = get_file_path(data_path)
        self.box = box
        self.target_size = target_size
        self.total_num = len(self.all_file)

    def __getitem__(self, index):
        path_white, path_blue = self.all_file[index]
        file_white, file_blue = cv.imread(path_white), cv.imread(path_blue) 
        key_info = get_keyinfo(path_white)

        '''
        #增加样本多样性
        x_left, x_right, y_up, y_down = self.box
        x_left = x_left - np.random.randint(0,50)
        x_right = x_right + np.random.randint(0,50)
        y_up = y_up - np.random.randint(0,50)
        y_down = y_down + np.random.randint(0,50)
        '''
        
        pic = preprocess(file_white, file_blue, self.box, self.target_size)

        key_list = key_info[1]
        return pic.astype(np.float32), torch.tensor(key_list,dtype=torch.long)    
    def __len__(self):
        return self.total_num
    
    def getlenofclass(self):
        num_exception = 0
        num_normal = 0 
        for file in self.all_file:
            key_info = get_keyinfo(file[0])
            if key_info[1] == 1:
                num_normal = num_normal + 1
            if key_info[1] == 0:
                num_exception = num_exception +1
        return num_normal,num_exception


