import cv2 as cv
import glob
from config import CONFIG 

def preprocess(data_path, box, target_size):
    data_path_1 = data_path + 'normal2/'
    files = glob.glob(data_path_1+'*.bmp')
    for file in files:
        file_cv = cv.imread(file)
        x_start, x_end, y_start, y_end = box
        pic = file_cv[y_start:y_end, x_start:x_end]
        if pic.shape != (156,0,3):
            pic = cv.resize(pic,(target_size,target_size))
            cv.imwrite(file,pic)
        else:
            continue

    data_path_2 = data_path + 'normal3/'
    files = glob.glob(data_path_2+'*.bmp')
    for file in files:
        file_cv = cv.imread(file)
        x_start, x_end, y_start, y_end = box
        pic = file_cv[y_start:y_end, x_start:x_end]
        if pic.shape != (156,0,3):
            pic = cv.resize(pic,(target_size,target_size))
            cv.imwrite(file,pic)
        else:
            continue

    data_path_2 = data_path + 'normal4/'
    files = glob.glob(data_path_2+'*.bmp')
    for file in files:
        file_cv = cv.imread(file)
        x_start, x_end, y_start, y_end = box
        pic = file_cv[y_start:y_end, x_start:x_end]
        if pic.shape != (156,0,3):
            pic = cv.resize(pic,(target_size,target_size))
            cv.imwrite(file,pic)
        else:
            continue
   
    return 
