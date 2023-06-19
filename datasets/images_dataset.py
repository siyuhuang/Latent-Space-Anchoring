import os
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from PIL import Image, ImageOps
from utils import data_utils
import numpy as np
import pandas as pd

from configs.paths_config import dataset_paths

label_list = {0:'background', 1:'skin', 2:'nose', 3:'eye_g', 4:'l_eye', 5:'r_eye', 6:'l_brow', 7:'r_brow', 8:'l_ear', 9:'r_ear', 10:'mouth', 11:'u_lip', 12:'l_lip', 13:'hair', 14:'hat', 15:'ear_r', 16:'neck_l', 17:'neck', 18:'cloth'}
classes = {}
classes[2] = {0:[0,8,9,13,14,15,16,17,18], 1:[1,2,3,4,5,6,7,10,11,12]} # add face
classes[7] = {0:[0,16,17,18], 1:[1,2,3,4,5,6,7,10,11,12], 2:[8], 3:[9], 4:[13], 5:[14], 6:[15]} # add hair and ears
classes[16] = {0:[0,16,17,18]} # add facial features
classes[19] = {} # add neck and cloth

def get_face_attribute_filelist(face_attribute, face_attribute_value):
    # Read CSV file
    df = pd.read_csv(dataset_paths['celeba_face_attribute'], skiprows=1, delimiter=' ')
    index = list(df.index[df[face_attribute] == face_attribute_value])
    file_list = [x[0] for x in index]
    return file_list

class ImagesDataset(Dataset):
    def __init__(self, source_root, opts, source_transform=None, preload=False, phase='train'):
        if opts.face_attribute is not None:
            if phase == 'test':
                face_attribute_value = -opts.face_attribute_value
            else:
                face_attribute_value = opts.face_attribute_value
            file_list = get_face_attribute_filelist(opts.face_attribute, face_attribute_value)
            print(opts.face_attribute, face_attribute_value, file_list[:10])
            self.source_paths = sorted(data_utils.make_dataset(source_root, file_list))
        else:
            self.source_paths = sorted(data_utils.make_dataset(source_root))
        self.source_transform = source_transform
        self.opts = opts
        self.preload = preload
    def __len__(self):
        return len(self.source_paths)

    def __getitem__(self, index):
        if not self.preload:
            from_path = self.source_paths[index]
            from_im = Image.open(from_path)
            if self.opts.input_nc==3:
                from_im = from_im.convert('RGB')
            else:
                from_im = from_im.convert('L')

            if 'seg' in self.opts.dataset_type:
                from_im = np.array(from_im)
                for key in classes[self.opts.input_nc].keys():
                    for i in range(19):
                        if i in classes[self.opts.input_nc][key]:
                            from_im[from_im==i] = key
                from_im = Image.fromarray(from_im)
                
            if self.source_transform:
                from_im = self.source_transform(from_im)

            return from_im
        
        else:
            return self.from_ims[index]
          
class UnpairedImagesDataset(Dataset):

    def __init__(self, source_root, opts, source_transform=None, input_nc=3, phase='train'):
        if opts.face_attribute is not None:
            if phase == 'test':
                face_attribute_value = -opts.face_attribute_value
            else:
                face_attribute_value = opts.face_attribute_value
            file_list = get_face_attribute_filelist(opts.face_attribute, face_attribute_value)
            self.paths = sorted(data_utils.make_dataset(source_root, file_list))
        else:
            self.paths = sorted(data_utils.make_dataset(source_root))
        self.transform = source_transform
        self.opts = opts

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        im = Image.open(path).convert('RGB')
        if self.transform:
            im = self.transform(im)
        return im

    
