import os
import torch
import numpy as np 
import h5py
from PIL import Image

class Test_Data():
    def __init__(self, Test_Label_Path, Test_Data_Path, H5py_Path, transform=None, is_multimodal=False):
        self.H5py_Path = H5py_Path
        self.annotations = np.load(Test_Data_Path, allow_pickle=True)  # Read The names of Test Signals 
        self.Label = np.load(Test_Label_Path, allow_pickle=True)
        self.Label = np.array(self.Label)
        self.transform = transform
        self.is_multimodal = is_multimodal
        # print(self.annotations)
    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        key = self.annotations[index]
        with h5py.File(self.H5py_Path, 'r') as f:
            
            
            if self.is_multimodal:
                SG_Data = f[str(key)]
                image_data = Image.fromarray(np.array(SG_Data['image']))
                audio_data = Image.fromarray(np.array(SG_Data['audio']))
                SG_Label = torch.from_numpy(np.array((self.Label[index])))
    
                if self.transform:
                    image_data = self.transform(image_data)
                    audio_data = self.transform(audio_data)
            else:
                SG_Data = f[key][()]
                SG_Data = np.array(SG_Data)
                SG_Label = torch.from_numpy(np.array((self.Label[index])))
            
                ES_Data = Image.fromarray(SG_Data)
                if self.transform:
                    ES_Data = self.transform(ES_Data)
     
        if self.is_multimodal:
            return (audio_data, image_data, SG_Label)
        else:
            return (ES_Data, SG_Label)

class Train_Data():
    def __init__(self, Train_Label_Path, Train_Data_Path, H5py_Path, transform=None, is_multimodal=False):
        self.H5py_Path = H5py_Path
        self.annotations = np.load(Train_Data_Path, allow_pickle=True)  # Read The names of Test Signals 
        self.Label = np.load(Train_Label_Path, allow_pickle=True)
        self.Label = np.array(self.Label)
        self.transform = transform
        self.is_multimodal = is_multimodal 

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        key = self.annotations[index]
        # print(key)
        with h5py.File(self.H5py_Path, 'r') as f: 
            
            if self.is_multimodal:
                SG_Data = f[str(key)]
                image_data = Image.fromarray(np.array(SG_Data['image']))
                audio_data = Image.fromarray(np.array(SG_Data['audio']))
                SG_Label = torch.from_numpy(np.array((self.Label[index])))
    
                if self.transform:
                    image_data = self.transform(image_data)
                    audio_data = self.transform(audio_data)
            else:
                SG_Data = f[key][()]
                SG_Data = np.array(SG_Data)
                SG_Label = torch.from_numpy(np.array((self.Label[index])))
            
                ES_Data = Image.fromarray(SG_Data)
                if self.transform:
                    ES_Data = self.transform(ES_Data)
     
        if self.is_multimodal:
            return (audio_data, image_data, SG_Label)
        else:
            return (ES_Data, SG_Label)