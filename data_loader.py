# Data loader centric imports....
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from constants import *
import PIL
import numpy as np


class H_random_translate(object):
    """
    Class to perform random Horizontal translation with image wraping....

    Args:
        perc_motion : (float) maximum % of image width to move left or right (sampled randomly)
    """

    def __init__(self, perc_motion = 0.1, h_val = 224, w_val = 224):
        self.perc_motion = perc_motion
        self.h_val = h_val
        self.w_val = w_val

    def __call__(self, sample):

        # performing wraping...
        total_width = sample.size[1]*3
        max_height  = sample.size[0]
        new_im = PIL.Image.new('RGB', (total_width, max_height))

        # concating the same image on right & left....
        x_offset = 0
        for i in range(3):
            new_im.paste(sample, (x_offset,0))
            x_offset+= sample.size[1]

        # random ranges for horizontal translation 
        motion_limit = int(sample.size[1]*self.perc_motion)
        crop_coord = torch.randint(-1*motion_limit, motion_limit, (1,)).numpy()[0]
    
        # wrapping cropping...
        proc_img = transforms.functional.crop(new_im, top = 0, left = sample.size[1] + crop_coord, height = self.h_val , width = self.w_val) 
        
        return proc_img

# data loader helper class
class GenerateDataset(Dataset):
    def __init__(self,
                 image_files,
                 label_list,
                 img_res = 224,
                 augment = False,
                 apply_random_prob = 0.2,
                 transform = None,
                 split_flag = "training",
                 multimodal_flag = False):


        # loading image paths & labels....
        self.image_list = np.array(image_files)
        
        # binary labels
        self.labels = np.array(label_list)        

        print("Number of images loaded for {} split : {} images".format(split_flag, len(self.image_list)))
        print("Number of labels loaded for {} split : {} labels".format(split_flag, len(self.labels)))

        # model specific pre-processing function.....
        self.transform = transform

        # other training constants.....
        self.multimodal_flag = multimodal_flag
        self.img_res = (img_res, img_res)
        self.augment = augment
        self.apply_random_prob = apply_random_prob
        self.batch_size = batch_size
        self.total_size = len(self.image_list)

        # inverse of class proportion serves as the class weights 
        # more frequent the class is, less is the associated class weight....
        self.class_weights = {0 : (1/len(self.labels[self.labels[:] == 0]))*(self.total_size/ num_classes),
                              1 : (1/len(self.labels[self.labels[:] == 1]))*(self.total_size/ num_classes)}

        print("Class weights are : ")
        for ct,i in enumerate(["Glaucoma", "Healthy"]):
            freq = len(self.labels[self.labels[:] == ct])
            print("Class Name : {} | Frequency : {} | Weight : {}".format(i, freq, self.class_weights[ct]))


    def __len__(self):
        return self.total_size

    def __getitem__(self, idx):
        
        class_label = self.labels[idx]
        # Randomly apply augmentations
        aug_transforms = None
        if self.augment:
            aug_transforms = transforms.Compose([H_random_translate(perc_motion = 0.25, h_val = input_shape[0], w_val = input_shape[1]),
                                                 transforms.RandomAffine(degrees = (-5,5),
                                                                         translate = (0.0,0.1),
                                                                         interpolation=transforms.InterpolationMode.BILINEAR),
                                                 transforms.ColorJitter(brightness = 0.3)])
            
            aug_transforms_2 = transforms.Compose([H_random_translate(perc_motion = 0.25, h_val = 224, w_val = 224),
                                                 transforms.RandomAffine(degrees = (-5,5),
                                                                         translate = (0.0,0.1),
                                                                         interpolation=transforms.InterpolationMode.BILINEAR),
                                                 transforms.ColorJitter(brightness = 0.3)])

        if self.multimodal_flag:
            llava_img = PIL.Image.open(self.image_list[idx]).convert('RGB').resize(self.img_res)
            ret_img = PIL.Image.open(self.image_list[idx]).convert('RGB').resize((224,224))
            

            llava_proc_img = llava_img
            ret_proc_img = ret_img
            if aug_transforms is not None:
                toss = np.random.choice([0,1],p=[1-self.apply_random_prob, self.apply_random_prob])
                if toss == 1:
                    llava_proc_img = aug_transforms(llava_img)
                    ret_proc_img   = aug_transforms_2(ret_img)
            
            return self.transform(llava_proc_img), self.transform(ret_proc_img), class_label, self.image_list[idx]
        else:   
            # load image & label
            img_inp     = PIL.Image.open(self.image_list[idx]).convert('RGB').resize(self.img_res)


     
            # randomness to apply all augmentations....
            proc_img = img_inp
            if aug_transforms is not None:
                toss = np.random.choice([0,1],p=[1-self.apply_random_prob, self.apply_random_prob])
                if toss == 1:
                    proc_img = aug_transforms(img_inp)
        
            # one hot encoded labels... 
            # self transform is model specific compulsory pre-processing
            return self.transform(proc_img), class_label, self.image_list[idx]


# To update when exploring the testing CODE>>>>>>>>>
class TestGenerateDataset(Dataset):
    def __init__(self,
                 image_files,
                 labels_list,
                 img_res = 224,
                 transform = None,
                 multimodal_flag = False):


        # loading image paths & labels....
        self.image_list = np.array(image_files)
        
        # binary labels
        self.labels = np.array(labels_list)

       # model specific pre-processing function.....
        self.transform = transform

        # other training constants.....
        self.img_res = (img_res, img_res)
        self.batch_size = batch_size
        self.total_size = len(self.image_list)
        self.multimodal_flag = multimodal_flag


    def __len__(self):
        return self.total_size

    def __getitem__(self, idx):
        class_label = self.labels[idx]

        if self.multimodal_flag:
            llava_img = PIL.Image.open(self.image_list[idx]).convert('RGB').resize(self.img_res)
            ret_img = PIL.Image.open(self.image_list[idx]).convert('RGB').resize((224,224))
            return self.transform(llava_img), self.transform(ret_img) ,class_label, self.image_list[idx]

        else:
            # fetching image & label....
            img_inp = PIL.Image.open(self.image_list[idx]).convert('RGB').resize(self.img_res)

        
            return self.transform(img_inp), class_label, self.image_list[idx]
