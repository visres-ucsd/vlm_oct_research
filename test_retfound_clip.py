# Basic Imports........
import os
import numpy as np
import pandas as pd
import pickle
from tqdm.auto import tqdm
import sys
from constants import *


# PyTorch related imports....
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import datasets, models, transforms
import time
from tempfile import TemporaryDirectory
import models_vit
import models_mae
from binary_classification_clip import *
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score
from data_loader import *
from focal_loss_imp import *
import pytorch_warmup as warmup
from sklearn.model_selection import KFold
import pandas as pd

# Loading model

w_path = "/tscc/nfs/home/vejoshi/vlm_comparison/ret_found_baseline_non_vlm/RETFound_MAE/experiments/no_augmentations_all_unfreeze_llava_clip_retFoundoct/multimodalmultimodal_shape_336_unfreeze_50_lr_0.0005_poolType_max.pt"

#oct_model_weights = torch.load("/tscc/projects/ps-visres-group/multi_modal_fundus_oct_project/oct_only_experiments/no_augmentations_adapter_unfreeze_llava_clip_retFoundoct/multimodalmultimodal_shape_336_unfreeze_50_lr_0.0005_poolType_max.pt",map_location=torch.device("cpu"))
oct_model_weights = torch.load(w_path,map_location=torch.device("cpu"))
oct_model = build_model(model_type = "multimodal", eval_config = "test")
print("Loading weights to built model...")
oct_model.load_state_dict(oct_model_weights, strict=False)
print("Loaded weights")
torch.cuda.empty_cache()
oct_model = oct_model.to(torch.float32).to("cuda")
oct_model.eval()


# Data Loading
# dataset loading (file path in constants.py file)
# testing is done in a different file....
all_data_df = pd.read_csv(dataset_csv)
# mapping labels to indices for training....(changing the label_mapping dict can change indices, present in the constants.py file)
all_data_df["new_labels"] = all_data_df["EyeDX_at_spectralisDate"].apply(lambda x : label_mapping[x])
train_df = all_data_df[all_data_df["Split"] == "train"]
valid_df = all_data_df[all_data_df["Split"] == "val"]
test_df  = all_data_df[all_data_df["Split"] == "test"]

# Training, Val, Test stats...
print("#"*30)
print("Total number of train images : ",len(train_df))
print("Total number of valid images : ",len(valid_df))

print("#"*30)
print("Label distribution :")
print("Training split   : Healthy : {} | Glaucoma : {}".format(len(train_df[train_df["new_labels"] == 1]) / len(train_df),
                                                               len(train_df[train_df["new_labels"] == 0]) / len(train_df)))
print("#"*30)
print("Validation split : Healthy : {} | Glaucoma : {}".format(len(valid_df[valid_df["new_labels"] == 1]) / len(valid_df),
                                                               len(valid_df[valid_df["new_labels"] == 0]) / len(valid_df)))
print("#"*30)
print("Testing split : Healthy : {} | Glaucoma : {}".format(len(test_df[test_df["new_labels"] == 1]) / len(test_df),
                                                               len(test_df[test_df["new_labels"] == 0]) / len(test_df)))
print("#"*30)

pre_proc_func = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])

test_data = GenerateDataset(image_files = test_df["OCTImagePath"],
                                  label_list  = test_df["new_labels"],
                                  img_res     = input_shape[0],
                                  augment     = False,
                                  transform = pre_proc_func,
                                  split_flag  = "test",
                                  multimodal_flag = True)

test_dataloader = DataLoader(test_data,
                              batch_size = 1,
                              shuffle=True,
                              num_workers=4)



# Inference loop
print("Testing loop.....")
# cleaning computation from before....
prob_values = []
pred_values = []
gt_values = []
cnt = 0
with torch.set_grad_enabled(False):
    for inputs_llava, inputs_oct, labels,_ in tqdm(test_dataloader, position = 0, leave = True):
        
        inputs_llava = inputs_llava.to(torch.float32).to(device)
        inputs_oct = inputs_oct.to(torch.float32).to(device)
        outputs = oct_model(llava_x  = inputs_llava,
                            ret_x    = inputs_oct)

        # normalising logits....
        proc_op = torch.sigmoid(outputs)
        preds = (proc_op > 0.5)*1
        preds = preds.squeeze(-1)

        prob_values.extend([i for i in proc_op.detach().cpu().numpy()])
        pred_values.extend([i for i in preds.detach().cpu().numpy()])
        gt_values.extend(labels)


gt_values = np.array(gt_values)
pred_values = np.array(pred_values)
prob_values = np.array(prob_values)

# with respect to the glaucoma class...
# precision :
oct_prec = precision_score(y_true = gt_values,
                              y_pred = pred_values,
                              average = "binary",
                              pos_label = 0,
                              zero_division = 0.0)
oct_recall = recall_score(y_true = gt_values,
                              y_pred = pred_values,
                              average = "binary",
                              pos_label = 0,
                              zero_division = 0.0)

oct_f1 = (2*(oct_prec*oct_recall))/(oct_prec + oct_recall)
oct_auc = roc_auc_score(y_true  = gt_values, y_score = prob_values)

print("OCT scores : ")
print("Precision : {:.3f} | Recall : {:.3f} | F1 Score : {:.3f} | AUC Score : {:.3f}".format(oct_prec, oct_recall, oct_f1, oct_auc))
print("#####################################################################")


