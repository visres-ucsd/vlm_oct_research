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





oct_model_weights = torch.load("/tscc/nfs/home/vejoshi/vlm_comparison/ret_found_baseline_non_vlm/RETFound_MAE/experiments/no_augmentations_all_unfreeze_llava_clip_oct/llava_CLIPllava_CLIP_shape_336_unfreeze_50_lr_0.0005_poolType_max.pt",map_location = "cuda")
oct_model = build_model(model_type = "llava")
oct_model.load_state_dict(oct_model_weights, strict=False)
oct_model = oct_model.to(torch.float32).to("cuda")
oct_model.eval()


pre_proc_func = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])



# dataset loading (file path in constants.py file)
# testing is done in a different file....
all_data_df = pd.read_csv(dataset_csv)
# mapping labels to indices for training....
all_data_df["new_labels"] = all_data_df["EyeDX_at_spectralisDate"].apply(lambda x : label_mapping[x])
train_df = all_data_df[all_data_df["Split"] == "train"]
valid_df = all_data_df[all_data_df["Split"] == "val"]
test_df  = all_data_df[all_data_df["Split"] == "test"]

print("#"*30)

print("Total number of train images : ",len(train_df))
print("Total number of valid images : ",len(valid_df))
print("Total number of test  images : ",len(test_df))

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



testing_data_oct = TestGenerateDataset(image_files = test_df["OCTImagePath"],
                                       labels_list = test_df["new_labels"],
                                       img_res = 336,
                                       transform = pre_proc_func)

test_dataloader_oct = DataLoader(testing_data_oct,
                                  batch_size = 1,
                                  shuffle=False,
                                  num_workers=4)


print("Testing loop.....")
# cleaning computation from before....
prob_values = []
pred_values = []
gt_values = []
cnt = 0
with torch.set_grad_enabled(False):
    for inputs, labels_index, file_path in tqdm(test_dataloader_oct, position=0, leave=True):
        inputs = inputs.to("cuda")
        preds  = oct_model(inputs)
        pred_label  = (preds > 0.5)*1
        prob_values.extend([i[0] for i in preds.detach().cpu().numpy()])
        pred_values.extend([i[0] for i in pred_label.detach().cpu().numpy()])
        gt_values.extend(labels_index)


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

