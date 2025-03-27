# Basic Imports
from copy import copy
import datetime
from glob import glob
import json
import math
import multiprocessing
import os
from pathlib import Path
import random
import urllib.request
import numpy as np

# PyTorch related imports....
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import torchvision
from transformers import CLIPModel
from transformers import AutoProcessor, LlavaForConditionalGeneration, LlavaConfig, CLIPVisionConfig, LlamaConfig


# retfound specific import.....
import models_vit

# Impoving code reproducability...
seed = 100
np.random.seed(seed)
torch.manual_seed(seed)
cudnn.benchmark = True

# setting up gpu details
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
r_found_oct_og_weights = "/tscc/nfs/home/vejoshi/oct_fundus_project/ret_found_exp/RETFound_MAE/pre_trained_weights/" + "RETFound_oct_weights.pth"

class classificationModel(nn.Module):
    def __init__(self,
                 base_model):
        super().__init__()
        # instantiate model...(a lot of training control variables are specified in the constants.py file)
        self.base_model = base_model
        self.layer_1 = nn.Linear(1024, 1) 
        self.activation_func = nn.Sigmoid()

    def forward(self, x):
        # using CLIP pooled output....
        x = self.base_model(x)[1]
        x = self.layer_1(x)
        x = self.activation_func(x)
        return x


class llavaClassificationModel(nn.Module):
    def __init__(self,
                 base_model):

        super().__init__()
    
        self.base_model = base_model.vision_tower  # Use the vision encoder from the Llava model
        self.mlp        = base_model.multi_modal_projector  # Use the multi-modal projector
        self.pooler     = nn.AdaptiveAvgPool1d(1)  # Pooling to aggregate patch features
        self.layer_1    = nn.Linear(5120, 1)
        self.activation_func = nn.Sigmoid()
    def forward(self, x):
        vision_output = self.base_model(pixel_values=x)  # Forward pass through the vision encoder       
        vision_features = vision_output[0]  # Assuming the pooled output is the first element in the tuple
        mlp_features = self.mlp(vision_features)  # Forward pass through the MLP
        pooled_output = self.pooler(mlp_features.permute(0, 2, 1)).squeeze(-1)  # Global average pooling
        logits = self.layer_1(pooled_output)
        x = self.activation_func(logits)
        return x

class multiModal(nn.Module):
    def __init__(self,
                 llava_model,
                 retFound_model):
        
        super().__init__()
        self.llava_model = llava_model
        self.retFound_model = retFound_model
        
        self.llava_base_model = llava_model.vision_tower  # Use the vision encoder from the Llava model
        self.llava_mlp        = llava_model.multi_modal_projector  # Use the multi-modal projector
        self.llava_pooler     = nn.AdaptiveAvgPool1d(1)  # Pooling to aggregate patch features
        self.project_llava    = nn.Linear(4096,768)
        self.project_oct      = nn.Linear(1024,768)
        self.int_activ_func   = nn.ReLU()
        self.layer_2          = nn.Linear(1536, 1)
        self.activation_func = nn.Sigmoid()

    def forward(self, llava_x, ret_x):
        
        # LLava inference
        llava_vision_output = self.llava_base_model(pixel_values = llava_x)[0]  # Forward pass through the vision encoder   
        llava_proj = self.llava_mlp(llava_vision_output)
        pooled_output = self.llava_pooler(llava_proj.permute(0, 2, 1)).squeeze(-1)
        llava_proj  = self.project_llava(pooled_output)
        llava_activ = self.int_activ_func(llava_proj)

        # RetFound inference
        oct_op = self.retFound_model.forward_features(ret_x)
        reduced_op = self.project_oct(oct_op)
        oct_red = self.int_activ_func(reduced_op)
        
        # Concat op
        x_cat  = torch.cat((llava_activ, oct_red), dim = 1)
        op = self.layer_2(x_cat)

        # No activation sigmoid due to numerical issues in amp training...
        return op

# Using only pre-trained models for our experiments...
# Add the model as the experiments progress....
# Using heavy models for best classification accuracy....
def build_model(model_type = "clip", eval_config = "train"):
    final_model = None
    if model_type == "clip":
        clip_model_name = "openai/clip-vit-large-patch14-336"
        model = CLIPModel.from_pretrained(clip_model_name).vision_model
        final_model = classificationModel(model)

    elif model_type == "llava":
        model_id = "llava-hf/llava-1.5-13b-hf"
        model = LlavaForConditionalGeneration.from_pretrained(model_id, torch_dtype = torch.bfloat16, low_cpu_mem_usage=True)
        #print(model)
        final_model = llavaClassificationModel(model)
    
    elif model_type == "multimodal":

        if eval_config == "train":
            # Loading LLaVA model
            model_id = "llava-hf/llava-1.5-7b-hf"
            llava_model = LlavaForConditionalGeneration.from_pretrained(model_id, torch_dtype = torch.bfloat16, low_cpu_mem_usage=True)
            llava_model.to(torch.float32)

            # Loading RetFound model
            model_oct   = models_vit.__dict__["vit_large_patch16"](img_size    = (224,224,3),
                                                               num_classes = 2,
                                                               drop_path_rate = 0.1,
                                                               global_pool=False)
        
            checkpoint_oct = torch.load(r_found_oct_og_weights, map_location = device)
            msg_o = model_oct.load_state_dict(checkpoint_oct['model'], strict=False)

            # Creating final multimodal pipeline....
            final_model = multiModal(llava_model = llava_model,
                                 retFound_model = model_oct)
        else:

            #vision_config = CLIPVisionConfig()

            # Initializing a Llama config
            #text_config = LlamaConfig()

            # Initializing a Llava llava-1.5-7b style configuration
            #configuration = LlavaConfig(vision_config, text_config)

            # Initializing a model from the llava-1.5-7b style configuration
            #vision_config = CLIPVisionConfig()
            
            #configuration = LlavaConfig(vision_config, text_config)
            #llava_model = LlavaForConditionalGeneration(configuration)
            
            model_id = "llava-hf/llava-1.5-7b-hf"
            llava_model = LlavaForConditionalGeneration.from_pretrained(model_id, torch_dtype = torch.bfloat16, low_cpu_mem_usage=True)

            model_oct   = models_vit.__dict__["vit_large_patch16"](img_size    = (224,224,3),
                                                               num_classes = 2,
                                                               drop_path_rate = 0.1,
                                                               global_pool=False)

            # Creating final multimodal pipeline....
            final_model = multiModal(llava_model = llava_model,
                                 retFound_model = model_oct)



    return final_model

"""
if __name__ == "__main__":
    model = build_model(model_type = "multimodal")
    print(model)
    total_cnt = 0
    trainable_cnt = 0
    for nm, param in model.named_parameters():
        
        print(nm)
        #if total_cnt < limit:
        #    param.required_grad = False
        #else:
        #    running the linear evaluation protocol
        #if "layer_1" in nm:
        #    param.requires_grad = True
        #    trainable_cnt+=1
        #else:
        #    param.required_grad = False

        #total_cnt+=1

    print("#"*30)
    #print("Percentage of trainabile parameters : {:.2f}".format((trainable_cnt / total_cnt)*100))
"""
