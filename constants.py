# file to store all code constants for training

# file paths....
modality_type = "oct"

project_dir = "/tscc/nfs/home/vejoshi/vlm_comparison/ret_found_baseline_non_vlm/RETFound_MAE/"
rFound_var  = project_dir + "pre_trained_weights/"

r_found_oct_og_weights = rFound_var + "RETFound_oct_weights.pth"
r_found_fun_og_weights = rFound_var + "RETFound_cfp_weights.pth"

ret_found_ext_og_weights = r_found_fun_og_weights if modality_type == "fundus" else r_found_oct_og_weights

dataset_csv = project_dir + "data_csv_file/V3_Spectralis_GON_GVFD_COMB_HEALTHY.csv"

# training constants
use_aug = False
training_nature = "supervised_only"
model_name = "llava"
input_shape = (336,336,3)
unfreeze_perc = 0.0
# lr schedule taken from retFound paper....
last_lr = 1e-06
learning_rate = 5e-04
dropout = 0.1
warmup_epochs = 10
focal_weight = 2.8
l2_reg = 1e-05
# avg or CLS type for the final feature computed from the base Retfound encoder...
pool_type = "max"
dense_1 = 8
dense_2 = 12
dense_3 = 24
batch_size = 8
decision_threshold = 0.5 # used by metrics
train_epochs = 100
num_train_samples_viz = 4
patience = 10
reduce_lr_patience = 3
lr_scale = 0.1
lab_smooth = 0.13
aug_prob = 0.6

# Label constants...
"""
label_mapping = {"healthy"  : [1,0,0],
                 "suspects" : [0,1,0],
                 "glaucoma" : [0,0,1]}
"""
label_mapping = {"GVFD" : 0,
                 "GVFD & GON" : 0,
                 "Healthy" : 1}

num_classes = 2 #len(set(label_mapping.values())

# Save directory ##########################################################
# Creating directory to save runs & best weights.....
save_dir_name = "./experiments/no_augmentations_all_freeze_final_layer_unfreeze_llava_max_" + modality_type + "/"
model_save_name = model_name + "_shape_" + str(input_shape[0]) + "_freeze_" + \
str(int(unfreeze_perc*100)) + "_lr_" + str(learning_rate) + "_poolType_" + str(pool_type)
