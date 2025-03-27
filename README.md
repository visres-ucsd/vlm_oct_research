# vlm_oct_research
Repository to maintain code for exploring the use of vision encoders from VLMs &amp; VLM pipelines to improve
the correctness in OCT image classification for glaucoma detection


Explaination of various files : 

1. binary_classification_clip.py : file that contains classes to build various classification models such as : clip, llava vision encoder & multimodal setup where retfound + llava vision encoder is used for binary classification of glaucoma.
2. constants.py : Most of the common variables such as file paths, loss weights, learning rate etc. used in model training are taken from this file.
3. data_loader.py : File that contains augmentation functions & data loading classes for both single + multiple model (eg RetFound + llava) cases.
4. focal_loss_imp.py : File that contains focal loss custom implementation for normal precision & automatic mixed precision cases.
5. llaVa_finetune.ipynb : Experimental notebook to finetune the entire llava (including the llm) using the new OCT descriptions dataset.
6. models_mae & models_vit : Files from retFound repository to build the retFound model for pre-training & fine tuning respectively.
7. rFmodel.py : File to build retFound model for our binary classification task.
8. script_to_viz_val_points.ipynb : Custom notebook to study internal latent representations of vision encoders after training for OCT classification in 2d space.
9. test_evaluate_single.py : File to run the test split & compute metrics for single modality retFound model.
10. test_evaluate_single_clip.py : File to run the test split & compute metrics for single modality clip or llava's vision encoder model.
11. test_retfound_clip.py : File to run the test split & compute metrics for the case of combining both llava vision encoder (clip) with retFound case.
12. train_clip.py : File to train clip model.
13. train_clip_multimodal.py : File to train llava vision encoder (clip) with retFound model.
14. train_retfound.py : File to train retFound model.
15. vlm_data_analysis.ipynb : Some basic EDA for the text descriptions dataset.

Another file data_csv_file contains the dataset & it can be found at : /tscc/projects/ps-visres-group/multi_modal_fundus_oct_project/oct_only_experiments/data_csv_file
