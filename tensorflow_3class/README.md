Training:
Use "tf_dnn_multilabel.py" for training. The default configuration is normally good.
Then checkout training results in directory like "Bin_*_features_*_weight_*_HL_*_NF_*_LR_*_epochs_*"


Training scripts:
"test_dnn_multilabel.py" is the working script for training.
"tf_dnn_multilabel.py" is a tested script for training.
All trained models will be saved in TF model checkpoint, text and xml.


Load TF model and resume trianing:
"load_resume_training.py" is a script that load TF checkpoints/models and resume training.
"load_model_from_text.py" is a script that load trained models from text files and do tests.
"load_plot_weights.py" is a script to load model from txt file and plot.


Evaluate models:
1. test different class weights with all features and one common NN architecture.
2. test different NN architecture (hidden layers and neuron numbers) with all features.
3. test different features groups.


Saved stuff:
1. TF model files: checkpoint, mkdel.ckpt.meta ...
2. trained weights in txt/xml format: mkdel_bin_*_hl_*.txt/xml
3. indicators like loss/accuracy/...: indicators_bub_*_hl_*.txt
4. performance images.





