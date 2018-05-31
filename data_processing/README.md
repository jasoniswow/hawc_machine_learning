
1. "1_xcdf_to_hdf5_Data.py" and "1_xcdf_to_hdf5_MC.py" are the script to convert data/MC xcdf file to hdf5 file. You need to activate "miniconda" to load the XCDF module.

2. "2_data_generation_Data_Hadron_3class.py" and "2_data_generation_MC_Hadron_4class.py" are the script to do feature engineering and generate the training/testing data set. The "Data_Hadron_3class" means using real data as hadron events for training and output 3 classes (good pointing gamma, bad pointing gamma, hadron). "MC_Hadron_4class" means using MC hadron events for training and output 4 classes (good pointing gamma, bad pointing gamma, good pointing hadron, bad pointing hadron). You need to activate "anaconda" to load all the python modules.

3. "3_check_hdf5.py" is the script to check the hdf5 file.

4. "4_plot_MC_Data_features.py" is the script to plot features.
