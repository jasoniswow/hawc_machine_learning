HAWC Machine Learning

1. data_processing
convert HAWC XCD file to HDF5 file
feature engineering and generate training/testing dataset

2. tensorflow_3class and tensorflow_4class
use tensorflow for training.
3class means using data as hadron events and output 3 classes: good pointing gamma, bad pointing gamma and hadron.
4class means using MC hadron and output 4 classes: good/bad pointing gamma, good/bad pointing hadron.

3. label_xcd
use c++ code to add an extra column to xcd files as labeling of class based on trained model (loaded with xml file).

4. mc_label_check
check the labeled MC file.

5. pytorch and keras
use other packages/API for training




