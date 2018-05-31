# this script is based on python3
import numpy as np
import pandas as pd



# load the HDF5 MC file in read only mode -----------------------------------------------------------------
hdf_train = pd.HDFStore('data/training_sweets_dec19_noise_MPF_allParticle_bin_0.h5','r')

# store the HDF5 file in memory (this would cause an error if the file is too large)
df_train = hdf_train["training"]



# print info------------------------------------------------------------------------------------------------
print ("The basic Info: ", df_train.info() )
#print ("The columns: ", df_train.columns )
#print ("The counts: ",  df_train.get_dtype_counts() )
#print ("The head: ", df_train.head() )
#print ("The tail: ", df_train.tail() )
print ("The statistic: ", df_train.describe() )


