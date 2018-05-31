# this script is based on python3
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
from subprocess import check_output
from sklearn.model_selection import train_test_split

if (len(sys.argv)==1):
    print ("Please define the analysis bin !")
    sys.exit()



print ( "Load the HDF5 MC file in read only mode ==================================================" )
hdf_rawMC = pd.HDFStore('data_input/sweets_dec20_noise_MPF_allParticle_100.h5','r')
# store the HDF5 file in memory (this would cause an error if the file is too large)
df = hdf_rawMC['sweets_xcdf'] # sweets_xcdf is the DataFrame object name



print ( "Data quality cuts ========================================================================" )
df = df[df.nHit>25] # events with enough nHit
df = df[df.zenithAngle<1.05] # zenith angle is less than 60 degree
df = df[df.coreFiduScale<150] # events that are not too far away
df = df[df.coreFitStatus==0] # events with successful core fit
df = df[df.angleFitStatus==0] # events with successful angle fit
df = df[df.nChAvail>=700] # requiring enough working PMTs
df['nCh90'] = df['nChAvail']>0.9*df['nChTot'] # boolean values



print ( "Extract GamCore parameters from packed ===================================================" )
# the lambda function is to creat a mini temporary function
# add extra columns of numPoints, scandelCore and numSum based on GamCorePackInt, GamCorePackInt = numPoints*100000 + scandelCore*100 + numSum
df['numPoints'] = (df['GamCorePackInt']/100000).astype(int) # number of tanks used in the fit (>=5)
df['scandelCore'] = ( (df['GamCorePackInt']%100000)/100).astype(float) # distance between scanned core and SFCF core
df['numSum'] = ( (df['GamCorePackInt']%100000)%100 ).astype(int) # number of tanks outside r_test with 2.5PE or above
# add extra columns of scanedFrac, fixedFrac and avePE based on GamCoreChi2, GamCoreChi2 = scaned_frac*10000 + fixed_frac*100 + averagePE*2
df['scanedFrac'] = (df['GamCoreChi2']/10000).astype(int) # scan the whole array and find the max fraction of hitted tank in 30m
df['fixedFrac'] = ( (df['GamCoreChi2']%10000)/100).astype(int) # the fraction of hitted tank in 30m around the SFCF core
df['avePE'] = ( ( (df['GamCoreChi2']%10000)%100 ) / 2. + 0.5).astype(float) # the average PE of tank in 30m around the SFCF core



print ( "Add cross features =======================================================================" )
df['compactness'] = df['nHitSP20']/df['CxPE40']
df['compactness'] = df['compactness'].map(lambda x: 0 if ((x==float("inf")) | (x==-float("inf"))) else x)
df['compactness'] = df['compactness'].map(lambda x: 0 if (x!=x) else x)
#print ( "compactness:   %d NaNs" %(df["compactness"].isnull().sum()) )
df['nHit10ratio'] = df['nHitSP10']/df['nHit']
df['nHit10ratio'] = df['nHit10ratio'].map(lambda x: 0 if ((x==float("inf")) | (x==-float("inf"))) else x)
df['nHit10ratio'] = df['nHit10ratio'].map(lambda x: 0 if (x!=x) else x)
#print ( "nHit10ratio:   %d NaNs" %(df["nHit10ratio"].isnull().sum()) )
df['nHit20ratio'] = df['nHitSP20']/df['nHit']
df['nHit20ratio'] = df['nHit20ratio'].map(lambda x: 0 if ((x==float("inf")) | (x==-float("inf"))) else x)
df['nHit20ratio'] = df['nHit20ratio'].map(lambda x: 0 if (x!=x) else x)
#print ( "nHit20ratio:   %d NaNs" %(df["nHit20ratio"].isnull().sum()) )
df['nHitRatio'] = df['nHit']/df['mPFnHits']
df['nHitRatio'] = df['nHitRatio'].map(lambda x: 0 if ((x==float("inf")) | (x==-float("inf"))) else x)
df['nHitRatio'] = df['nHitRatio'].map(lambda x: 0 if (x!=x) else x)
#print ( "nHitRatio:   %d NaNs" %(df["nHitRatio"].isnull().sum()) )



print ( "DataFrame for analysis bin ===============================================================" )
nBin = int(sys.argv[1])
df_bin = {}
df_bin[nBin] = df[df['nCh90']] # require 90% PMT working
# old binning uses nHit instead of nHitSP20
#nhlow = [0.030, 0.044, 0.067, 0.105, 0.162, 0.247, 0.356, 0.485, 0.618, 0.740, 0.840]
#nhhigh = [0.044, 0.067, 0.105, 0.162, 0.247, 0.356, 0.485, 0.618, 0.740, 0.840, 1.010]
# new way to bin, the last is a bin of all high energy stuff
nhlow = [0.030, 0.050, 0.075, 0.100, 0.200, 0.300]
nhhigh = [0.050, 0.075, 0.100, 0.200, 0.300, 1.000]
#*************************************************************************************
#***************** operates on df_bin[nBin] instead of df from here ******************
df_bin[nBin] = df_bin[nBin][(df_bin[nBin]["nHitSP20"]>=nhlow[nBin]*df_bin[nBin]["nChAvail"]) & (df_bin[nBin]["nHitSP20"]<nhhigh[nBin]*df_bin[nBin]["nChAvail"])]
#*************************************************************************************


print ( "Define feature groups ====================================================================" )
# only do feature engineering on all training features (not features_plus)
# 40 features in total
features = ['nTankHit', 'SFCFChi2', 'planeChi2', 'coreFitUnc', 'zenithAngle', 'azimuthAngle', 'coreFiduScale', 'nHitSP10', 'nHitSP20', 'CxPE20', 'CxPE30', 'CxPE40', 'CxPE50', 'CxPE40SPTime', 'PINC', 'GamCoreAge', 'numPoints', 'scandelCore', 'numSum', 'scanedFrac', 'fixedFrac', 'avePE', 'nHit', 'mPFnHits', 'mPFnPlanes', 'mPFp1nAssign', 'fAnnulusCharge0', 'fAnnulusCharge1', 'fAnnulusCharge2', 'fAnnulusCharge3', 'fAnnulusCharge4', 'fAnnulusCharge5', 'fAnnulusCharge6', 'fAnnulusCharge7', 'fAnnulusCharge8', 'disMax', 'compactness', 'nHit10ratio', 'nHit20ratio', 'nHitRatio']

features_plus = features +  ['signal', 'pclass', 'TWgt']



print ( "Feature engineering ======================================================================" )
# align 1 percent outliers
for f in features:
    low_limit = df_bin[nBin][f].quantile(0.01)
    high_limit = df_bin[nBin][f].quantile(0.99)
    df_bin[nBin][f] = df_bin[nBin][f].clip( low_limit, high_limit )

temp_min = []
temp_max = []
for f in features:
    temp_min.append(min(df_bin[nBin][f].values))
    temp_max.append(max(df_bin[nBin][f].values))
print ("Min:", temp_min)
print ("Max:", temp_max)

# normalization
df_bin[nBin][features] = df_bin[nBin][features].apply(lambda x: (x - x.min()) / (x.max() - x.min())) # only normalize features, not output class
#df_bin[nBin][features] = df_bin[nBin][features].clip(0.0, 1.0) # (optional) set hard cut on features

# limit the events weights if it's too large
df_bin[nBin]['TWgt'] = df_bin[nBin]['TWgt'].map(lambda x: x if x<10000 else 10000).astype(float)



print ( "Calculate class labels ===================================================================" )
# add an extra column that indicates gamma (as 1) or hadron (as 0)
df_bin[nBin]['signal'] = df_bin[nBin]['corsikaParticleId'].map(lambda x: 1 if x==1 else 0).astype(int)
# add an extra column that indicates the good gamma(==0), bad gamma(==1), good hadron(==2), bad hadron(==3)
# to be good event, it requires delAngle<threshold 
#          goodG   badG   goodH   badH
# signal     1      1       0      0   
# pclass'    0      1       0      1
# pclass     0      1       2      3

# using quantile as threshold for delAngle to separate good and bad events for gamma and hadron
delangle_threshold_g = df_bin[nBin]['delAngle'][df_bin[nBin]['signal']==1].quantile(0.80)
delangle_threshold_h = df_bin[nBin]['delAngle'][df_bin[nBin]['signal']!=1].quantile(0.80)
print ("delAngle threshold G: ", delangle_threshold_g, "**********************")
print ("delAngle threshold H: ", delangle_threshold_h, "**********************")

df_bin[nBin]['pclass'] = int(9)
df_bin[nBin].loc[ (df_bin[nBin]['signal']==1) & (df_bin[nBin]['delAngle']<=delangle_threshold_g), 'pclass'] = int(0)
df_bin[nBin].loc[ (df_bin[nBin]['signal']==1) & (df_bin[nBin]['delAngle']>delangle_threshold_g), 'pclass'] = int(1)
df_bin[nBin].loc[ (df_bin[nBin]['signal']!=1) & (df_bin[nBin]['delAngle']<=delangle_threshold_h), 'pclass'] = int(2)
df_bin[nBin].loc[ (df_bin[nBin]['signal']!=1) & (df_bin[nBin]['delAngle']>delangle_threshold_h), 'pclass'] = int(3)

print ("Wrong values: ", df_bin[nBin].loc[ df_bin[nBin]['pclass']==9 ].head())
print ("Misindentified Gamma: ", df_bin[nBin].loc[ (df_bin[nBin]['signal']==1) & ( (df_bin[nBin]['pclass']==2) | (df_bin[nBin]['pclass']==3) ) ].head() )
print ("Misindentified Hadron: ", df_bin[nBin].loc[ (df_bin[nBin]['signal'] ==0) & ( (df_bin[nBin]['pclass']==0) | (df_bin[nBin]['pclass']==1) ) ].head() )
#df0 = df_bin[nBin].loc[df_bin[nBin]['pclass'] ==0]
#df1 = df_bin[nBin].loc[df_bin[nBin]['pclass'] ==1]
#df2 = df_bin[nBin].loc[df_bin[nBin]['pclass'] ==2]
#df3 = df_bin[nBin].loc[df_bin[nBin]['pclass'] ==3]
#print ( df0.head(30)['delAngle'] )
#print ('=============================')
#print ( df1.head(30)['delAngle'] )
#print ('=============================')
#print ( df2.head(30)['delAngle'] )
#print ('=============================')
#print ( df3.head(30)['delAngle'] )



print ( "Prepare the DataFrame for one bin ========================================================" )
# only keep the feature_plus fields
df_bin[nBin] = df_bin[nBin][features_plus] # select interested features only
print ("DataFrame(reduced): ", df_bin[nBin].describe() )
for f in features_plus:
    print ('Check value =====')
    print ( "%s :"%f )
    print ( "  Min value = %f"%(min(df_bin[nBin][f].values)) )
    print ( "  Max value = %f"%(max(df_bin[nBin][f].values)) )
    print ( "  NaN's: ",df_bin[nBin][f].isnull().sum() )
#print ( df_bin[nBin].info() )
#print ( df_bin[nBin].columns )
#print ( df_bin[nBin].get_dtype_counts() )
#print ( df_bin[nBin].describe() )
#print ( df_bin[nBin].head() )



print ( "Plot feature =============================================================================" )
for f in features:
    plt.clf()
    df_bin[nBin][f].hist(bins=100)
    plt.savefig('%s_%d.png'%(f,nBin))



print ( "Create dataset for training and testing ==================================================" )
data_train = {}
data_test = {}
testFraction = {}
testFraction[nBin] = 0.2 # the percentage of testing set
data_train[nBin], data_test[nBin] = train_test_split( df_bin[nBin], test_size=testFraction[nBin], random_state=42 ) # split data set
print ("Training data: ", data_train[nBin].head())
print ("Testing data: ", data_test[nBin].head())



print ( "Calculate weighted sum for each class ====================================================" )
# Calculation for GoodGamma, BadGamma, GoodHadron, BadHadron
weighted_sum_GG = {}
weighted_sum_BG = {}
weighted_sum_GH = {}
weighted_sum_BH = {}

weighted_sum_GG[nBin] = {}
weighted_sum_BG[nBin] = {}
weighted_sum_GH[nBin] = {}
weighted_sum_BH[nBin] = {}

weighted_sum_GG[nBin]['weighted_GG'] = df_bin[nBin]['TWgt'][df_bin[nBin]['pclass']==0].sum()
weighted_sum_BG[nBin]['weighted_BG'] = df_bin[nBin]['TWgt'][df_bin[nBin]['pclass']==1].sum()
weighted_sum_GH[nBin]['weighted_GH'] = df_bin[nBin]['TWgt'][df_bin[nBin]['pclass']==2].sum()
weighted_sum_BH[nBin]['weighted_BH'] = df_bin[nBin]['TWgt'][df_bin[nBin]['pclass']==3].sum()

weighted_sum_GG[nBin]['weighted_GG_train'] = data_train[nBin]['TWgt'][data_train[nBin]['pclass']==0].sum()
weighted_sum_BG[nBin]['weighted_BG_train'] = data_train[nBin]['TWgt'][data_train[nBin]['pclass']==1].sum()
weighted_sum_GH[nBin]['weighted_GH_train'] = data_train[nBin]['TWgt'][data_train[nBin]['pclass']==2].sum()
weighted_sum_BH[nBin]['weighted_BH_train'] = data_train[nBin]['TWgt'][data_train[nBin]['pclass']==3].sum()

weighted_sum_GG[nBin]['weighted_GG_test'] = data_test[nBin]['TWgt'][data_test[nBin]['pclass']==0].sum()
weighted_sum_BG[nBin]['weighted_BG_test'] = data_test[nBin]['TWgt'][data_test[nBin]['pclass']==1].sum()
weighted_sum_GH[nBin]['weighted_GH_test'] = data_test[nBin]['TWgt'][data_test[nBin]['pclass']==2].sum()
weighted_sum_BH[nBin]['weighted_BH_test'] = data_test[nBin]['TWgt'][data_test[nBin]['pclass']==3].sum()

print ( "weighted GG: ", weighted_sum_GG )
print ( "weighted BG: ", weighted_sum_BG )
print ( "weighted GH: ", weighted_sum_GH )
print ( "weighted BH: ", weighted_sum_BH )



print ( "Store the training and testing file in HDF5 format =======================================" )
hdf_data_train = pd.HDFStore('data_output/training_sweets_dec20_noise_MPF_allParticle_bin_%d_4class.h5'%(nBin))
hdf_data_train.put('training', data_train[nBin], format='table', data_columns=True)
hdf_data_train.close()
hdf_data_test = pd.HDFStore('data_output/testing_sweets_dec20_noise_MPF_allParticle_bin_%d_4class.h5'%(nBin))
hdf_data_test.put('testing', data_test[nBin], format='table', data_columns=True)
hdf_data_test.close()




'''
######################################################################################################
# some old code
print ( "Re-scale weights for gamma events ========================================================" )
gamma_number = {}
gamma_number[nBin] = {}
gamma_number[nBin]['allUnweighted'] = len(df_bin[nBin][df_bin[nBin]['signal']==1])
gamma_number[nBin]['allWeighted'] = df_bin[nBin]['TWgt'][df_bin[nBin]['signal']==1].sum()
gamma_number[nBin]['training'] = data_train[nBin]['TWgt'][data_train[nBin]['signal']==1].sum()
gamma_number[nBin]['testing'] = data_test[nBin]['TWgt'][data_test[nBin]['signal']==1].sum()
hadron_number = {}
hadron_number[nBin] = {}
hadron_number[nBin]['allUnweighted'] = len(df_bin[nBin][df_bin[nBin]['signal']==0])
hadron_number[nBin]['allWeighted'] = df_bin[nBin]['TWgt'][df_bin[nBin]['signal']==0].sum()
hadron_number[nBin]['training'] = data_train[nBin]['TWgt'][data_train[nBin]['signal']==0].sum()
hadron_number[nBin]['testing'] = data_test[nBin]['TWgt'][data_test[nBin]['signal']==0].sum()
weightScaleFactor_training = float(hadron_number[nBin]['training']) / float(gamma_number[nBin]['training'])
weightScaleFactor_testing = float(hadron_number[nBin]['testing']) / float(gamma_number[nBin]['testing'])
print ( "Gammas: ", gamma_number )
print ( "Hadrons: ", hadron_number )
print ( "The Scale Factor(training): ", weightScaleFactor_training )
print ( "The Scale Factor(testing): ", weightScaleFactor_testing )

#Scale up gamma events weight
data_train[nBin].loc[ data_train[nBin]['signal']==1, 'TWgt' ] =  data_train[nBin].loc[ data_train[nBin]['signal']==1, 'TWgt' ] * weightScaleFactor_training
data_train[nBin]['TWgt'] = data_train[nBin]['TWgt'].map(lambda x: x if x<10000 else 10000).astype(float) # TWgt has very large values for some low energy events
data_test[nBin].loc[ data_test[nBin]['signal']==1, 'TWgt' ] =  data_test[nBin].loc[ data_test[nBin]['signal']==1, 'TWgt' ] * weightScaleFactor_testing
data_test[nBin]['TWgt'] = data_test[nBin]['TWgt'].map(lambda x: x if x<10000 else 10000).astype(float) 

#Calculate the final sum of weighted event number 
gamma_number[nBin]['training'] = data_train[nBin]['TWgt'][data_train[nBin]['signal']==1].sum()
gamma_number[nBin]['testing'] = data_test[nBin]['TWgt'][data_test[nBin]['signal']==1].sum()
hadron_number[nBin]['training'] = data_train[nBin]['TWgt'][data_train[nBin]['signal']==0].sum()
hadron_number[nBin]['testing'] = data_test[nBin]['TWgt'][data_test[nBin]['signal']==0].sum()
weightScaleFactor_training = float(hadron_number[nBin]['training']) / float(gamma_number[nBin]['training'])
weightScaleFactor_testing = float(hadron_number[nBin]['testing']) / float(gamma_number[nBin]['testing'])
print ( "Gammas: ", gamma_number )
print ( "Hadrons: ", hadron_number )
######################################################################################################
'''
