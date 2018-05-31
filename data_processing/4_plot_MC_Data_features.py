# this script is based on python3
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt


if (len(sys.argv)==1):
    print ("Please define the analysis bin !")
    sys.exit()
nBin = int(sys.argv[1])

out_directory_name = 'plot_bin_' + str(nBin) 
if not os.path.exists(out_directory_name):
    os.makedirs(out_directory_name)



print ( "Load the HDF5 Gamma file and Data off-source =============================================" )
hdf_rawMC = pd.HDFStore('data_input/sweets_dec20_noise_MPF_allParticle_100.h5','r')
hdf_rawDA = pd.HDFStore('data_input/data_hadron_combined.h5','r')

# store the HDF5 file in memory (this would cause an error if the file is too large)
df_MC = hdf_rawMC['sweets_xcdf'] # sweets_xcdf is the MC Gamma DataFrame object name
df_DA = hdf_rawDA['data_hadron'] # data_hadron is the Data off-source DataFrame object name



print ( "Assign label and weight ==================================================================" )
# add an extra column that indicates gamma (as 1) or hadron (as 0)
df_MC['signal'] = df_MC['corsikaParticleId'].map(lambda x: 1 if x==1 else 0).astype(int)
df_DA['signal'] = int(2)
# limit the events weights if it's too large
df_MC['TWgt'] = df_MC['TWgt'].map(lambda x: x if x<10000 else 10000).astype(float)
df_DA['TWgt'] = float(1)
# combine MC gamma and data hadron
# MC Gamma signal=1, MC Hadron signal=0, data signal=2
df = df_MC.append(df_DA, ignore_index=True)



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

# normalization
df_bin[nBin][features] = df_bin[nBin][features].apply(lambda x: (x - x.min()) / (x.max() - x.min())) # only normalize features, not output class
#df_bin[nBin][features] = df_bin[nBin][features].clip(0.0, 1.0) # (optional) set hard cut on features



#print ( "Prepare the DataFrame for one bin ========================================================" )
# only keep the feature_plus fields
#df_bin[nBin] = df_bin[nBin][features_plus] # select interested features only



print ( "Get all set  =============================================================================" )

# multiply features values with TWgt
for f in features:
    df_bin[nBin][f] = df_bin[nBin][f] * df_bin[nBin]['TWgt']

# get array for each type
array_MC_G = np.array( df_bin[nBin][ (df_bin[nBin]["signal"]==1) ] )
array_MC_H = np.array( df_bin[nBin][ (df_bin[nBin]["signal"]==0) ] )
array_Data = np.array( df_bin[nBin][ (df_bin[nBin]["signal"]==2) ] )
print ("=============================")
print (len(array_MC_G))
print ("---------")
print (len(array_MC_H))
print ("---------")
print (len(array_Data))

# normalization
#MC_G_plot = array_MC_G / np.linalg.norm(array_MC_G)
#MC_H_plot = array_MC_H / np.linalg.norm(array_MC_H)
#Data_plot = array_Data / np.linalg.norm(array_Data)
#print ("=============================")
#print (len(MC_G_plot))
#print ("---------")
#print (len(MC_H_plot))
#print ("---------")
#print (len(Data_plot))



print ( "Plot feature =============================================================================" )
counter = 0
fig = []

for f in features:

    fig = plt.figure(1, figsize=(12,8), dpi=200, facecolor='red', edgecolor='blue')
    plt.clf()

    plt.hist(array_MC_G, 100, normed=1, facecolor='red', alpha=1.0, label='MC G')
    plt.hist(array_MC_H, 100, normed=1, facecolor='green', alpha=1.0, label='MC H')
    plt.hist(array_Data, 100, normed=1, facecolor='blue', alpha=1.0, label='Data')

    #plt.ylim([0,50])
    plt.title("%s"%(f), fontsize=30)
    matplotlib.rc('font', size=18)
    plt.tick_params(labelsize=20)
    plt.legend(loc=1)
    fig.savefig("%s/%s.png"%(out_directory_name,f))

    counter = counter + 1



