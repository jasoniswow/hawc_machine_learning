# if you use IWgt, that would give the weight for one second for the whole sky, which can be compared to just on sub run data
# if you use TWgt, then the weight is for one source per transit, which can only be compared to one transit data

from __future__ import print_function
import numpy as np
import argparse
import os.path
import matplotlib.pyplot as plt
from subprocess import call

parser = argparse.ArgumentParser(description='plot of Monte Carlo gamma events (as signal)')
parser.add_argument("-i","--input", help="input MC xcd file")
parser.add_argument("-b","--fbin",type=int, help="fHit analysis bin (0-5)")
args = parser.parse_args()

from xcdf import XCDFFile
from ROOT import *

gROOT.SetBatch(kTRUE) # keep the shell quite

bin_num = int(args.fbin)

# store values for each bin --------------------
total_events = 0
true_0 = 0
true_1 = 0
true_2 = 0
true_3 = 0
truec_0 = [0, 0, 0, 0]
truec_1 = [0, 0, 0, 0]
truec_2 = [0, 0, 0, 0]
truec_3 = [0, 0, 0, 0]
label_0 = 0
label_1 = 0
label_2 = 0
label_3 = 0
labelt_0 = [0, 0, 0, 0]
labelt_1 = [0, 0, 0, 0]
labelt_2 = [0, 0, 0, 0]
labelt_3 = [0, 0, 0, 0]
true_label_0 = 0
true_label_1 = 0
true_label_2 = 0
true_label_3 = 0
precision0 = 0
precision1 = 0
precision2 = 0
precision3 = 0
recall0 = 0
recall1 = 0
recall2 = 0
recall3 = 0
preudoprecision = 0
rejection = 0


# temp values for calculation lists above --------
true_class = 9
true0 = true0_w= 0
true1 = true1_w= 0
true2 = true2_w= 0
true3 = true3_w= 0
label0 = label0_w = 0
label1 = label1_w = 0
label2 = label2_w = 0
label3 = label3_w = 0
true_label_0 = true_label_0_w = 0
true_label_1 = true_label_1_w = 0
true_label_2 = true_label_2_w = 0
true_label_3 = true_label_3_w = 0
true01_label0 = 0
true23_label0 = 0
true123_label0 = 0

# delAngle threshold (in radian) for separating good and bad events for 6 fHit bins
delAngle_threshold_g = np.array([ 0.0341, 0.0237, 0.0182, 0.0131, 0.0091, 0.0072 ])
delAngle_threshold_h = np.array([ 0.0536, 0.0376, 0.032,  0.0273, 0.0226, 0.016  ])

xcdf = XCDFFile(args.input)
# event loop -----------------------------------
for record in xcdf.fields("rec.eventID, rec.nChTot, rec.nChAvail, rec.nTankHit, rec.nHit, rec.nHitSP10, rec.nHitSP20, rec.zenithAngle, rec.azimuthAngle, rec.dec, rec.ra, rec.CxPE20, rec.CxPE30, rec.CxPE40, rec.CxPE50, rec.CxPE40SPTime, rec.PINC, rec.angleFitStatus, rec.coreFitStatus, rec.coreFiduScale, rec.coreX, rec.coreY, rec.coreFitUnc, rec.SFCFChi2, rec.planeChi2, rec.fAnnulusCharge0, rec.fAnnulusCharge1, rec.fAnnulusCharge2, rec.fAnnulusCharge3, rec.GamCoreAge, rec.GamCoreAmp, rec.GamCoreChi2, rec.GamCorePackInt, rec.mPFnHits, rec.mPFnPlanes, rec.mPFp0nAssign, rec.mPFp0Weight, rec.mPFp0toangleFit, rec.mPFp1nAssign, rec.mPFp1Weight, rec.mPFp1toangleFit, rec.disMax, mc.zenithAngle, mc.azimuthAngle, mc.corsikaParticleId, mc.delAngle, mc.delCore, mc.logEnergy, mc.coreX, mc.coreY, sweets.IWgt, sweets.TWgt, rec.logNNEnergy, mc.coreFiduScale, rec.classLabel"):
    eventID, nChTot, nChAvail, nTankHit, nHit, nHitSP10, nHitSP20, zenithAngle, azimuthAngle, dec, ra, CxPE20, CxPE30, CxPE40, CxPE50, CxPE40SPTime, PINC, angleFitStatus, coreFitStatus, coreFiduScale, coreX, coreY, coreFitUnc, SFCFChi2, planeChi2, fAnnulusCharge0, fAnnulusCharge1, fAnnulusCharge2, fAnnulusCharge3, GamCoreAge, GamCoreAmp, GamCoreChi2, GamCorePackInt, mPFnHits, mPFnPlanes, mPFp0nAssign, mPFp0Weight, mPFp0toangleFit, mPFp1nAssign, mPFp1Weight, mPFp1toangleFit, disMax, zenithAngle_true, azimuthAngle_true, corsikaParticleId, delAngle, delCore, logEnergy, coreX_true, coreY_true, IWgt, TWgt, logNNEnergy, true_coreFiduScale, classLabel = record
    
    nChAvail = float(nChAvail)
    nChTot = float(nChTot)
    angleFitStatus = float(angleFitStatus)
    coreFitStatus = float(coreFitStatus)
    coreFiduScale = float(coreFiduScale)
    zenithAngle = float(zenithAngle)
    nHit = float(nHit)
    nHitSP20 = float(nHitSP20)
    classLabel = int(classLabel)
    TWgt = float(TWgt)

    if (TWgt>10000):
        TWgt = 10000

    # new way with fHit
    nfracs = [ (nChAvail>0.9*nChTot)and(nHitSP20>0.030*nChAvail)and(nHitSP20<0.050*nChAvail), (nChAvail>0.9*nChTot)and(nHitSP20>0.050*nChAvail)and(nHitSP20<0.075*nChAvail), (nChAvail>0.9*nChTot)and(nHitSP20>0.075*nChAvail)and(nHitSP20<0.100*nChAvail), (nChAvail>0.9*nChTot)and(nHitSP20>0.100*nChAvail)and(nHitSP20<0.200*nChAvail), (nChAvail>0.9*nChTot)and(nHitSP20>0.200*nChAvail)and(nHitSP20<0.300*nChAvail), (nChAvail>0.9*nChTot)and(nHitSP20>0.300*nChAvail)and(nHitSP20<1.000*nChAvail),]

    # quality and bin cuts
    if ( nfracs[bin_num] and nChAvail>=700 and nHit>25 and zenithAngle<1.05 and angleFitStatus==0 and coreFitStatus==0 and coreFiduScale<=150 ):

        total_events = total_events + TWgt

        # calculate true class
        if ( corsikaParticleId==1 and delAngle<=delAngle_threshold_g[bin_num] ):
            true_class = 0    
        if ( corsikaParticleId==1 and delAngle>delAngle_threshold_g[bin_num] ):
            true_class = 1
        if ( corsikaParticleId!=1 and delAngle<=delAngle_threshold_h[bin_num] ):
            true_class = 2
        if ( corsikaParticleId!=1 and delAngle>delAngle_threshold_h[bin_num] ):
            true_class = 3

        # calculate some values
        if (true_class==0):
            true0 = true0 + 1
            true0_w = true0_w + TWgt
            truec_0[classLabel] = truec_0[classLabel] + TWgt
        if (true_class==1):
            true1 = true1 + 1
            true1_w = true1_w + TWgt
            truec_1[classLabel] = truec_1[classLabel] + TWgt
        if (true_class==2):
            true2 = true2 + 1
            true2_w = true2_w + TWgt
            truec_2[classLabel] = truec_2[classLabel] + TWgt
        if (true_class==3):
            true3 = true3 + 1
            true3_w = true3_w + TWgt
            truec_3[classLabel] = truec_3[classLabel] + TWgt

        if (classLabel==0):
            label0 = label0 + 1
            label0_w = label0_w + TWgt
            labelt_0[true_class] = labelt_0[true_class] + TWgt
        if (classLabel==1):
            label1 = label1 + 1
            label1_w = label1_w + TWgt
            labelt_1[true_class] = labelt_1[true_class] + TWgt
        if (classLabel==2):
            label2 = label2 + 1
            label2_w = label2_w + TWgt
            labelt_2[true_class] = labelt_2[true_class] + TWgt
        if (classLabel==3):
            label3 = label3 + 1
            label3_w = label3_w + TWgt
            labelt_3[true_class] = labelt_3[true_class] + TWgt

        if (true_class==classLabel==0):
            true_label_0 = true_label_0 + 1
            true_label_0_w = true_label_0_w + TWgt
        if (true_class==classLabel==1):
            true_label_1 = true_label_1 + 1
            true_label_1_w = true_label_1_w + TWgt
        if (true_class==classLabel==2):
            true_label_2 = true_label_2 + 1
            true_label_2_w = true_label_2_w + TWgt
        if (true_class==classLabel==3):
            true_label_3 = true_label_3 + 1
            true_label_3_w = true_label_3_w + TWgt

        # always use weights for below
        if ( (true_class==0 or true_class==1) and classLabel==0 ):
            true01_label0 = true01_label0 + TWgt

        if ( (true_class==2 or true_class==3) and classLabel==0 ):
            true23_label0 = true23_label0 + TWgt

        if ( (true_class==1 or true_class==2 or true_class==3) and classLabel==0 ):
            true123_label0 = true123_label0 + TWgt


# fill the list for ------------------------
'''
true_0 = true0
true_1 = true1
true_2 = true2
true_3 = true3
label_0 = label0
label_1 = label1
label_2 = label2
label_3 = label3
correctpred_0 = true_label_0
correctpred_1 = true_label_1
correctpred_2 = true_label_2
correctpred_3 = true_label_3
'''
true_0 = true0_w
true_1 = true1_w
true_2 = true2_w
true_3 = true3_w
label_0 = label0_w
label_1 = label1_w
label_2 = label2_w
label_3 = label3_w
correctpred_0 = true_label_0_w
correctpred_1 = true_label_1_w
correctpred_2 = true_label_2_w
correctpred_3 = true_label_3_w


# precision = true_label / all label
if label_0==0:
    precision0 = 0
else:
    precision0 = float(correctpred_0) / label_0
if label_1==0:
    precision1 = 0
else:
    precision1 = float(correctpred_1) / label_1
if label_2==0:
    precision2 = 0
else:
    precision2 = float(correctpred_2) / label_2
if label_3==0:
    precision3 = 0
else:
    precision3 = float(correctpred_3) / label_3

# recall = true_label / all true
if true_0==0:
    recall0 = 0
else:
    recall0 = float(correctpred_0) / true_0
if true_1==0:
    recall1 = 0
else:
    recall1 = float(correctpred_1) / true_1
if true_2==0:
    recall2 = 0
else:
    recall2 = float(correctpred_2) / true_2
if true_3==0:
    recall3 = 0
else:
    recall3 = float(correctpred_3) / true_3

# pseudoprecision = (true0/1 labeled 0) / sqrt(true2/3 labeled 0)
if true23_label0==0:
    preudoprecision = 0
else:
    preudoprecision = float(true01_label0) / sqrt(true23_label0)

# rejection = 1- (true1/2/3 labeled 0) / (all true1/2/3)
if (true_1 + true_2 + true_3)==0:
    rejection = 0
else:
    rejection = 1 - float(true123_label0) / (true1_w + true2_w + true3_w)

print ("**********************")
print ("Bin %d -------------------------------------------------------"%(bin_num))
print ("Total events: ", total_events)
print ("True: ", true_0, "|", true_1, "|", true_2, "|", true_3)
print ("True_0 labeled: ", truec_0)
print ("True_1 labeled: ", truec_1)
print ("True_2 labeled: ", truec_2)
print ("True_3 labeled: ", truec_3)
print ("Labeled: ", label_0, "|", label_1, "|", label_2, "|", label_3)
print ("Labeled_0 true: ", labelt_0)
print ("Labeled_1 true: ", labelt_1)
print ("Labeled_2 true: ", labelt_2)
print ("Labeled_3 true: ", labelt_3)
print ("Precision: ", precision0, "|", precision1, "|", precision2, "|", precision3)
print ("Recall: ", recall0, "|", recall1, "|", recall2, "|", recall3)
print ("preudoPrecison:", preudoprecision)
print ("Rejection: ", rejection)
print ("**********************")

'''
plt.hist(truec_1, bins=100, normed=False, color='blue',label='hist')
#plt.xlim([-1,10])
plt.savefig('test.png')
plt.show()
'''
