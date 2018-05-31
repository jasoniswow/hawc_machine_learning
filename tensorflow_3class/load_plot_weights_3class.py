import os
import sys
import argparse
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# execution start time
start_time = time.time()


print ( "Parse arguments ===============================================================================" )
parser = argparse.ArgumentParser(description='Pytorch training')

parser.add_argument('--no-augment', dest='augment', action='store_false',
                    help='whether to use standard augmentation (default: True)')

parser.add_argument('--bin', type=int, 
                    help='analysis bin')

parser.add_argument('--modelFile', type=str,
                    help='model text file')

parser.set_defaults(augment=True)
args = parser.parse_args()

if args.bin is None:
    print ("Please specify analysis bin !")
    sys.exit()

# create a directory containing all output images and trained model
out_directory_name = 'colorMaps'
if not os.path.exists(out_directory_name):
    os.makedirs(out_directory_name)



print ( "Define feature groups =========================================================================" )
# 40 features in total
features = ['nTankHit', 'SFCFChi2', 'planeChi2', 'coreFitUnc', 'zenithAngle', 'azimuthAngle', 'coreFiduScale', 'nHitSP10', 'nHitSP20', 'CxPE20', 'CxPE30', 'CxPE40', 'CxPE50', 'CxPE40SPTime', 'PINC', 'GamCoreAge', 'numPoints', 'scandelCore', 'numSum', 'scanedFrac', 'fixedFrac', 'avePE', 'nHit', 'mPFnHits', 'mPFnPlanes', 'mPFp1nAssign', 'fAnnulusCharge0', 'fAnnulusCharge1', 'fAnnulusCharge2', 'fAnnulusCharge3', 'fAnnulusCharge4', 'fAnnulusCharge5', 'fAnnulusCharge6', 'fAnnulusCharge7', 'fAnnulusCharge8', 'disMax', 'compactness', 'nHit10ratio', 'nHit20ratio', 'nHitRatio']



print ( "Load model(1-HL-NN) ===========================================================================" )
# the trained results loaded here should be based on 40-features, 1-hidden-with-21-neurons and 3-output-classes
# these are global variables
inum = 40
mnum = 21
onum = 3

def IsFloat(s):
    try: 
        float(s)
        return True
    except ValueError:
        return False

def getVariable(modelFile):
    var_list = [ [], [], [], [] ]
    # hard-coded line numbers for different cases
    if onum==3:
        start_line = [0, 240, 262, 266]
        end_line = [240, 261, 266, 267]
    if onum==4:
        start_line = [0, 240, 263, 267]
        end_line = [240, 262, 267, 268]
    for i, line in enumerate(modelFile):
        for v in range(4):
            if (i>=start_line[v] and i<end_line[v]):
                for word in line.split():
                    word.replace("[", " ")
                    word.replace("]", " ")
                    word.replace("[[", " ")
                    word.replace("]]", " ")
                    if (word=="[" or word=="]" or word=="[[" or word=="]]"):
                        continue
                    if (word[0]=="[" and word[1]=="["):
                        word = word[2:]
                    if (word[-1]=="]" and word[-2]=="]"):
                        word = word[:-2]
                    if (word[0]=="["):
                        word = word[1:]
                    if (word[-1]=="]"):
                        word = word[:-1]
                    if IsFloat(word):
                        var_list[v].append( float(word) )
    w0 = []
    for i in range(inum):
        w0.append([])
        for j in range(mnum):
            a = mnum*i + j
            w0[i].append(var_list[0][a])
    w1 = []
    for i in range(mnum):
        w1.append([])
        for j in range(onum):
            a = onum*i + j
            w1[i].append(var_list[1][a])
    b0 = var_list[2]
    b1 = var_list[3]
    return np.array(w0), np.array(w1), np.array(b0), np.array(b1)

modelFile = open("%s"%(args.modelFile), "r")
w0, w1, b0, b1 = getVariable(modelFile)
#print (w0, len(w0))
#print (w1, len(w1))
#print (b0, len(b0))
#print (b1, len(b1))



print ( "Calculation ===================================================================================" )
# calculation function
def calFunc(x,w0,w1,b0,b1):
    res0 = np.matmul(x, w0)
    res0 = np.add(res0, b0)
    # reLU
    for i in range(mnum):
        if res0[i]<0:
            res0[i] = 0
    res1 = np.matmul(res0, w1)
    res1 = np.add(res1, b1)
    # softmax
    exp_sum = 0
    for i in range(onum):
        res1[i] = np.exp(res1[i])
        exp_sum = exp_sum + res1[i]
    for i in range(onum):
        res1[i] = res1[i] / exp_sum
    return res0,res1

# get variables
def variables(w0,w1,b0,b1):
    # add b0 to each row of w0
    res0 = np.array( w0 + b0[None,:] )
    # apply reLU
    res0 = np.clip(res0,0,999)
    # add b1 to each row of w1
    res1 = np.array( w1 + b1[None,:] )
    # apply softmax
    exp_sum = 0
    for i in range(onum):
        res1[i] = np.exp(res1[i])
        exp_sum = exp_sum + res1[i]
    for i in range(onum):
        res1[i] = res1[i] / exp_sum
    return res0, res1

# res0 and res1 are the 2D array to be plotted
# res0 = 21*40, res1 = 21*3
res0, res1 = variables(w0,w1,b0,b1)
res0 = np.transpose(res0)


print ( "Plot ==========================================================================================" )

fig0 = plt.figure(0, figsize=(6,4), dpi=200, facecolor='white', edgecolor='red')
plt.imshow(res0)
plt.xlabel('neurons', fontsize=8)
plt.ylabel('features', fontsize=8)
plt.title("Mapping_0", fontsize=12)
matplotlib.rc('font', size=8)
plt.tick_params(labelsize=8)
plt.axis('off')
plt.show()
fig0.savefig('%s/colorPlot_bin_%d_0.png'%(out_directory_name,args.bin))

fig1 = plt.figure(1, figsize=(6,4), dpi=200, facecolor='white', edgecolor='red')
plt.imshow(res1)
plt.xlabel('neurons', fontsize=8)
plt.ylabel('features', fontsize=8)
plt.title("Mapping_1", fontsize=12)
matplotlib.rc('font', size=8)
plt.tick_params(labelsize=8)
plt.axis('off')
plt.show()
fig1.savefig('%s/colorPlot_bin_%d_1.png'%(out_directory_name,args.bin))




















