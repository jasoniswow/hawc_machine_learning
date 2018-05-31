import os
import sys
import argparse
import time
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
# execution start time
start_time = time.time()


print ( "Parse arguments ===============================================================================" )
parser = argparse.ArgumentParser(description='Pytorch training')

parser.add_argument('--no-augment', dest='augment', action='store_false',
                    help='whether to use standard augmentation (default: True)')
parser.add_argument('--name', type=str, default='Exp001',
                    help='name of experiment')

parser.add_argument('--fileDir', type=str, default='/home/zhixiang/machine_learning/data_processing/data_output',
                    help='directory of files')

parser.add_argument('--modelFile', type=str,
                    help='model text file')

parser.add_argument('--bin', type=int, 
                    help='analysis bin')

parser.add_argument('--featureGroup', type=int, default=9, 
                    help='9=All_Features(default), 0-6=minus_one_group, -1=4_Features')

parser.add_argument('--weightedEvents', type=int, default=1, 
                    help='0=no_eventWeights, 1=using_weighed(default)')

parser.add_argument('--weightedClasses', type=int, default=1, 
                    help='0=no_classWeights, 1=allEqual_weighed(default), 2=double_Gamma, 3=double_Hadron')

parser.add_argument('--trainType', type=int, default=1, 
                    help='0=mini-batch, 1=full-batch(default)')

parser.add_argument('--hiddenlayer', type=int, default=1,
                    help='number of hidden layers (default=1)')

parser.add_argument('--neuronfactor', type=float, default=1,
                    help='scale factor for neurons in hidden layers (default=1)')

parser.add_argument('--learningRate', type=float, default=0.001, 
                    help='initial learning rate (default=10e-3)')

parser.add_argument('--batchSize', type=int, default=1000, 
                    help='mini-batch size for debugging (default=1000)')

parser.add_argument('--epochs', type=int, default=10, 
                    help='number of total epochs (default=10)')

parser.add_argument('--displayStep', type=int, default=1,
                    help='terminal display step (default=1)')

parser.set_defaults(augment=True)
args = parser.parse_args()

dtype = tf.float32 # set data type to be float tensor (32 bits)

if args.bin is None:
    print ("Please specify analysis bin !")
    sys.exit()

# create a directory containing all output images and trained model
out_directory_name = 'load_model_results'
if not os.path.exists(out_directory_name):
    os.makedirs(out_directory_name)



print ( "Define feature groups =========================================================================" )
# 40 features in total
features_all = ['nTankHit', 'SFCFChi2', 'planeChi2', 'coreFitUnc', 'zenithAngle', 'azimuthAngle', 'coreFiduScale', 'nHitSP10', 'nHitSP20', 'CxPE20', 'CxPE30', 'CxPE40', 'CxPE50', 'CxPE40SPTime', 'PINC', 'GamCoreAge', 'numPoints', 'scandelCore', 'numSum', 'scanedFrac', 'fixedFrac', 'avePE', 'nHit', 'mPFnHits', 'mPFnPlanes', 'mPFp1nAssign', 'fAnnulusCharge0', 'fAnnulusCharge1', 'fAnnulusCharge2', 'fAnnulusCharge3', 'fAnnulusCharge4', 'fAnnulusCharge5', 'fAnnulusCharge6', 'fAnnulusCharge7', 'fAnnulusCharge8', 'disMax', 'compactness', 'nHit10ratio', 'nHit20ratio', 'nHitRatio']

# essential features were used previously
f_essential = ['PINC', 'compactness', 'nHit20ratio', 'CxPE40']
# 7 features groups (compactness & pincness not included) to be tested
# each time exclude one group and see how the loss (or other indicator) changes
f_group0 = ['coreFiduScale', 'CxPE20', 'CxPE30', 'CxPE40', 'CxPE50', 'CxPE40SPTime'] # core fit
f_group1 = ['nHitSP10', 'nHitSP20', 'nHit10ratio', 'nHit20ratio'] # plane fit
f_group2 = ['SFCFChi2', 'planeChi2', 'coreFitUnc'] # errors
f_group3 = ['nHit', 'mPFp1nAssign', 'mPFnPlanes'] # MPF
f_group4 = ['GamCoreAge', 'numPoints', 'scandelCore', 'numSum', 'scanedFrac', 'fixedFrac', 'avePE'] #GamCore
f_group5 = ['fAnnulusCharge0', 'fAnnulusCharge1', 'fAnnulusCharge2', 'fAnnulusCharge3', 'fAnnulusCharge4', 'fAnnulusCharge5', 'fAnnulusCharge6', 'fAnnulusCharge7', 'fAnnulusCharge8']
f_group6 = ['zenithAngle', 'azimuthAngle', 'mPFnHits', 'disMax',  'nHitRatio', 'nTankHit'] # others

if (args.featureGroup==9):
    features = features_all
elif (args.featureGroup==-1):
    features = f_essential
elif (args.featureGroup==0):
    features = list( set(features_all) - set(f_group0) )
elif (args.featureGroup==1):
    features = list( set(features_all) - set(f_group1) )
elif (args.featureGroup==2):
    features = list( set(features_all) - set(f_group2) )
elif (args.featureGroup==3):
    features = list( set(features_all) - set(f_group3) )
elif (args.featureGroup==4):
    features = list( set(features_all) - set(f_group4) )
elif (args.featureGroup==5):
    features = list( set(features_all) - set(f_group5) )
elif (args.featureGroup==6):
    features = list( set(features_all) - set(f_group6) )
else:
    print ("Please define feature group !")
    sys.exit()



print ( "Class weights =================================================================================" )
# The weighted sum of each class for each bin (fHitH is a bin with all high energy stuff)
#           GoodGamma       BadGamma        Hadron
# fHit0a    174.27493       48.293300       1377762  
# fHit0b    84.773993       17.604392       973255  
# fHit0c    41.196664       7.8900392       578942
# fHit1     55.650121       10.229783       960058
# fHit2     14.470345       2.5089225       324683  
# fHitH     11.993302       2.2295808       356878   
# weighted sum of all classes for each bin 
weightedSum_all = [
174.27493+       48.293300+       1377762,
84.773993+       17.604392+       973255 ,
41.196664+       7.8900392+       578942 ,
55.650121+       10.229783+       960058 ,
14.470345+       2.5089225+       324683 ,
11.993302+       2.2295808+       356878 ]
wSum_1 = weightedSum_all[args.bin]

# weights for each bin & class
classWeights_noWeight = [
[ 1.0, 1.0, 1.0 ],
[ 1.0, 1.0, 1.0 ],
[ 1.0, 1.0, 1.0 ],
[ 1.0, 1.0, 1.0 ],
[ 1.0, 1.0, 1.0 ],
[ 1.0, 1.0, 1.0 ] ]
classWeights_allEqual = [
[ wSum_1/174.27493,       wSum_1/48.293300,       wSum_1/1377762],
[ wSum_1/84.773993,       wSum_1/17.604392,       wSum_1/973255 ],
[ wSum_1/41.196664,       wSum_1/7.8900392,       wSum_1/578942 ],
[ wSum_1/55.650121,       wSum_1/10.229783,       wSum_1/960058 ],
[ wSum_1/14.470345,       wSum_1/2.5089225,       wSum_1/324683 ],
[ wSum_1/11.993302,       wSum_1/2.2295808,       wSum_1/356878 ]]
classWeights_doubleG = [
[ wSum_1/174.27493*2,       wSum_1/48.293300*2,       wSum_1/1377762],
[ wSum_1/84.773993*2,       wSum_1/17.604392*2,       wSum_1/973255 ],
[ wSum_1/41.196664*2,       wSum_1/7.8900392*2,       wSum_1/578942 ],
[ wSum_1/55.650121*2,       wSum_1/10.229783*2,       wSum_1/960058 ],
[ wSum_1/14.470345*2,       wSum_1/2.5089225*2,       wSum_1/324683 ],
[ wSum_1/11.993302*2,       wSum_1/2.2295808*2,       wSum_1/356878 ]]
classWeights_doubleH = [
[ wSum_1/174.27493,       wSum_1/48.293300,       wSum_1/1377762*2],
[ wSum_1/84.773993,       wSum_1/17.604392,       wSum_1/973255 *2],
[ wSum_1/41.196664,       wSum_1/7.8900392,       wSum_1/578942 *2],
[ wSum_1/55.650121,       wSum_1/10.229783,       wSum_1/960058 *2],
[ wSum_1/14.470345,       wSum_1/2.5089225,       wSum_1/324683 *2],
[ wSum_1/11.993302,       wSum_1/2.2295808,       wSum_1/356878 *2]]

if (args.weightedClasses==0):
    classWeights = tf.constant(classWeights_noWeight[args.bin])
if (args.weightedClasses==1):
    classWeights = tf.constant(classWeights_allEqual[args.bin])
if (args.weightedClasses==2):
    classWeights = tf.constant(classWeights_doubleG[args.bin])
if (args.weightedClasses==3):
    classWeights = tf.constant(classWeights_doubleH[args.bin])



print ( "Load the data =================================================================================" )
def load_data():
    # Load files.
    hdf_train = pd.HDFStore('%s/training_sweets_dec20_noise_MPF_allParticle_bin_%d_3class.h5'%(args.fileDir,args.bin),'r')
    hdf_test = pd.HDFStore('%s/testing_sweets_dec20_noise_MPF_allParticle_bin_%d_3class.h5'%(args.fileDir,args.bin),'r')
    # Store the HDF5 dataframe object in memory (this would cause an error if the file is too large).
    df_train = hdf_train["training"]
    df_test = hdf_test["testing"]
    
    # Scale up the weights for Gamma events.
    df_train_weight = df_train['TWgt']
    #print ("Training weights (DataFrame): ", df_train_weight.head())
    df_test_weight = df_test['TWgt']
    #print ("Testing weights (DataFrame): ", df_test_weight.head())

    # Process features.
    df_train_feature = df_train[features]
    #df_train_feature['bias'] = float(1)
    #print ("Training features (DataFrame): ", df_train_feature.head())
    df_test_feature = df_test[features]
    #df_test_feature['bias'] = float(1)
    #print ("Testing features (DataFrame): ", df_test_feature.head())

    # Process classes using one-hot expression.
    df_train['pclass0'] = df_train['pclass'].map(lambda x: 1 if (x==0) else 0)
    df_train['pclass1'] = df_train['pclass'].map(lambda x: 1 if (x==1) else 0)
    df_train['pclass2'] = df_train['pclass'].map(lambda x: 1 if (x==2) else 0)
    df_train_class = df_train[['pclass0','pclass1','pclass2']]
    df_train_class_1 = df_train['pclass']
    #print ("Training classes (DataFrame): ", df_train_class.head())
    df_test['pclass0'] = df_test['pclass'].map(lambda x: 1 if (x==0) else 0)
    df_test['pclass1'] = df_test['pclass'].map(lambda x: 1 if (x==1) else 0)
    df_test['pclass2'] = df_test['pclass'].map(lambda x: 1 if (x==2) else 0)
    df_test_class = df_test[['pclass0','pclass1','pclass2']]
    df_test_class_1 = df_test['pclass']
    #print ("Testing classes (DataFrame): ", df_test_class.head())

    # Convert DataFrame to array.
    array_train_feature = np.array(df_train_feature) 
    array_train_class = np.array(df_train_class) 
    array_train_class_1 = np.array(df_train_class_1) 
    array_train_weight = np.array(df_train_weight) 
    array_test_feature = np.array(df_test_feature) 
    array_test_class = np.array(df_test_class) 
    array_test_class_1 = np.array(df_test_class_1) 
    array_test_weight = np.array(df_test_weight) 

    # Return the arrays.
    return (array_train_feature,array_train_class,array_train_class_1,array_train_weight),(array_test_feature,array_test_class,array_train_class_1,array_test_weight)
    
# Call load_data()
# train_class is one-hot expression, train_class_1 is single value (0-3) expression
(train_feature,train_class,train_class_1,train_weight),(test_feature,test_class,test_class_1,test_weight) = load_data()
#print ("Training features (array): ", train_feature)
#print ("Training classes (array): ", train_class)
#print ("Training weights (array): ", train_weight)
#print ("Testing features (array): ", test_feature)
#print ("Testing classes (array): ", test_class)
#print ("Testing weights (array): ", test_weight)

# Define a function to generate mini-batch set.
def next_batch(batch_num, X_data, Y_labels, W_labels):
    idx = np.arange(0 , len(X_data)) # get all evnets number as a list 
    np.random.shuffle(idx) # shuffle the list
    idx = idx[:batch_num] # get the index for one mini-batch events 
    data_shuffle = [X_data[i] for i in idx] # get "X" for one mini-batch
    labels_shuffle = [Y_labels[i] for i in idx] # get corresponding "Y" 
    weights_shuffle = [W_labels[i] for i in idx] # get corresponding weights 
    # Return "X", "Y"  and event weights for one min-batch with random events
    return np.asarray(data_shuffle), np.asarray(labels_shuffle), np.asarray(weights_shuffle)



print ( "Training settings =============================================================================" )
learning_rate = args.learningRate
training_epochs = args.epochs # each epoch goes through the full dataset
batch_size = args.batchSize # event number in each mini-batch
total_batch = int(len(train_feature)/batch_size) # batch needed to cover all events
display_step = args.displayStep # display with a epoch step



print ( "Load model(1-HL-NN, 40 features, 3 or 4 outputs) ==============================================" )
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
    if (inum==40 and onum==3):
        start_line = [0, 240, 262, 266]
        end_line = [240, 261, 266, 267]
    if (inum==40 and onum==4):
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

weights_test = {
    'h': tf.Variable(w0, dtype=dtype),
    'out': tf.Variable(w1, dtype=dtype) }
biases_test = {
    'b': tf.Variable(b0, dtype=dtype),
    'out': tf.Variable(b1, dtype=dtype) }

variable_w = weights_test
variable_b = biases_test

# debugging function
def testFunc(x):
    res_w0 = tf.matmul(x, weights_test['h'])
    res_b0 = tf.add(res_w0, biases_test['b'])
    res_layer0 = tf.nn.relu( res_b0 )
    res_w1 = tf.matmul(res_layer0, weights_test['out'])
    res_b1 = tf.add(res_w1, biases_test['out'])
    return res_w0,res_b0,res_layer0,res_w1,res_b1



print ( "Construct and save the mode ===================================================================" )
num_features = 40
num_classes = 3
# Graph input.
X = tf.placeholder("float", [None, num_features])
Y = tf.placeholder("float", [None, num_classes]) # one-hot labels
W = tf.placeholder("float", [None]) # event weights

res_w0,res_b0,res_layer0,res_w1,logits = testFunc(X)

# Use "softmax" to calculate the probability for each class.
prediction = tf.nn.softmax(logits) 

# Define the loss with cross entropy.
if (args.weightedEvents!=1): # unweighted loss
    # loss_vec is a vector that has length of batch size, each element is a loss for one event
    loss_temp = tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=logits) # classes are mutually exclusive
    # loss_op calculates the mean of losses in each batch
    loss_op = tf.reduce_mean(loss_temp)
else: # weighted loss, as default
    loss_temp1 = tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=logits)
    # calculate the weight on top of class weight, a vector with size of batch
    tempWeights = tf.reduce_sum( classWeights * Y, axis=1 )
    loss_temp2 = loss_temp1 * tempWeights
    # element-wise multiply for two vectors that both have length of batzh size
    loss_temp3 = tf.reduce_sum( tf.multiply(loss_temp2, W) )
    # the weighted version of loss = weighted sum of losses / all weights sum
    loss_op = tf.divide( loss_temp3, tf.reduce_sum(W) )
    # test the model without using any class weights (event weights is still valid)
    loss_op_test = tf.divide( tf.reduce_sum( tf.multiply(loss_temp1, W) ), tf.reduce_sum(W) )

# Define optimizer and train operation.
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate) # Adam already handles learning rate optimization
train_op = optimizer.minimize(loss_op)



print ( "Evaluate the model ============================================================================" )
# Accuracy checks if the prediction equals to the true value.
# tf.argmax() returns the index with the largest value across axes of a tensor.
# tf.argmax(*,0) returns the indices of max in each column.
# tf.argmax(*,1) returns the indices of max in each row.
# "prediction" for each event is a list of 3 float probabilities.
correct_pred = tf.equal( tf.argmax(prediction, 1), tf.argmax(Y, 1) ) # a list of boolean
accuracy = tf.reduce_mean( tf.cast(correct_pred, dtype) )

# Precision(for one class) = true predicted / all predicted
def precision(true, pred):
    # sum of events with correct prediction
    correctP = tf.reduce_sum( tf.cast(tf.logical_and(true,pred), dtype) )
    # sum of events predicted as that class
    sum_pred = tf.reduce_sum( tf.cast(pred, dtype) )
    if (sum_pred!=0):
        res = tf.divide(correctP, sum_pred)
    else:
        res = -1
    return res

# Precision with event weights
def precision_w(true, pred, weights):
    # sum of events with correct prediction
    correctP = tf.reduce_sum( tf.multiply(tf.cast(tf.logical_and(true,pred), dtype), weights) )
    # sum of events predicted as that class
    sum_pred = tf.reduce_sum( tf.multiply(tf.cast(pred, dtype), weights) )
    if (sum_pred!=0):
        res = tf.divide(correctP, sum_pred)
    else:
        res = -1
    return res

# Recall(for one class) = true predicted / all true
def recall(true, pred):
    # sum of events with correct prediction
    correctP = tf.reduce_sum( tf.cast(tf.logical_and(true,pred), dtype) )
    # sum of events which are truely that class
    sum_true = tf.reduce_sum( tf.cast(true, dtype) )
    if (sum_true!=0):
        res = tf.divide(correctP, sum_true)
    else:
        res = -1
    return res

# Recall with event weights
def recall_w(true, pred, weights):
    # sum of events with correct prediction
    correctP = tf.reduce_sum( tf.multiply(tf.cast(tf.logical_and(true,pred), dtype), weights) )
    # sum of events which are truely that class
    sum_true = tf.reduce_sum( tf.multiply(tf.cast(true, dtype), weights) )
    if (sum_true!=0):
        res = tf.divide(correctP, sum_true)
    else:
        res = -1
    return res

# pseudoPrecision returns "precision" of one predicted class.
# use event weights for the calculation.
def pseudoPrecision(true0, true1, true2, pred, weights):
    # calculate the sum of each true labeled as that class
    true0P = tf.reduce_sum( tf.multiply(tf.cast(tf.logical_and(true0,pred), dtype), weights) )
    true1P = tf.reduce_sum( tf.multiply(tf.cast(tf.logical_and(true1,pred), dtype), weights) )
    true2P = tf.reduce_sum( tf.multiply(tf.cast(tf.logical_and(true2,pred), dtype), weights) )
    denominator = tf.sqrt(true2P)
    if (denominator!=0):
        res = tf.divide( tf.reduce_sum(true0P + true1P), denominator )
    else:
        res = -1
    return res

# pseudoRecall returns "recall" for one predicted class.
# use event weights for the calculation.
def pseudoRecall(true0, true1, pred, weights):
    # calculate the sum of each true labeled as that class
    true0P = tf.reduce_sum( tf.multiply(tf.cast(tf.logical_and(true0,pred), dtype), weights) )
    true1P = tf.reduce_sum( tf.multiply(tf.cast(tf.logical_and(true1,pred), dtype), weights) )
    trueY = tf.reduce_sum( tf.multiply(tf.cast(tf.logical_or(true0, true1), dtype), weights) )
    if (trueY!=0):
        res = tf.divide( tf.reduce_sum(true0P + true1P), trueY )
    else:
        res = -1
    return res

# rejection measures the NN ability of removing noise(non-GG) events.
# use event weights for the calculation.
# rejection = 1 - fraction of non-GG labeled as GG
#           = 1 - (true non-GG labeled as GG) / (all true non-GG)
#           = 1-  ( (1-precision_GG)*pred_GG ) / ( true non-GG )
#           = 1 - (1-precision_GG) * (pred_GG)/(true_nonGG)
def rejection(precision_GG, pred_GG, true_nonGG, weights):
    count_predGG = tf.reduce_sum( tf.multiply(tf.cast(pred_GG, dtype), weights) )
    count_trueNoneGG = tf.reduce_sum(tf.multiply(tf.cast(true_nonGG, dtype), weights) )
    misLabel = tf.divide( count_predGG, count_trueNoneGG )
    if ( tf.reduce_sum(tf.cast(true_nonGG, dtype)) !=0 ):
        res = 1 - (1-precision_GG) * misLabel
    else:
        res = -1
    return res

# Get boolean list for each type.
trueGG = tf.equal( tf.argmax(Y, 1), 0 )
trueBG = tf.equal( tf.argmax(Y, 1), 1 )
trueH = tf.equal( tf.argmax(Y, 1), 2 )
trueNoneGG = tf.not_equal( tf.argmax(Y, 1), 0 )
predGG = tf.equal( tf.argmax(prediction, 1), 0 )
predBG = tf.equal( tf.argmax(prediction, 1), 1 )
predH = tf.equal( tf.argmax(prediction, 1), 2 )

# Precision (weighted and not) for each class.
precision_GG = precision(trueGG, predGG)
precision_BG = precision(trueBG, predBG)
precision_H = precision(trueH, predH)
precision_w_GG = precision_w(trueGG, predGG, W)
precision_w_BG = precision_w(trueBG, predBG, W)
precision_w_H = precision_w(trueH, predH, W)

# Recall (weighted and not) for each class.
recall_GG = recall(trueGG, predGG)
recall_BG = recall(trueBG, predBG)
recall_H = recall(trueH, predH)
recall_w_GG = recall_w(trueGG, predGG, W)
recall_w_BG = recall_w(trueBG, predBG, W)
recall_w_H = recall_w(trueH, predH, W)


# Since eventually we will select out events labelas as GoodGamma (for map-making), we define below parameters:
# pseudo-Precision =  (true (good+bad)Gamma labeled as GoodGamma) / sqrt(true Hadron labeled as GoodGamma )
# pseudo-Precision is comparable to the final significance.
pseudoprecisionGG = pseudoPrecision(trueGG, trueBG, trueH, predGG, W)

# pseudo-Recall for Gamma = (true (good+bad)Gamma labeled as GoodGamma) / (true (good+bad)Gamma)
# pseudo-Recall for Hadron = (true Hadron labeled as GoodGamma) / (true Hadron)
pseudorecallG = pseudoRecall(trueGG, trueBG, predGG, W)
pseudorecallH = pseudoRecall(trueH, trueH, predGG, W)

# rejection of non-GoodGamma events (noise)
rejectionNoneGG = rejection(precision_w_GG, predGG, trueNoneGG, W)
rejectionNoneGG_ = 1 - rejection(precision_w_GG, predGG, trueNoneGG, W)

# indicator regardless of statistics
significance_ratio = tf.divide( recall_w_GG, tf.sqrt(1-rejectionNoneGG) )



print ( "Run Session ===================================================================================" )
# Initialize the variables.
# This initializer is an op that initializes global variables in the graph.
init = tf.global_variables_initializer()
# Use "with...as" statement to try safely.
# A Session object encapsulates the environment in which
# Operation objects are executed and Tensor objects are evaluated.
# Create list to hold numbers for plotting.
with tf.Session() as sess:

    # session.run() runs operations and
    # evaluates tensors in fetches.
    sess.run(init)

    # use all data
    train_x, train_y, train_w = train_feature, train_class, train_weight
    test_x, test_y, test_w = test_feature, test_class, test_weight

    #############################################################################
    # Debugging mode.
    #batch_size = 100000
    #total_batch = 5
    #total_0 = 0
    #mislabel_0 = 0
    #for batch_number in range(total_batch):
        # train.next_batch() returns a tuple of two arrays.
        #train_x, train_y, train_w = next_batch(batch_size,train_feature,train_class,train_weight)

    # Training cycle.
    for epoch in range(1, training_epochs+1):

        # Calculate all parameters with training data and add them to the lists.
        loss_tr, acc_tr, \
        preGG_tr, preBG_tr, preH_tr, preGG_w_tr, preBG_w_tr, preH_w_tr, \
        reGG_tr, reBG_tr, reH_tr, reGG_w_tr, reBG_w_tr, reH_w_tr, \
        pseudoPGG_tr, pseudoRG_tr, pseudoRH_tr, rejNoneGG_tr, rejNoneGG_tr_, sigRatio_tr = sess.run( \
        [loss_op, accuracy, \
        precision_GG, precision_BG, precision_H, precision_w_GG, precision_w_BG, precision_w_H, \
        recall_GG, recall_BG, recall_H, recall_w_GG, recall_w_BG, recall_w_H, \
        pseudoprecisionGG, pseudorecallG, pseudorecallH, rejectionNoneGG, rejectionNoneGG_, significance_ratio ], \
        feed_dict={X: train_x, Y: train_y, W: train_w})

        #print ("Epoch: ", epoch, "============")
        #print ("Loss: ", loss_tr)
        #print ("Accuracy: ", acc_tr)
        #print ("Recall(w) GG", reGG_w_tr)
        #print ("Rejection: ", rejNoneGG_tr)

        if (epoch==1):
            # outputs
            print ("Loss: -----------------")
            print (loss_tr)
            print ("Accuracy: -------------")
            print (acc_tr)
            print ("Precision Sig: --------")
            print (preGG_w_tr)
            print ("Precision Bkg: --------")
            print (preBG_w_tr,preH_w_tr)
            print ("Recall Sig: -----------")
            print (reGG_w_tr)
            print ("Recall Bkg: -----------")
            print (reBG_w_tr,reH_w_tr)
            print ("pseudoPrecision: ------")
            print (pseudoPGG_tr)
            print ("Rejection: ------------")
            print (rejNoneGG_tr)

        '''
        # check with condition
        trueClass, predClass = sess.run([tf.argmax(Y, 1),tf.argmax(prediction, 1)], feed_dict={X:train_x, Y:train_y, W:train_w})
        if (trueClass[0]==0):
            total_0 += 1
            if (predClass[0]!=0):
                mislabel_0 +=1

                print ("-----")
                print (total_0)
                print (mislabel_0)
            
            eventX, eventY, eventW, ttt_w0, ttt_b0, ttt_layer0, ttt_w1, logit, pred = sess.run( \
            [X, Y, W, res_w0, res_b0, res_layer0, res_w1, logits, prediction], \
            feed_dict={X:train_x, Y:train_y, W:train_w})

            print ("Printing ****************************************************")
            print ("Check event %d"%batch_number)
            print ("input X:")
            print (eventX)
            print ("output Y:")
            print (eventY)
            print ("event weights:")
            print (eventW)
            print ("-------------------------------")
            print ("Res_w0: ", ttt_w0 )
            print ("Res_b0: ", ttt_b0 )
            print ("Res_layer0: ", ttt_layer0 )
            print ("Res_w1: ", ttt_w1 )
            print ("Res_b1(logit): ", logit )
            print ("Pred Prob: ", pred)
            print ("prediction class: ", predClass)
            print ("true Class: ", trueClass)
            print ("-------------------------------")

            print ("Variables used ----------------")
            variables_w = []
            variables_b = []
            for i in range(len(variable_w)):
                # dictionary values doesn't return to a list directly
                variables_w.append( list(variable_w.values())[i].eval() )
                variables_b.append( list(variable_b.values())[i].eval() )
            #print (variables_w)
            #print (variables_b)
            '''

        #print ("One event check finished ************************************")
         
    #print ("Total mislabeled GoodGamma: ", mislabel_0)    
    print("Debugging Finished!---------------------------------------------")
    #############################################################################
    

