import os
import sys
import argparse
import time
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
from xml.etree import ElementTree as ET
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

parser.add_argument('--epochs', type=int, default=5000, 
                    help='number of total epochs (default=5000)')

parser.add_argument('--displayStep', type=int, default=50,
                    help='terminal display step (default=50)')

parser.set_defaults(augment=True)
args = parser.parse_args()

dtype = tf.float32 # set data type to be float tensor (32 bits)

if args.bin is None:
    print ("Please specify analysis bin !")
    sys.exit()

# create a directory containing all output images and trained model
out_directory_name = 'Bin_' + str(args.bin) + '_features_' + str(args.featureGroup) + '_weight_' + str(args.weightedEvents) + str(args.weightedClasses) + '_HL_' + str(args.hiddenlayer) + '_NF_' + str(args.neuronfactor) + '_LR_' + str(args.learningRate) + '_epochs_' + str(args.epochs)
if not os.path.exists(out_directory_name):
    os.makedirs(out_directory_name)

currentPath = str(os.getcwd())
modelPath = str( currentPath + '/' + out_directory_name + '/model.ckpt' )

if (args.weightedEvents==0):
    args.weightedClasses = 0



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
    features = list( set(features_all) - set(f_group0) ) # the order of features will be shuffled !
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
#           GoodGamma       BadGamma        GoodHadron        BadHadron
# fHit0a    174.27094       48.297292       34819907          8511550
# fHit0b    84.773993       17.604392       15939123          2770632
# fHit0c    41.192327       7.8943760       7178234           989307
# fHit1     55.650121       10.229783       10158870          1064614
# fHit2     14.470345       2.5089225       2989884           231220
# fHitH     11.993302       2.2295808       3039715           221999
# weighted sum of all classes for each bin 
weightedSum_all = [
174.27094+       48.297292+       34819907+          8511550,
84.773993+       17.604392+       15939123+          2770632,
41.192327+       7.8943760+       7178234 +          989307 ,
55.650121+       10.229783+       10158870+          1064614,
14.470345+       2.5089225+       2989884 +          231220 ,
11.993302+       2.2295808+       3039715 +          221999 ]
wSum_1 = weightedSum_all[args.bin]

# weights for each bin & class
classWeights_noWeight = [
[ 1.0, 1.0, 1.0, 1.0 ],
[ 1.0, 1.0, 1.0, 1.0 ],
[ 1.0, 1.0, 1.0, 1.0 ],
[ 1.0, 1.0, 1.0, 1.0 ],
[ 1.0, 1.0, 1.0, 1.0 ],
[ 1.0, 1.0, 1.0, 1.0 ] ]
classWeights_allEqual = [
[ wSum_1/174.27094,       wSum_1/48.297292,       wSum_1/34819907,          wSum_1/8511550],
[ wSum_1/84.773993,       wSum_1/17.604392,       wSum_1/15939123,          wSum_1/2770632],
[ wSum_1/41.192327,       wSum_1/7.8943760,       wSum_1/7178234 ,          wSum_1/989307 ],
[ wSum_1/55.650121,       wSum_1/10.229783,       wSum_1/10158870,          wSum_1/1064614],
[ wSum_1/14.470345,       wSum_1/2.5089225,       wSum_1/2989884 ,          wSum_1/231220 ],
[ wSum_1/11.993302,       wSum_1/2.2295808,       wSum_1/3039715 ,          wSum_1/221999 ]]
classWeights_doubleG = [
[ wSum_1/174.27094*2,       wSum_1/48.297292*2,       wSum_1/34819907,          wSum_1/8511550],
[ wSum_1/84.773993*2,       wSum_1/17.604392*2,       wSum_1/15939123,          wSum_1/2770632],
[ wSum_1/41.192327*2,       wSum_1/7.8943760*2,       wSum_1/7178234 ,          wSum_1/989307 ],
[ wSum_1/55.650121*2,       wSum_1/10.229783*2,       wSum_1/10158870,          wSum_1/1064614],
[ wSum_1/14.470345*2,       wSum_1/2.5089225*2,       wSum_1/2989884 ,          wSum_1/231220 ],
[ wSum_1/11.993302*2,       wSum_1/2.2295808*2,       wSum_1/3039715 ,          wSum_1/221999 ]]
classWeights_doubleH = [
[ wSum_1/174.27094,       wSum_1/48.297292,       wSum_1/34819907*2,          wSum_1/8511550*2],
[ wSum_1/84.773993,       wSum_1/17.604392,       wSum_1/15939123*2,          wSum_1/2770632*2],
[ wSum_1/41.192327,       wSum_1/7.8943760,       wSum_1/7178234 *2,          wSum_1/989307 *2],
[ wSum_1/55.650121,       wSum_1/10.229783,       wSum_1/10158870*2,          wSum_1/1064614*2],
[ wSum_1/14.470345,       wSum_1/2.5089225,       wSum_1/2989884 *2,          wSum_1/231220 *2],
[ wSum_1/11.993302,       wSum_1/2.2295808,       wSum_1/3039715 *2,          wSum_1/221999 *2]]

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
    hdf_train = pd.HDFStore('%s/training_sweets_dec20_noise_MPF_allParticle_bin_%d_4class.h5'%(args.fileDir,args.bin),'r')
    hdf_test = pd.HDFStore('%s/testing_sweets_dec20_noise_MPF_allParticle_bin_%d_4class.h5'%(args.fileDir,args.bin),'r')
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
    df_train['pclass3'] = df_train['pclass'].map(lambda x: 1 if (x==3) else 0)
    df_train_class = df_train[['pclass0','pclass1','pclass2','pclass3']]
    df_train_class_1 = df_train['pclass']
    #print ("Training classes (DataFrame): ", df_train_class.head())
    df_test['pclass0'] = df_test['pclass'].map(lambda x: 1 if (x==0) else 0)
    df_test['pclass1'] = df_test['pclass'].map(lambda x: 1 if (x==1) else 0)
    df_test['pclass2'] = df_test['pclass'].map(lambda x: 1 if (x==2) else 0)
    df_test['pclass3'] = df_test['pclass'].map(lambda x: 1 if (x==3) else 0)
    df_test_class = df_test[['pclass0','pclass1','pclass2','pclass3']]
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



print ( "Create model(NN) ==============================================================================" )
# For most ML problems, it's sufficient to use just one hidden layer.
# For most ML problems, the neuron number is between the input and output (one can start with the mean value)
# For this problem, we can have 7 to more than 20 input features, with 4 output classes.
num_features = len(features)
num_classes = 4
neuronsFactor = args.neuronfactor

# Model 0: NN-0HL ---------------------------------------------------
weights_0HL = {
    'out': tf.Variable(tf.random_normal([num_features, num_classes])) }
biases_0HL = {
    'out': tf.Variable(tf.random_normal([num_classes])) }
# Create model (fully connected neurons).
def neural_net_0HL(x):
    out_layer = tf.matmul(x, weights_0HL['out']) + biases_0HL['out']
    return out_layer

# Model 1: NN-1HL ---------------------------------------------------
hidden_1HL = int( (num_features + num_classes)*0.5*neuronsFactor )
# Store layers weight & bias.
weights_1HL = {
    'h': tf.Variable(tf.random_normal([num_features, hidden_1HL])),
    'out': tf.Variable(tf.random_normal([hidden_1HL, num_classes])) }
biases_1HL = {
    'b': tf.Variable(tf.random_normal([hidden_1HL])),
    'out': tf.Variable(tf.random_normal([num_classes])) }
# Create model (fully connected neurons).
def neural_net_1HL(x):
    layer = tf.nn.relu( tf.add(tf.matmul(x, weights_1HL['h']), biases_1HL['b']) )
    out_layer = tf.matmul(layer, weights_1HL['out']) + biases_1HL['out']
    return out_layer

# Model 2: NN-2HL ---------------------------------------------------
hidden_2HL_1 = int( (num_features + num_classes)*0.5*1.2*neuronsFactor ) 
hidden_2HL_2 = int( (num_features + num_classes)*0.5*0.8*neuronsFactor )
# Store layers weight & bias.
weights_2HL = {
    'h1': tf.Variable(tf.random_normal([num_features, hidden_2HL_1])),
    'h2': tf.Variable(tf.random_normal([hidden_2HL_1, hidden_2HL_2])),
    'out': tf.Variable(tf.random_normal([hidden_2HL_2, num_classes])) }
biases_2HL = {
    'b1': tf.Variable(tf.random_normal([hidden_2HL_1])),
    'b2': tf.Variable(tf.random_normal([hidden_2HL_2])),
    'out': tf.Variable(tf.random_normal([num_classes])) }
# Create model (fully connected neurons).
def neural_net_2HL(x):
    layer_1 = tf.nn.relu( tf.add(tf.matmul(x, weights_2HL['h1']), biases_2HL['b1']) )
    layer_2 = tf.nn.relu( tf.add(tf.matmul(layer_1, weights_2HL['h2']), biases_2HL['b2']) )
    out_layer = tf.matmul(layer_2, weights_2HL['out']) + biases_2HL['out']
    return out_layer

# Model 3: NN-3HL ---------------------------------------------------
hidden_3HL_1 = int( (num_features + num_classes)*0.5*1.2*neuronsFactor ) 
hidden_3HL_2 = int( (num_features + num_classes)*0.5*1.0*neuronsFactor )
hidden_3HL_3 = int( (num_features + num_classes)*0.5*0.8*neuronsFactor )
# Store layers weight & bias.
weights_3HL = {
    'h1': tf.Variable(tf.random_normal([num_features, hidden_3HL_1])),
    'h2': tf.Variable(tf.random_normal([hidden_3HL_1, hidden_3HL_2])),
    'h3': tf.Variable(tf.random_normal([hidden_3HL_2, hidden_3HL_3])),
    'out': tf.Variable(tf.random_normal([hidden_3HL_3, num_classes])) }
biases_3HL = {
    'b1': tf.Variable(tf.random_normal([hidden_3HL_1])),
    'b2': tf.Variable(tf.random_normal([hidden_3HL_2])),
    'b3': tf.Variable(tf.random_normal([hidden_3HL_3])),
    'out': tf.Variable(tf.random_normal([num_classes])) }
# Create model (fully connected neurons).
def neural_net_3HL(x):
    layer_1 = tf.nn.relu( tf.add(tf.matmul(x, weights_3HL['h1']), biases_3HL['b1']) )
    layer_2 = tf.nn.relu( tf.add(tf.matmul(layer_1, weights_3HL['h2']), biases_3HL['b2']) )
    layer_3 = tf.nn.relu( tf.add(tf.matmul(layer_2, weights_3HL['h3']), biases_3HL['b3']) )
    out_layer = tf.matmul(layer_3, weights_3HL['out']) + biases_3HL['out']
    return out_layer

# Model 4: NN-4HL ---------------------------------------------------
hidden_4HL_1 = int( (num_features + num_classes)*0.5*1.5*neuronsFactor ) 
hidden_4HL_2 = int( (num_features + num_classes)*0.5*1.2*neuronsFactor )
hidden_4HL_3 = int( (num_features + num_classes)*0.5*0.8*neuronsFactor )
hidden_4HL_4 = int( (num_features + num_classes)*0.5*0.5*neuronsFactor )
# Store layers weight & bias.
weights_4HL = {
    'h1': tf.Variable(tf.random_normal([num_features, hidden_4HL_1])),
    'h2': tf.Variable(tf.random_normal([hidden_4HL_1, hidden_4HL_2])),
    'h3': tf.Variable(tf.random_normal([hidden_4HL_2, hidden_4HL_3])),
    'h4': tf.Variable(tf.random_normal([hidden_4HL_3, hidden_4HL_4])),
    'out': tf.Variable(tf.random_normal([hidden_4HL_4, num_classes])) }
biases_4HL = {
    'b1': tf.Variable(tf.random_normal([hidden_4HL_1])),
    'b2': tf.Variable(tf.random_normal([hidden_4HL_2])),
    'b3': tf.Variable(tf.random_normal([hidden_4HL_3])),
    'b4': tf.Variable(tf.random_normal([hidden_4HL_4])),
    'out': tf.Variable(tf.random_normal([num_classes])) }
# Create model (fully connected neurons).
def neural_net_4HL(x):
    layer_1 = tf.nn.relu( tf.add(tf.matmul(x, weights_4HL['h1']), biases_4HL['b1']) )
    layer_2 = tf.nn.relu( tf.add(tf.matmul(layer_1, weights_4HL['h2']), biases_4HL['b2']) )
    layer_3 = tf.nn.relu( tf.add(tf.matmul(layer_2, weights_4HL['h3']), biases_4HL['b3']) )
    layer_4 = tf.nn.relu( tf.add(tf.matmul(layer_3, weights_4HL['h4']), biases_4HL['b4']) )
    out_layer = tf.matmul(layer_4, weights_4HL['out']) + biases_4HL['out']
    return out_layer



print ( "Construct and save the mode ===================================================================" )
# Graph input.
X = tf.placeholder("float", [None, num_features])
Y = tf.placeholder("float", [None, num_classes]) # one-hot labels
W = tf.placeholder("float", [None]) # event weights

# Raw output out of NN (logits).
if (args.hiddenlayer==0):
    nnModel = neural_net_0HL
    variable_w = weights_0HL
    variable_b = biases_0HL
if (args.hiddenlayer==1):
    nnModel = neural_net_1HL
    variable_w = weights_1HL
    variable_b = biases_1HL
if (args.hiddenlayer==2):
    nnModel = neural_net_2HL
    variable_w = weights_2HL
    variable_b = biases_2HL
if (args.hiddenlayer==3):
    nnModel = neural_net_3HL
    variable_w = weights_3HL
    variable_b = biases_3HL
if (args.hiddenlayer==4):
    nnModel = neural_net_4HL
    variable_w = weights_4HL
    variable_b = biases_4HL

logits = nnModel(X)

# Use "softmax" to calculate the probability for each class.
prediction = tf.nn.softmax(logits) 

# Define the loss with cross entropy.
if (args.weightedEvents==0): # unweighted loss
    # loss_vec is a vector that has length of batch size, each element is a loss for one event
    loss_temp = tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=logits) # classes are mutually exclusive
    # loss_op calculates the mean of losses in each batch
    loss_op_test = loss_op = tf.reduce_mean(loss_temp)
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

# 'Saver' op to save and restore all the variables
saver = tf.train.Saver()



print ( "Evaluate the model ============================================================================" )
# Accuracy checks if the prediction equals to the true value.
# tf.argmax() returns the index with the largest value across axes of a tensor.
# tf.argmax(*,0) returns the indices of max in each column.
# tf.argmax(*,1) returns the indices of max in each row.
# "prediction" for each event is a list of 4 float probabilities.
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
def pseudoPrecision(true0, true1, true2, true3, pred, weights):
    # calculate the sum of each true labeled as that class
    true0P = tf.reduce_sum( tf.multiply(tf.cast(tf.logical_and(true0,pred), dtype), weights) )
    true1P = tf.reduce_sum( tf.multiply(tf.cast(tf.logical_and(true1,pred), dtype), weights) )
    true2P = tf.reduce_sum( tf.multiply(tf.cast(tf.logical_and(true2,pred), dtype), weights) )
    true3P = tf.reduce_sum( tf.multiply(tf.cast(tf.logical_and(true3,pred), dtype), weights) )
    denominator = tf.sqrt(true2P + true3P)
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
trueGH = tf.equal( tf.argmax(Y, 1), 2 )
trueBH = tf.equal( tf.argmax(Y, 1), 3 )
trueNoneGG = tf.not_equal( tf.argmax(Y, 1), 0 )
predGG = tf.equal( tf.argmax(prediction, 1), 0 )
predBG = tf.equal( tf.argmax(prediction, 1), 1 )
predGH = tf.equal( tf.argmax(prediction, 1), 2 )
predBH = tf.equal( tf.argmax(prediction, 1), 3 )

# Precision (weighted and not) for each class.
precision_GG = precision(trueGG, predGG)
precision_BG = precision(trueBG, predBG)
precision_GH = precision(trueGH, predGH)
precision_BH = precision(trueBH, predBH)
precision_w_GG = precision_w(trueGG, predGG, W)
precision_w_BG = precision_w(trueBG, predBG, W)
precision_w_GH = precision_w(trueGH, predGH, W)
precision_w_BH = precision_w(trueBH, predBH, W)

# Recall (weighted and not) for each class.
recall_GG = recall(trueGG, predGG)
recall_BG = recall(trueBG, predBG)
recall_GH = recall(trueGH, predGH)
recall_BH = recall(trueBH, predBH)
recall_w_GG = recall_w(trueGG, predGG, W)
recall_w_BG = recall_w(trueBG, predBG, W)
recall_w_GH = recall_w(trueGH, predGH, W)
recall_w_BH = recall_w(trueBH, predBH, W)

# Since eventually we will select out events labelas as GoodGamma (for map-making), we define below parameters:
# pseudo-Precision =  (true (good+bad)Gamma labeled as GoodGamma) / sqrt(true (good+bad)Hadron labeled as GoodGamma )
# pseudo-Precision is comparable to the final significance.
pseudoprecisionGG = pseudoPrecision(trueGG, trueBG, trueGH, trueBH, predGG, W)

# pseudo-Recall for Gamma = (true (good+bad)Gamma labeled as GoodGamma) / (true (good+bad)Gamma)
# pseudo-Recall for Hadron = (true (good+bad)Hadron labeled as GoodGamma) / (true (good+bad)Hadron)
pseudorecallG = pseudoRecall(trueGG, trueBG, predGG, W)
pseudorecallH = pseudoRecall(trueGH, trueBH, predGG, W)

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

final_indicators = []

# Create list to hold numbers for plotting.
DisplayStep = []

LossPlot_train = []
AccuracyPlot_train = []
PrecisionGGPlot_train = []
PrecisionBGPlot_train = []
PrecisionGHPlot_train = []
PrecisionBHPlot_train = []
PrecisionGGPlot_w_train = []
PrecisionBGPlot_w_train = []
PrecisionGHPlot_w_train = []
PrecisionBHPlot_w_train = []
RecallGGPlot_train = []
RecallBGPlot_train = []
RecallGHPlot_train = []
RecallBHPlot_train = []
RecallGGPlot_w_train = []
RecallBGPlot_w_train = []
RecallGHPlot_w_train = []
RecallBHPlot_w_train = []
PseudoPGGPlot_train = []
PseudoRGPlot_train = []
PseudoRHPlot_train = []
RejectionNoneGG_train = []
RejectionNoneGG_train_ = []
SigRatio_train = []

LossPlot_test = []
AccuracyPlot_test = []
PrecisionGGPlot_test = []
PrecisionBGPlot_test = []
PrecisionGHPlot_test = []
PrecisionBHPlot_test = []
PrecisionGGPlot_w_test = []
PrecisionBGPlot_w_test = []
PrecisionGHPlot_w_test = []
PrecisionBHPlot_w_test = []
RecallGGPlot_test = []
RecallBGPlot_test = []
RecallGHPlot_test = []
RecallBHPlot_test = []
RecallGGPlot_w_test = []
RecallBGPlot_w_test = []
RecallGHPlot_w_test = []
RecallBHPlot_w_test = []
PseudoPGGPlot_test = []
PseudoRGPlot_test = []
PseudoRHPlot_test = []
RejectionNoneGG_test = []
RejectionNoneGG_test_ = []
SigRatio_test = []

with tf.Session() as sess:

    # session.run() runs operations and
    # evaluates tensors in fetches.
    sess.run(init)
    
    '''
    #############################################################################
    # Debugging mode.
    batch_size = 100
    total_batch = 10
    for epoch in range(1):
        for batch_number in range(total_batch):
            # train.next_batch() returns a tuple of two arrays.
            train_x, train_y, train_w = next_batch(batch_size,train_feature,train_class,train_weight)
            test_x, test_y, test_w = next_batch(batch_size,test_feature,test_class,test_weight)
            # Run optimization op (backprop).
            sess.run(train_op, feed_dict={X:train_x, Y:train_y, W:train_w})

            # display ---------------------------
            DisplayStep.append(int(batch_number))

            loss_tr, acc_tr, \
            preGG_tr, preBG_tr, preGH_tr, preBH_tr, preGG_w_tr, preBG_w_tr, preGH_w_tr, preBH_w_tr, \
            reGG_tr, reBG_tr, reGH_tr, reBH_tr, reGG_w_tr, reBG_w_tr, reGH_w_tr, reBH_w_tr, \
            pseudoPGG_tr, pseudoRG_tr, pseudoRH_tr, rejNoneGG_tr, rejNoneGG_tr_, sigRatio_tr = sess.run( \
            [loss_op, accuracy, \
            precision_GG, precision_BG, precision_GH, precision_BH, precision_w_GG, precision_w_BG, precision_w_GH, precision_w_BH, \
            recall_GG, recall_BG, recall_GH, recall_BH, recall_w_GG, recall_w_BG, recall_w_GH, recall_w_BH, \
            pseudoprecisionGG, pseudorecallG, pseudorecallH, rejectionNoneGG, rejectionNoneGG_, significance_ratio ], \
            feed_dict={X: train_x, Y: train_y, W: train_w})

            LossPlot_train.append(float(loss_tr))
            AccuracyPlot_train.append(float(acc_tr))
            PrecisionGGPlot_train.append(float(preGG_tr))
            PrecisionBGPlot_train.append(float(preBG_tr))
            PrecisionGHPlot_train.append(float(preGH_tr))
            PrecisionBHPlot_train.append(float(preBH_tr))
            PrecisionGGPlot_w_train.append(float(preGG_w_tr))
            PrecisionBGPlot_w_train.append(float(preBG_w_tr))
            PrecisionGHPlot_w_train.append(float(preGH_w_tr))
            PrecisionBHPlot_w_train.append(float(preBH_w_tr))
            RecallGGPlot_train.append(float(reGG_tr))
            RecallBGPlot_train.append(float(reBG_tr))
            RecallGHPlot_train.append(float(reGH_tr))
            RecallBHPlot_train.append(float(reBH_tr))
            RecallGGPlot_w_train.append(float(reGG_w_tr))
            RecallBGPlot_w_train.append(float(reBG_w_tr))
            RecallGHPlot_w_train.append(float(reGH_w_tr))
            RecallBHPlot_w_train.append(float(reBH_w_tr))
            PseudoPGGPlot_train.append(float(pseudoPGG_tr))
            PseudoRGPlot_train.append(float(pseudoRG_tr))
            PseudoRHPlot_train.append(float(pseudoRH_tr))
            RejectionNoneGG_train.append(float(rejNoneGG_tr))
            RejectionNoneGG_train_.append(float(rejNoneGG_tr_))
            SigRatio_train.append(float(sigRatio_tr))

            # Calculate all parameters in trained net with test data.
            loss_te, acc_te, \
            preGG_te, preBG_te, preGH_te, preBH_te, preGG_w_te, preBG_w_te, preGH_w_te, preBH_w_te, \
            reGG_te, reBG_te, reGH_te, reBH_te, reGG_w_te, reBG_w_te, reGH_w_te, reBH_w_te, \
            pseudoPGG_te, pseudoRG_te, pseudoRH_te, rejNoneGG_te, rejNoneGG_te_, sigRatio_te = sess.run( \
            [loss_op_test, accuracy, \
            precision_GG, precision_BG, precision_GH, precision_BH, precision_w_GG, precision_w_BG, precision_w_GH, precision_w_BH, \
            recall_GG, recall_BG, recall_GH, recall_BH, recall_w_GG, recall_w_BG, recall_w_GH, recall_w_BH, \
            pseudoprecisionGG, pseudorecallG, pseudorecallH, rejectionNoneGG, rejectionNoneGG_, significance_ratio ], \
            feed_dict={X:test_x, Y:test_y, W:test_w})

            LossPlot_test.append(float(loss_te))
            AccuracyPlot_test.append(float(acc_te))
            PrecisionGGPlot_test.append(float(preGG_te))
            PrecisionBGPlot_test.append(float(preBG_te))
            PrecisionGHPlot_test.append(float(preGH_te))
            PrecisionBHPlot_test.append(float(preBH_te))
            PrecisionGGPlot_w_test.append(float(preGG_w_te))
            PrecisionBGPlot_w_test.append(float(preBG_w_te))
            PrecisionGHPlot_w_test.append(float(preGH_w_te))
            PrecisionBHPlot_w_test.append(float(preBH_w_te))
            RecallGGPlot_test.append(float(reGG_te))
            RecallBGPlot_test.append(float(reBG_te))
            RecallGHPlot_test.append(float(reGH_te))
            RecallBHPlot_test.append(float(reBH_te))
            RecallGGPlot_w_test.append(float(reGG_w_te))
            RecallBGPlot_w_test.append(float(reBG_w_te))
            RecallGHPlot_w_test.append(float(reGH_w_te))
            RecallBHPlot_w_test.append(float(reBH_w_te))
            PseudoPGGPlot_test.append(float(pseudoPGG_te))
            PseudoRGPlot_test.append(float(pseudoRG_te))
            PseudoRHPlot_test.append(float(pseudoRH_te))
            RejectionNoneGG_test.append(float(rejNoneGG_te))
            RejectionNoneGG_test_.append(float(rejNoneGG_te_))
            SigRatio_test.append(float(sigRatio_te))

            print ( "Batch: " + "{:3d}".format(batch_number) + \
                    ",  Loss= " + "{:.3f}".format(loss_tr) + \
                    ",  Accuracy= " + "{:.3f}".format(acc_tr) + "-------------------------------" )   
            
            trueClass,predClass,eventW,logit,pred = sess.run( \
            [tf.argmax(Y, 1), tf.argmax(prediction, 1), W, logits, prediction], \
            feed_dict={X:train_x, Y:train_y, W:train_w})
            #print ("input X:")
            #print (train_x)
            #print ("output Y:")
            #print (train_y)
            #print ("event weights:")
            #print (eventW)
            #print ("true Class:")
            #print (trueClass)
            #print ("NN logits:")
            #print (logit)
            #print ("prediction prob:")
            #print (pred)
            #print ("prediction class:")
            #print (predClass)

    variables_w = []
    variables_b = []
    for i in range(len(variable_w)):
        # dictionary values doesn't return to a list directly
        variables_w.append( list(variable_w.values())[i].eval() )
        variables_b.append( list(variable_b.values())[i].eval() )
    print (variables_w)
    print (variables_b)

    print("Optimization Finished!------------------------------------------")
    #############################################################################
    '''

    
    #############################################################################
    # Training cycle.
    for epoch in range(1, training_epochs+1):

        # training with mini-batch
        if (args.trainType==0):
            for batch_number in range(total_batch):
                # train.next_batch() returns a tuple of two arrays.
                train_x, train_y, train_w = next_batch(batch_size,train_feature,train_class,train_weight)
                test_x, test_y, test_w = next_batch(batch_size,test_feature,test_class,test_weight)
                # Run optimization op (backprop).
                sess.run(train_op, feed_dict={X:train_x, Y:train_y, W:train_w})

        # training with full-batch
        else:
            train_x, train_y, train_w = train_feature, train_class, train_weight
            test_x, test_y, test_w = test_feature, test_class, test_weight
            # Run optimization op (backprop).
            sess.run(train_op, feed_dict={X:train_x, Y:train_y, W:train_w})

        # Display in terminal --------------------------
        if (epoch % display_step == 0 or epoch == 1):
            DisplayStep.append(int(epoch))

            # Calculate all parameters with training data and add them to the lists.
            loss_tr, acc_tr, \
            preGG_tr, preBG_tr, preGH_tr, preBH_tr, preGG_w_tr, preBG_w_tr, preGH_w_tr, preBH_w_tr, \
            reGG_tr, reBG_tr, reGH_tr, reBH_tr, reGG_w_tr, reBG_w_tr, reGH_w_tr, reBH_w_tr, \
            pseudoPGG_tr, pseudoRG_tr, pseudoRH_tr, rejNoneGG_tr, rejNoneGG_tr_, sigRatio_tr = sess.run( \
            [loss_op, accuracy, \
            precision_GG, precision_BG, precision_GH, precision_BH, precision_w_GG, precision_w_BG, precision_w_GH, precision_w_BH, \
            recall_GG, recall_BG, recall_GH, recall_BH, recall_w_GG, recall_w_BG, recall_w_GH, recall_w_BH, \
            pseudoprecisionGG, pseudorecallG, pseudorecallH, rejectionNoneGG, rejectionNoneGG_, significance_ratio ], \
            feed_dict={X: train_x, Y: train_y, W: train_w})

            LossPlot_train.append(float(loss_tr))
            AccuracyPlot_train.append(float(acc_tr))
            PrecisionGGPlot_train.append(float(preGG_tr))
            PrecisionBGPlot_train.append(float(preBG_tr))
            PrecisionGHPlot_train.append(float(preGH_tr))
            PrecisionBHPlot_train.append(float(preBH_tr))
            PrecisionGGPlot_w_train.append(float(preGG_w_tr))
            PrecisionBGPlot_w_train.append(float(preBG_w_tr))
            PrecisionGHPlot_w_train.append(float(preGH_w_tr))
            PrecisionBHPlot_w_train.append(float(preBH_w_tr))
            RecallGGPlot_train.append(float(reGG_tr))
            RecallBGPlot_train.append(float(reBG_tr))
            RecallGHPlot_train.append(float(reGH_tr))
            RecallBHPlot_train.append(float(reBH_tr))
            RecallGGPlot_w_train.append(float(reGG_w_tr))
            RecallBGPlot_w_train.append(float(reBG_w_tr))
            RecallGHPlot_w_train.append(float(reGH_w_tr))
            RecallBHPlot_w_train.append(float(reBH_w_tr))
            PseudoPGGPlot_train.append(float(pseudoPGG_tr))
            PseudoRGPlot_train.append(float(pseudoRG_tr))
            PseudoRHPlot_train.append(float(pseudoRH_tr))
            RejectionNoneGG_train.append(float(rejNoneGG_tr))
            RejectionNoneGG_train_.append(float(rejNoneGG_tr_))
            SigRatio_train.append(float(sigRatio_tr))

            # Calculate all parameters in trained net with test data.
            loss_te, acc_te, \
            preGG_te, preBG_te, preGH_te, preBH_te, preGG_w_te, preBG_w_te, preGH_w_te, preBH_w_te, \
            reGG_te, reBG_te, reGH_te, reBH_te, reGG_w_te, reBG_w_te, reGH_w_te, reBH_w_te, \
            pseudoPGG_te, pseudoRG_te, pseudoRH_te, rejNoneGG_te, rejNoneGG_te_, sigRatio_te = sess.run( \
            [loss_op_test, accuracy, \
            precision_GG, precision_BG, precision_GH, precision_BH, precision_w_GG, precision_w_BG, precision_w_GH, precision_w_BH, \
            recall_GG, recall_BG, recall_GH, recall_BH, recall_w_GG, recall_w_BG, recall_w_GH, recall_w_BH, \
            pseudoprecisionGG, pseudorecallG, pseudorecallH, rejectionNoneGG, rejectionNoneGG_, significance_ratio ], \
            feed_dict={X:test_x, Y:test_y, W:test_w})

            LossPlot_test.append(float(loss_te))
            AccuracyPlot_test.append(float(acc_te))
            PrecisionGGPlot_test.append(float(preGG_te))
            PrecisionBGPlot_test.append(float(preBG_te))
            PrecisionGHPlot_test.append(float(preGH_te))
            PrecisionBHPlot_test.append(float(preBH_te))
            PrecisionGGPlot_w_test.append(float(preGG_w_te))
            PrecisionBGPlot_w_test.append(float(preBG_w_te))
            PrecisionGHPlot_w_test.append(float(preGH_w_te))
            PrecisionBHPlot_w_test.append(float(preBH_w_te))
            RecallGGPlot_test.append(float(reGG_te))
            RecallBGPlot_test.append(float(reBG_te))
            RecallGHPlot_test.append(float(reGH_te))
            RecallBHPlot_test.append(float(reBH_te))
            RecallGGPlot_w_test.append(float(reGG_w_te))
            RecallBGPlot_w_test.append(float(reBG_w_te))
            RecallGHPlot_w_test.append(float(reGH_w_te))
            RecallBHPlot_w_test.append(float(reBH_w_te))
            PseudoPGGPlot_test.append(float(pseudoPGG_te))
            PseudoRGPlot_test.append(float(pseudoRG_te))
            PseudoRHPlot_test.append(float(pseudoRH_te))
            RejectionNoneGG_test.append(float(rejNoneGG_te))
            RejectionNoneGG_test_.append(float(rejNoneGG_te_))
            SigRatio_test.append(float(sigRatio_te))

            print ( "Epoch: " + "{:3d}".format(epoch) + \
                    ",  Loss(Train&Test)= " + "{:.3f}".format(loss_tr) + " | " + "{:.3f}".format(loss_te) + \
                    ",  Accuracy(Train&Test)= " + "{:.3f}".format(acc_tr) + " | " + "{:.3f}".format(acc_te) )          

            if (epoch == training_epochs):
                final_indicators.append("Loss:             %.4f \n"%(loss_tr))
                final_indicators.append("Accuracy:         %.4f \n"%(acc_tr))
                final_indicators.append("Precision(Sig_w): %.4f \n"%(preGG_w_tr))
                final_indicators.append("Precision(Bkg_w): %.4f, %.4f, %.4f \n"%(preBG_w_tr,preGH_w_tr,preBH_w_tr))
                final_indicators.append("Recall(Sig_w):    %.4f \n"%(reGG_w_tr))
                final_indicators.append("Recall(Bkg_w):    %.4f, %.4f, %.4f \n"%(reBG_w_tr,reGH_w_tr,reBH_w_tr))
                final_indicators.append("pseudoPrecision:  %.4f \n"%(pseudoPGG_tr))
                final_indicators.append("Rejection:        %.4f \n"%(rejNoneGG_tr))
                final_indicators.append("SigRatio:         %.4f \n"%(sigRatio_tr))

    # Save trained model.
    save_path = saver.save(sess, modelPath)
    print("Model saved in file: %s"%(save_path))

    # save trained variables
    variables_w = []
    variables_b = []
    for i in range(len(variable_w)):
        # dictionary values doesn't return to a list directly
        variables_w.append( list(variable_w.values())[i].eval() )
        variables_b.append( list(variable_b.values())[i].eval() )
    print ("all weights: ",variables_w)
    print ("all biases: ",variables_b)

    '''
    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    list(variable_w.values())[0].eval().tofile('testfile_w.txt', sep=', ')
    list(variable_w.values())[1].eval().tofile('testfile_w.txt', sep=', ')
    list(variable_b.values())[0].eval().tofile('testfile_b.txt', sep=', ')
    list(variable_b.values())[1].eval().tofile('testfile_b.txt', sep=', ')
    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    '''

    print("Optimization Finished!------------------------------------------")
    #############################################################################
    


print ( "Plotting Results ==============================================================================" )
# Plot loss & accuracy separately, plot all precisions in one, plot all recalls in one.
# Plot pseudoPrecision, preudoRecall separately.
print ( "Plotting Loss -----------------------" )
fig1 = plt.figure(1, figsize=(12,8), dpi=200, facecolor='red', edgecolor='blue')
plt.plot(DisplayStep, LossPlot_train, linestyle='-', marker='.', markersize=6, color='red', alpha=1.0, label='Training Result')
plt.plot(DisplayStep, LossPlot_test, linestyle='-', marker='.', markersize=6, color='blue', alpha=1.0, label='Testing Result')
plt.ylim([0,50])
plt.xlabel('Epoch Number', fontsize=20)
plt.ylabel('Loss', fontsize=20)
plt.title("Loss vs Epoch", fontsize=30)
matplotlib.rc('font', size=18)
plt.tick_params(labelsize=20)
plt.legend(loc=1)
fig1.savefig('%s/%s_loss.png'%(out_directory_name,args.name))

print ( "Plotting Accuracy -------------------" )
fig2 = plt.figure(2, figsize=(12,8), dpi=200, facecolor='red', edgecolor='blue')
plt.plot(DisplayStep, AccuracyPlot_train, linestyle='-', marker='.', markersize=6, color='red', alpha=1.0, label='Training Result')
plt.plot(DisplayStep, AccuracyPlot_test, linestyle='-', marker='.', markersize=6, color='blue', alpha=1.0, label='Testing Result')
plt.ylim([0,1])
plt.xlabel('Epoch Number', fontsize=20)
plt.ylabel('Accuracy', fontsize=20)
plt.title("Accuracy vs Epoch", fontsize=30)
matplotlib.rc('font', size=18)
plt.tick_params(labelsize=20)
plt.legend(loc=4)
fig2.savefig('%s/%s_accuracy.png'%(out_directory_name,args.name))

print ( "Plotting Precision ------------------" )
fig3 = plt.figure(3, figsize=(12,8), dpi=200, facecolor='red', edgecolor='blue')
plt.plot(DisplayStep, PrecisionGGPlot_train, linestyle='-', marker='.', markersize=6, color='red', alpha=1.0, label='Training (Good Gamma)')
plt.plot(DisplayStep, PrecisionBGPlot_train, linestyle='-', marker='.', markersize=6, color='magenta', alpha=1.0, label='Training (Bad Gamma)')
plt.plot(DisplayStep, PrecisionGHPlot_train, linestyle='-', marker='.', markersize=6, color='blue', alpha=1.0, label='Training (Good Hadron)')
plt.plot(DisplayStep, PrecisionBHPlot_train, linestyle='-', marker='.', markersize=6, color='green', alpha=1.0, label='Training (Bad Hadron)')
plt.plot(DisplayStep, PrecisionGGPlot_test, linestyle='-', marker='.', markersize=6, color='red', alpha=0.2, label='Testing (Good Gamma)')
plt.plot(DisplayStep, PrecisionBGPlot_test, linestyle='-', marker='.', markersize=6, color='magenta', alpha=0.2, label='Testing (Bad Gamma)')
plt.plot(DisplayStep, PrecisionGHPlot_test, linestyle='-', marker='.', markersize=6, color='blue', alpha=0.2, label='Testing (Good Hadron)')
plt.plot(DisplayStep, PrecisionBHPlot_test, linestyle='-', marker='.', markersize=6, color='green', alpha=0.2, label='Testing (Bad Hadron)')
plt.ylim([0,1])
plt.xlabel('Epoch Number', fontsize=20)
plt.ylabel('Precision', fontsize=20)
plt.title("Precision vs Epoch", fontsize=30)
matplotlib.rc('font', size=18)
plt.tick_params(labelsize=20)
plt.legend(loc=4)
fig3.savefig('%s/%s_precision.png'%(out_directory_name,args.name))

print ( "Plotting Precision (weighted) -------" )
fig4 = plt.figure(4, figsize=(12,8), dpi=200, facecolor='red', edgecolor='blue')
plt.plot(DisplayStep, PrecisionGGPlot_w_train, linestyle='-', marker='.', markersize=6, color='red', alpha=1.0, label='Training (Good Gamma)')
plt.plot(DisplayStep, PrecisionBGPlot_w_train, linestyle='-', marker='.', markersize=6, color='magenta', alpha=1.0, label='Training (Bad Gamma)')
plt.plot(DisplayStep, PrecisionGHPlot_w_train, linestyle='-', marker='.', markersize=6, color='blue', alpha=1.0, label='Training (Good Hadron)')
plt.plot(DisplayStep, PrecisionBHPlot_w_train, linestyle='-', marker='.', markersize=6, color='green', alpha=1.0, label='Training (Bad Hadron)')
plt.plot(DisplayStep, PrecisionGGPlot_w_test, linestyle='-', marker='.', markersize=6, color='red', alpha=0.2, label='Testing (Good Gamma)')
plt.plot(DisplayStep, PrecisionBGPlot_w_test, linestyle='-', marker='.', markersize=6, color='magenta', alpha=0.2, label='Testing (Bad Gamma)')
plt.plot(DisplayStep, PrecisionGHPlot_w_test, linestyle='-', marker='.', markersize=6, color='blue', alpha=0.2, label='Testing (Good Hadron)')
plt.plot(DisplayStep, PrecisionBHPlot_w_test, linestyle='-', marker='.', markersize=6, color='green', alpha=0.2, label='Testing (Bad Hadron)')
plt.ylim([0,1])
plt.xlabel('Epoch Number', fontsize=20)
plt.ylabel('Precision(weighted)', fontsize=20)
plt.title("Precision(weighted) vs Epoch", fontsize=30)
matplotlib.rc('font', size=18)
plt.tick_params(labelsize=20)
plt.legend(loc=4)
fig4.savefig('%s/%s_precision_w.png'%(out_directory_name,args.name))

print ( "Plotting Recall ---------------------" )
fig5 = plt.figure(5, figsize=(12,8), dpi=200, facecolor='red', edgecolor='blue')
plt.plot(DisplayStep, RecallGGPlot_train, linestyle='-', marker='.', markersize=6, color='red', alpha=1.0, label='Training (Good Gamma)')
plt.plot(DisplayStep, RecallBGPlot_train, linestyle='-', marker='.', markersize=6, color='magenta', alpha=1.0, label='Training (Bad Gamma)')
plt.plot(DisplayStep, RecallGHPlot_train, linestyle='-', marker='.', markersize=6, color='blue', alpha=1.0, label='Training (Good Hadron)')
plt.plot(DisplayStep, RecallBHPlot_train, linestyle='-', marker='.', markersize=6, color='green', alpha=1.0, label='Training (Bad Hadron)')
plt.plot(DisplayStep, RecallGGPlot_test, linestyle='-', marker='.', markersize=6, color='red', alpha=0.2, label='Testing (Good Gamma)')
plt.plot(DisplayStep, RecallBGPlot_test, linestyle='-', marker='.', markersize=6, color='magenta', alpha=0.2, label='Testing (Bad Gamma)')
plt.plot(DisplayStep, RecallGHPlot_test, linestyle='-', marker='.', markersize=6, color='blue', alpha=0.2, label='Testing (Good Hadron)')
plt.plot(DisplayStep, RecallBHPlot_test, linestyle='-', marker='.', markersize=6, color='green', alpha=0.2, label='Testing (Bad Hadron)')
plt.ylim([0,1])
plt.xlabel('Epoch Number', fontsize=20)
plt.ylabel('Recall', fontsize=20)
plt.title("Recall vs Epoch", fontsize=30)
matplotlib.rc('font', size=18)
plt.tick_params(labelsize=20)
plt.legend(loc=4)
fig5.savefig('%s/%s_recall.png'%(out_directory_name,args.name))

print ( "Plotting Recall (weighted) ----------" )
fig6 = plt.figure(6, figsize=(12,8), dpi=200, facecolor='red', edgecolor='blue')
plt.plot(DisplayStep, RecallGGPlot_w_train, linestyle='-', marker='.', markersize=6, color='red', alpha=1.0, label='Training (Good Gamma)')
plt.plot(DisplayStep, RecallBGPlot_w_train, linestyle='-', marker='.', markersize=6, color='magenta', alpha=1.0, label='Training (Bad Gamma)')
plt.plot(DisplayStep, RecallGHPlot_w_train, linestyle='-', marker='.', markersize=6, color='blue', alpha=1.0, label='Training (Good Hadron)')
plt.plot(DisplayStep, RecallBHPlot_w_train, linestyle='-', marker='.', markersize=6, color='green', alpha=1.0, label='Training (Bad Hadron)')
plt.plot(DisplayStep, RecallGGPlot_w_test, linestyle='-', marker='.', markersize=6, color='red', alpha=0.2, label='Testing (Good Gamma)')
plt.plot(DisplayStep, RecallBGPlot_w_test, linestyle='-', marker='.', markersize=6, color='magenta', alpha=0.2, label='Testing (Bad Gamma)')
plt.plot(DisplayStep, RecallGHPlot_w_test, linestyle='-', marker='.', markersize=6, color='blue', alpha=0.2, label='Testing (Good Hadron)')
plt.plot(DisplayStep, RecallBHPlot_w_test, linestyle='-', marker='.', markersize=6, color='green', alpha=0.2, label='Testing (Bad Hadron)')
plt.ylim([0,1])
plt.xlabel('Epoch Number', fontsize=20)
plt.ylabel('Recall(weighted)', fontsize=20)
plt.title("Recall(weighted) vs Epoch", fontsize=30)
matplotlib.rc('font', size=18)
plt.tick_params(labelsize=20)
plt.legend(loc=4)
fig6.savefig('%s/%s_recall_w.png'%(out_directory_name,args.name))

print ( "Plotting pseudoPrecision ------------" )
fig7 = plt.figure(7, figsize=(12,8), dpi=200, facecolor='red', edgecolor='blue')
plt.plot(DisplayStep, PseudoPGGPlot_train, linestyle='-', marker='.', markersize=6, color='red', alpha=1.0, label='Training Result')
plt.plot(DisplayStep, PseudoPGGPlot_test, linestyle='-', marker='.', markersize=6, color='blue', alpha=1.0, label='Testing Result')
#plt.ylim([0,100])
plt.xlabel('Epoch Number', fontsize=20)
plt.ylabel('pseudoPrecision Good Gamma', fontsize=20)
plt.title("(trueGamma labeled as GoodGamma) / sqrt(trueHadron labeled as GoodGamma)", fontsize=20)
matplotlib.rc('font', size=18)
plt.tick_params(labelsize=20)
plt.legend(loc=4)
fig7.savefig('%s/%s_pseudoPrecisionGG.png'%(out_directory_name,args.name))

print ( "Plotting pseudoRecall Gamma ---------" )
fig8 = plt.figure(8, figsize=(12,8), dpi=200, facecolor='red', edgecolor='blue')
plt.plot(DisplayStep, PseudoRGPlot_train, linestyle='-', marker='.', markersize=6, color='red', alpha=1.0, label='Training Result')
plt.plot(DisplayStep, PseudoRGPlot_test, linestyle='-', marker='.', markersize=6, color='blue', alpha=1.0, label='Testing Result')
plt.ylim([0,1])
plt.xlabel('Epoch Number', fontsize=20)
plt.ylabel('pseudoRecall Gamma', fontsize=20)
plt.title("(trueGamma labeled as GoodGamma) / (trueGamma)", fontsize=20)
matplotlib.rc('font', size=18)
plt.tick_params(labelsize=20)
plt.legend(loc=1)
fig8.savefig('%s/%s_pseudoRecallG.png'%(out_directory_name,args.name))

print ( "Plotting pseudoRecall Hadron --------" )
fig9 = plt.figure(9, figsize=(12,8), dpi=200, facecolor='red', edgecolor='blue')
plt.plot(DisplayStep, PseudoRHPlot_train, linestyle='-', marker='.', markersize=6, color='red', alpha=1.0, label='Training Result')
plt.plot(DisplayStep, PseudoRHPlot_test, linestyle='-', marker='.', markersize=6, color='blue', alpha=1.0, label='Testing Result')
plt.ylim([0,1])
plt.xlabel('Epoch Number', fontsize=20)
plt.ylabel('pseudoRecall Hadron', fontsize=20)
plt.title("(trueHadron labeled as GoodGamma) / (trueHadron)", fontsize=20)
matplotlib.rc('font', size=18)
plt.tick_params(labelsize=20)
plt.legend(loc=1)
fig9.savefig('%s/%s_pseudoRecallH.png'%(out_directory_name,args.name))

print ( "Plotting Rejection GoodGamma --------" )
fig10 = plt.figure(10, figsize=(12,8), dpi=200, facecolor='red', edgecolor='blue')
plt.plot(DisplayStep, RejectionNoneGG_train, linestyle='-', marker='.', markersize=6, color='red', alpha=1.0, label='Training Result')
plt.plot(DisplayStep, RejectionNoneGG_test, linestyle='-', marker='.', markersize=6, color='blue', alpha=1.0, label='Testing Result')
plt.yscale('log')
plt.ylim([0.9,1])
plt.xlabel('Epoch Number', fontsize=20)
plt.ylabel('Rejection', fontsize=20)
plt.title("Rejection (removing non-GoodGamma)", fontsize=30)
matplotlib.rc('font', size=18)
plt.tick_params(labelsize=20)
plt.legend(loc=4)
fig10.savefig('%s/%s_rejection.png'%(out_directory_name,args.name))

print ( "Plotting 1 - Rejection (GoodGamma) --" )
fig11 = plt.figure(11, figsize=(12,8), dpi=200, facecolor='red', edgecolor='blue')
plt.plot(DisplayStep, RejectionNoneGG_train_, linestyle='-', marker='.', markersize=6, color='red', alpha=1.0, label='Training Result')
plt.plot(DisplayStep, RejectionNoneGG_test_, linestyle='-', marker='.', markersize=6, color='blue', alpha=1.0, label='Testing Result')
plt.yscale('log')
plt.ylim([0,0.1])
plt.xlabel('Epoch Number', fontsize=20)
plt.ylabel('1 - Rejection', fontsize=20)
plt.title("1 - Rejection (Keeping BKG)", fontsize=30)
matplotlib.rc('font', size=18)
plt.tick_params(labelsize=20)
plt.legend(loc=1)
fig11.savefig('%s/%s_rejection_.png'%(out_directory_name,args.name))

print ( "Plotting Significance Ratio ---------" )
fig12 = plt.figure(12, figsize=(12,8), dpi=200, facecolor='red', edgecolor='blue')
plt.plot(DisplayStep, SigRatio_train, linestyle='-', marker='.', markersize=6, color='red', alpha=1.0, label='Training Result')
plt.plot(DisplayStep, SigRatio_test, linestyle='-', marker='.', markersize=6, color='blue', alpha=1.0, label='Testing Result')
#plt.yscale('log')
plt.ylim([0,10])
plt.xlabel('Epoch Number', fontsize=20)
plt.ylabel('Significance Ratio', fontsize=20)
plt.title("recall(signal) / sqrt(1-rejection)", fontsize=30)
matplotlib.rc('font', size=18)
plt.tick_params(labelsize=20)
plt.legend(loc=4)
fig12.savefig('%s/%s_sigRatio.png'%(out_directory_name,args.name))

#plt.show()
plt.close()



print ( "Write all indicators to text file =============================================================" )
indicators_text = open('%s/indicators_bin_%s_hl_%s.txt'%(out_directory_name,str(args.bin),str(args.hiddenlayer)),'w')
for i in range(len(final_indicators)):
    indicators_text.write(final_indicators[i])
indicators_text.close()



print ( "Write all variables to text file ==============================================================" )
output_text = open('%s/model_bin_%s_hl_%s.txt'%(out_directory_name,str(args.bin),str(args.hiddenlayer)),'w')
for i in range(len(variables_w)):
    output_text.write(str(variables_w[i]))
    output_text.write('\n')
output_text.write('\n')
for i in range(len(variables_b)):
    output_text.write(str(variables_b[i]))
    output_text.write('\n')
output_text.close()



print ( "Write all variables to text file ==============================================================" )
'''
# One way to write xml file, the format is not pretty.
root_xml = ET.Element("model")
tree = ET.ElementTree(root_xml)
weight_xml = ET.SubElement(root_xml, "weight")
for i in range(len(variables_w)):
    ET.SubElement(weight_xml, 'weight_%d'%(i), name='weight_%d'%(i)).text = '   ' + str(variables_w[i]) + '\n'
bias_xml = ET.SubElement(root_xml, "bias")
for i in range(len(variables_b)):
    ET.SubElement(bias_xml, 'bias_%d'%(i), name='bias_%d'%(i)).text = '    ' + str(variables_b[i]) + '\n'
tree.write('%s/model_bin_%s_hl_%s.xml'%(out_directory_name,str(args.bin),str(args.hiddenlayer)))
'''
# The way to write a pretty xml file.
def indent(elem, level=0):
    i = "\n" + level*"  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            indent(elem, level+1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i

'''
# two levels
def WriteXML():
    model = ET.Element("model")
 
    weight = ET.SubElement(model, "weight")
    weight.set( "Type", "%s"%(int(args.hiddenlayer)+1) )
    for i in range(len(variables_w)):
        ET.SubElement(weight, "weight_%d"%(i)).text = str(variables_w[i])
 
    bias = ET.SubElement(model, "bias")
    bias.set( "Type", "%s"%(int(args.hiddenlayer)+1) )
    for i in range(len(variables_b)):
        ET.SubElement(bias, "bias_%d"%(i)).text = str(variables_b[i])

    indent(model)
    tree = ET.ElementTree(model)
    tree.write('%s/model_bin_%s_hl_%s.xml'%(out_directory_name,str(args.bin),str(args.hiddenlayer)), xml_declaration=False, encoding='utf-8', method="xml")
'''
# One level
def WriteXML():
    model = ET.Element("model")
 
    for i in range(len(variables_w)):
        ET.SubElement(model, "weight_%d"%(i)).text = str(variables_w[i])
 
    for i in range(len(variables_b)):
        ET.SubElement(model, "bias_%d"%(i)).text = str(variables_b[i])

    indent(model)
    tree = ET.ElementTree(model)
    tree.write('%s/model_bin_%s_hl_%s.xml'%(out_directory_name,str(args.bin),str(args.hiddenlayer)), xml_declaration=False, encoding='utf-8', method="xml")


WriteXML()



# Total execution time.
print( "Total execution time:  %s seconds -----------" % (time.time() - start_time) )

