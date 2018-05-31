import argparse
import os.path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim
from torch.autograd import Variable
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets


print ( "Parse arguments ===============================================================================" )
parser = argparse.ArgumentParser(description='Pytorch training')

parser.add_argument('--no-augment', dest='augment', action='store_false',
                    help='whether to use standard augmentation (default: True)')
parser.add_argument('--name', type=str, default='Test',
                    help='name of experiment')
parser.add_argument('-eps', '--epochs', type=int, default=300, 
                    help='number of total epochs to run')
parser.add_argument('-b', '--batchSize', type=int, default=64, 
                    help='mini-batch size (default: 64)')
parser.add_argument('-lr', '--learningRate', type=float, default=0.1, 
                    help='initial learning rate')
parser.add_argument('-p', '--printFreq', type=int, default=10,
                    help='print frequency (default: 10)')

parser.set_defaults(augment=True)
args = parser.parse_args()


print ( "Define customized class/function ==============================================================" )
dtype = torch.FloatTensor # set data type to be float tensor (32 bits)
class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        # In the constructor we instantiate two nn.Linear modules and assign them as
        # member variables.
        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)
    def forward(self, x):
        # In the forward function we accept a Variable of input data and we must return
        # a Variable of output data. We can use Modules defined in the constructor as
        # well as arbitrary operators on Variables.
        h_relu = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(h_relu)
        return y_pred



print ( "Load data ==================================================================================== " )
hdf_train = pd.HDFStore('data/training_sweets_dec19_noise_MPF_allParticle_bin_0.h5','r')
#hdf_test = pd.HDFStore('data/testing_sweets_dec19_noise_MPF_allParticle_bin_0.h5','r')
# store the HDF5 dataframe object in memory (this would cause an error if the file is too large)
df_train = hdf_train["training"]



print ( "Process classes ============================================================================== " )
df_train['pclass0'] = df_train['pclass'].map(lambda x: 1 if (x==0) else 0)
df_train['pclass1'] = df_train['pclass'].map(lambda x: 1 if (x==1) else 0)
df_train['pclass2'] = df_train['pclass'].map(lambda x: 1 if (x==2) else 0)
df_train['pclass3'] = df_train['pclass'].map(lambda x: 1 if (x==3) else 0)
df_train_class = df_train[['pclass0','pclass1','pclass2','pclass3']]
df_train_weight = df_train['TWgt']
print ("Training classes (DataFrame): ", df_train_class.head())
print ("Training weights (DataFrame): ", df_train_weight.head())



print ( "Define feature groups =========================================================================" )
features = ['zenithAngle', 'coreFiduScale', 'nHit', 'nHitSP10', 'CxPE40', 'PINC', 'SFCFChi2']
#features = ['zenithAngle', 'coreFiduScale', 'nHit', 'nHitSP10', 'CxPE40', 'PINC', 'SFCFChi2', 'compactness', 'nHit10ratio', 'nHitRatio']
#features = ['zenithAngle', 'coreFiduScale', 'nHit', 'nHitSP10', 'CxPE40', 'PINC', 'SFCFChi2', 'nHitSP20', 'GamCoreAge', 'numPoints', 'scandelCore', 'numSum', 'scanedFrac', 'fixedFrac', 'avePE', 'disMax', 'CxPE40SPTime', 'fAnnulusCharge0', 'fAnnulusCharge1', 'fAnnulusCharge2', 'fAnnulusCharge3']
#features = ['zenithAngle', 'coreFiduScale', 'nHit', 'nHitSP10', 'CxPE40', 'PINC', 'SFCFChi2', 'nHitSP20', 'GamCoreAge', 'numPoints', 'scandelCore', 'numSum', 'scanedFrac', 'fixedFrac', 'avePE', 'disMax', 'CxPE40SPTime', 'fAnnulusCharge0', 'fAnnulusCharge1', 'fAnnulusCharge2', 'fAnnulusCharge3', 'compactness', 'nHit10ratio', 'nHitRatio']
# only do feature engineering on features (not features_plus)
features_plus = features +  ['signal', 'pclass', 'TWgt']



print ( "Process features ============================================================================= " )
df_train_feature = df_train
df_train_feature = df_train_feature[features]
df_train_feature['bias'] = float(1)
print ("Training features (DataFrame): ", df_train_feature.head())



print ( "Create input/output layers ====================================================================" )
# convert DataFrame to array, then convert array to tensor, then convert Tensor to Variable
array_train_feature = np.array(df_train_feature)
array_train_class = np.array(df_train_class)
array_train_weight = np.array(df_train_weight)
tensor_train_feature =  torch.Tensor(array_train_feature) 
tensor_train_class =  torch.Tensor(array_train_class)
tensor_train_weight =  torch.Tensor(array_train_weight)
train_feature = Variable(tensor_train_feature.type(dtype), requires_grad=False)
train_class = Variable(tensor_train_class.type(dtype), requires_grad=False)
train_weight = Variable(tensor_train_weight.type(dtype), requires_grad=False)
print ("Training features (Variable): ", train_feature)
print ("Training classes (Variable): ", train_class)
print ("Training weights (Variable): ", train_weight)



print ( "Define the NN architecture ====================================================================" )
# D_in is input layer dimension, D_h0 is first hidden layer dimension, D_out is output layer dimension.
bSize, D_in, D_h0, D_out = train_feature.size()[0], train_feature.size()[1], train_feature.size()[1], train_class.size()[1]
print ("Batch Size: ", bSize)
print ("In_Layer dim: ", D_in)
print ("H0_Layer dim: ", D_h0)
print ("Out_Layer dim: ", D_out)



print ( "Define the model ==============================================================================" )
###########################################################################################################
# bSize is the event number, "InNodes" is feature number, "OutNodes" is class number                     ##
##     Input_layer          weight0            Hidden0             weight1          Output_layer         ##
##   [bSize*InNodes]   [InNodes*H0Nodes]   [bSize*H0Nodes]   [H0Nodes*OutNodes]   [bSize*OutNodes]       ##
# initialize the weight tensors                                                                          ## 
#weight0 = Variable(torch.randn(D_in, D_h0).type(dtype), requires_grad=True)                             ##
#weight1 = Variable(torch.randn(D_h0, D_out).type(dtype), requires_grad=True)                            ##
#print ("Weight0: ", weight0)                                                                            ##
#print ("Weight1: ", weight1)                                                                            ##
###########################################################################################################
# a high level way to define NN model (construct model by instantiating self-defined class)
model = TwoLayerNet(D_in, D_h0, D_out)



print ( "Construct loss function and optimization ======================================================" )
learning_rate = args.learningRate
print ("Learning Rate: ", learning_rate)
# -----------------------------------------------------------------
criterion = torch.nn.MultiLabelSoftMarginLoss() # multi-label one-vs-all max entropy
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
#scheduler = torch.optim.StepLR(optimizer, step_size=30, gamma=0.1)



print ( "Run optimization for multiple epochs ==========================================================" )
epochs = args.epochs
print ("Epochs: ", epochs)
lossMean = []
for epoch in range(epochs):
    # adjust learning rate dynamically
    #scheduler.step()
    # forward pass: compute prediction by passing data to the model
    train_pred = model(train_feature)
    # compute and print loss
    loss = criterion(train_pred, train_class)
    print("Epoch [%4d/%4d] Loss: %4f"%(epoch,epochs,loss.data[0]))
    # zero gradients before backpropagation
    optimizer.zero_grad()
    # backpropagation, this will accumulate the gradients for each parameter
    # backpropagation is only possible after each forward pass
    loss.backward()
    # update parameters based on current gradients
    optimizer.step()
    lossMean.append(loss.data.mean())

print ("The mean loss: %4f"%(np.mean(lossMean)))




