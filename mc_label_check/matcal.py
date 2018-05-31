from __future__ import print_function
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# calculation for bin0 event


modelFile = open("chosen_models_small_LR/model_bin_2_hl_1.txt", "r")

def IsFloat(s):
    try: 
        float(s)
        return True
    except ValueError:
        return False

def getVariable(modelFile):
    var_list = [ [], [], [], [] ]
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
    w1 = []
    b0 = var_list[2]
    b1 = var_list[3]
    for i in range(40):
        w0.append([])
        for j in range(22):
            a = 22*i + j
            w0[i].append(var_list[0][a])
    for i in range(22):
        w1.append([])
        for j in range(4):
            a = 4*i + j
            w1[i].append(var_list[1][a])

    return w0, w1, b0, b1

w0, w1, b0, b1 = getVariable(modelFile)
#print (w0, len(w0))
#print (w1, len(w1))
#print (b0, len(b0))
#print (b1, len(b1))


features = [0.318841, 0.13747, 0.262004, 0.0215, 0.554645, 0.517475, 0.758389, 0.552632, 0.433333, 0.0124699, 0.0168043, 0.0182726, 0.0182726, 0.13, 0.178252, 0, 0, 0.220264, 0, 0.573333, 0, 0, 0.17284, 0.117063, 0.5, 0.0616114, 0, 0.134021, 0.0208333, 0.0520833, 0.229167, 0.1875, 0.0319149, 0, 0.0315789, 0.62458, 0.0126063, 0.563013, 0.660547, 0.783067]


res0 = np.matmul(features, w0)
test = res0
print ("--------------")
print ("w0: ", test)

res0 = np.add(res0, b0)
test = res0
print ("--------------")
print ("b0: ", test)

for i in range(22):
    if res0[i]<0:
        res0[i] = 0
test = res0
print ("--------------")
print ("reLU: ", test)

res1 = np.matmul(res0, w1)
test = res1
print ("--------------")
print ("w1: ", test)

res1 = np.add(res1, b1)
test = res1
print ("--------------")
print ("b1: ", test)

exp_sum = 0
for i in range(4):
    res1[i] = np.exp(res1[i])
    exp_sum = exp_sum + res1[i]
for i in range(4):
    res1[i] = res1[i] / exp_sum

test = res1
print ("--------------")
print ("softmax: ", test)

