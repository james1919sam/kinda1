# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 19:35:39 2018

@author: Vinay M HARITSA
@ vtricks technologies (gurukulam vidya)
@ for any queries contact : 9620749749
@CREDITS : STACKK ABUSE
@ steps are as : 
@FEED FORWARD AND BACK PROPOGATION
@ FEED FOWARD HAS 1)DOT PRODUCT OF INPUT AND WEIGHTS 
@2)PASS RESULT FROM STEP 1 TO ACIVATION FUNCTION(SIGMOID FUNCTION)
@BACK PROPAGATION HAS 2 STEPS
@STEP 1 : CALCULATE COST
@STEP 2 : MINIMIZE COST (GAUSS THEOREM)
Person	Smoking	Obesity	Exercise	Diabetic
Person 1	0	1	0	1
Person 2	0	0	1	0
Person 3	1	0	0	0
Person 4	1	1	0	1
Person 5	1	1	1	1

"""

## importing numpy package
import numpy as np  

# creating the dataset mentioned above
feature_set = np.array([[0,1,0],[0,0,1],[1,0,0],[1,1,0],[1,1,1]])  

#creating labels 
labels = np.array([[1,0,0,1,1]])  
labels = labels.reshape(5,1)  

# creating 3 paramerters for forward network they are weights,bias and learning rate
np.random.seed(42)  
weights = np.random.rand(3,1)  
bias = np.random.rand(1)  
lr = 0.0

# creating a sigmoid function for forward network
def sigmoid(x):  
    return 1/(1+np.exp(-x))


# creating the derivative of sigmoid
def sigmoid_der(x):  
    return sigmoid(x)*(1-sigmoid(x))


# now we are trying to run feed forward nework will all the above parameters
    # epoch 20000 is number of iterations
for epoch in range(20000):  
    inputs = feature_set

    # feedforward step1
    XW = np.dot(feature_set, weights) + bias

    #feedforward step2
    z = sigmoid(XW)


    # backpropagation step 1
    error = z - labels

    print(error.sum())

    # backpropagation step 2
    dcost_dpred = error
    dpred_dz = sigmoid_der(z)

  
    z_delta = dcost_dpred * dpred_dz

    inputs = feature_set.T
    weights -= lr * np.dot(inputs, z_delta)

    for num in z_delta:
        bias -= lr * num
        

XW = np.dot(feature_set, weights) + bias 


z = sigmoid(XW) 


error = z - labels  

dcost_dpred = error 

dpred_dz = sigmoid_der(z)


#slope = input x dcost_dpred x dpred_dz


z_delta = dcost_dpred * dpred_dz  
inputs = feature_set.T  
weights -= lr * np.dot(inputs, z_delta)  
    