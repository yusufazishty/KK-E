# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 18:15:18 2016

@author: yusufazishty
"""

#Using dummy data, make model, classify
from sklearn import tree
import numpy as np
import copy
import math
from sklearn.externals.six import StringIO
import os
import pydot
from sklearn import datasets
from IPython.display import Image  

def transform(data_to_transform):
    set_value = sorted(list(set(data_to_transform)))
    dictionary={}
    for i in range(len(set_value)):
        dictionary[set_value[i]]=i
    #sebelum
    print(data_to_transform)
    for i in range(len(data_to_transform)):
        data_to_transform[i]=dictionary[data_to_transform[i]]
    print(data_to_transform)
    return data_to_transform
    
#def transform_num(data_to_transform):
#    set_value = sorted(list(set(data_to_transform)))
#    min_val = set_value[0]; max_val = set_value[-1]
#    step = int((max_val-min_val)/10)
#    steps=[]
#    for i in range(min_val+step, max_val+step, step):
#        steps.append(i)
#    print(data_to_transform)
#    for i in range(len(data_to_transform)):
#        for j in range(len(steps)):
#            if data_to_transform[i]<=steps[j]:
#                data_to_transform[i]=j
#                break                
#    print(data_to_transform)
#    return data_to_transform

def transform_num(data_to_transform):
    square = [ float(x)**2 for x in list(data_to_transform/1000)]
    square = math.sqrt(sum(square))
    data_to_transform2 = data_to_transform/square
    print(data_to_transform)
    print(data_to_transform2)
    return data_to_transform2

#Dummy
X=[[0,0], [1,1]]
Y=[0,1]
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X,Y)
clf.predict([[1,1],[0,0],[-2,-1],[2,2],[3,3]])
#the result give insight that data <=0 class==0 and vice versa

#let's dealt with real data
#read data
aha = datasets.load_iris()
data_loc = "E:\\UserTA\\5112100086\\Dropbox\\[PENTING TIDAK URGENT]\\[ARSIP KULIAH]\\Semester 9\\Asistensi KK E\\datakelas.txt"
with open(data_loc,"r") as f:
    data=[]
    for x in f.readlines():
        temp = x.strip().split("\t")
        data.append(temp)
data = np.array(data)
names = data[0,:]; data=data[1:,:]
m_record, n_feature = np.shape(data)
data_transformed = copy.deepcopy(data)

#transform to discrete, except continuous bring to int
exception=[]
for i in range(n_feature):
    if i in exception:
        data_to_transform = data[:,i].astype(int)
        data_transformed[:,i] = transform_num(data_to_transform)
    else:
        data_to_transform = data[:,i]
        data_transformed[:,i] = transform(data_to_transform)
        
Y = data[:,-1]
X = data[:,:n_feature-1] 
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X,Y)
#dot_data = StringIO()  
tree.export_graphviz(clf, out_file='tree2.dot', feature_names=names)
#graph = pydot.graph_from_dot_data(dot_data.getvalue())  
#Image(graph.create_png())

#f = open("tree.dot", "r")
#with open("tree.dot", "rt") as f:
#    dot_data = f.readlines()
#dot_data = ''.join(dot_data)
#dot_data = StringIO()
#tree.export_graphviz(clf, out_file=dot_data,feature_names=names[:-1])  
#with open("tree2.dot", 'w') as f:
#     f = tree.export_graphviz(clf, out_file=f)
#graph = pydot.graph_from_dot_data(dot_data.getvalue())  
#Image(graph.create_png())  
#os.unlink('tree.dot')       


#digraph Tree {
#0 [label="outlook <= 0.5000\ngini = 0.459183673469\nsamples = 14", shape="box"] ;
#1 [label="gini = 0.0000\nsamples = 4\nvalue = [ 0.  4.]", shape="box"] ;
#0 -> 1 ;
#2 [label="humidity <= 0.5000\ngini = 0.5\nsamples = 10", shape="box"] ;
#0 -> 2 ;
#3 [label="outlook <= 1.5000\ngini = 0.32\nsamples = 5", shape="box"] ;
#2 -> 3 ;
#4 [label="wind <= 0.5000\ngini = 0.5\nsamples = 2", shape="box"] ;
#3 -> 4 ;
#5 [label="gini = 0.0000\nsamples = 1\nvalue = [ 1.  0.]", shape="box"] ;
#4 -> 5 ;
#6 [label="gini = 0.0000\nsamples = 1\nvalue = [ 0.  1.]", shape="box"] ;
#4 -> 6 ;
#7 [label="gini = 0.0000\nsamples = 3\nvalue = [ 3.  0.]", shape="box"] ;
#3 -> 7 ;
#8 [label="wind <= 0.5000\ngini = 0.32\nsamples = 5", shape="box"] ;
#2 -> 8 ;
#9 [label="outlook <= 1.5000\ngini = 0.5\nsamples = 2", shape="box"] ;
#8 -> 9 ;
#10 [label="gini = 0.0000\nsamples = 1\nvalue = [ 1.  0.]", shape="box"] ;
#9 -> 10 ;
#11 [label="gini = 0.0000\nsamples = 1\nvalue = [ 0.  1.]", shape="box"] ;
#9 -> 11 ;
#12 [label="gini = 0.0000\nsamples = 3\nvalue = [ 0.  3.]", shape="box"] ;
#8 -> 12 ;
#}



        