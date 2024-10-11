import csv
import numpy as np 
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import pickle as pkl

import os

# First attempt it with one file
file1 = "./data/hw_dataset/parkinson/P_02100002.txt"
control = "./data/hw_dataset/control/C_000"
file2 = "./data/new_dataset/parkinson"
new_data = "./modified_data"
x0, y0 = 200.0, 204.0
xmin, xmax = 32.0, 401.25

parkinsons_1= "./data/hw_dataset/parkinson/"

def get_subset(X, y, filename):
    subset = [] # format ([Xi, X2. . . Xf], [yi, y2, . . . yf])
    for i in tqdm(range(len(X)-11)): # was len(X)
        X_temp, y_temp = [], []
        for j in range(i+11, len(X)):
            X_temp.append(X[j])
            y_temp.append(y[j])
            subset.append((filename, i, X_temp[:], y_temp[:]))
        if i!=0 and i%100==0:
            df = pd.DataFrame(subset, columns=['Filename', 'index', 'X', 'y'])
            df.to_pickle(f"{new_data}/value_{i}.pkl")
    df = pd.DataFrame(subset, columns=['Filename', 'index', 'X', 'y'])
    df.to_pickle(f"{new_data}/value_{i}.pkl")
    print(subset)

    return subset


def read_data(filename, label="name"): 
    with open(filename) as source:
        reader = csv.reader(source, delimiter=';')
        x = []
        # X ; Y; Z; Pressure; GripAngle; Timestamp; Test ID
        # 9 second intervals --> 
        # 0: Static Spiral Test ( Draw on the given spiral pattern)
        # 1: Dynamic Spiral Test ( Spiral pattern will blink in a certain time, so subjects need to continue on their draw)
        # 2: Circular Motion Test (Subjectd draw circles around the red point)
        for line in reader:
            x.append(line)
        df = np.array(x)
        df = df.astype(float)
        cond1 = (np.array([df[:, 2]])).T
        
        # Split based on Test 
        cond = (np.array([df[:, -1]])).T
        df_stat = df[cond[:,0]==0]
        df_dyn = df[cond[:,0]==1]
        df_circ = df[cond[:,0]==2]

        
        X, y, z, c, t = df_stat[:,0], df_stat[:,1], df_stat[:,2], df_stat[:, 3], df_stat[:, 4]
        print(t.size)
        
        X = X * (xmax-xmin)/(X.max()-X.min())
        X = X- (X[0]-x0)
        y = y- (y[0]-y0)
        c = (c-c.min())/(c.max()-c.min())
        c= c.astype(str)
        
        
        #plt.scatter(X, y, c=c, cmap='viridis')
        get_subset(X, y, filename)

def visualize_subsets(filepath):
    with open(filepath, 'rb') as f:
        classification_dict = pkl.load(f)
        print(classification_dict)

#visualize_subsets(f"{new_data}/dummy.pkl")

parkinsons_1 = "./data/new_dataset/parkinson/"
ls = os.listdir(parkinsons_1)
for i in ls[0:1]:
    file1 = f"data/new_dataset/parkinson/{i}"
    read_data(f"./{file1}", label="Parkinson's")

# for i in range(1,10):
#     control = "data/hw_dataset/control/C_000"
#     read_data(f"./{control}{i}.txt", label=f"control{i}")

# print(xmin/8, xmax/8)

# plt.legend(loc="upper left")
# plt.show()