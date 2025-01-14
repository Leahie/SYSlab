import csv
import numpy as np 
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import pickle as pkl
import random

import os

# First attempt it with one file
file1 = "./data/hw_dataset/parkinson/P_02100002.txt"
control = "./data/hw_dataset/control/C_000"
file2 = "./data/new_dataset/parkinson"
new_data = "G:/My Drive/Sys Lab/modified_data/new"
x0, y0 = 200.0, 204.0
xmin, xmax = 32.0, 401.25

parkinsons_1= "./data/hw_dataset/parkinson/"
subset = []
def get_subset(X, y, filename): 
    #  format ([Xi, X2. . . Xf], [yi, y2, . . . yf])
    X_temp, y_temp = [], []
    for i in tqdm(range(len(X)-50)): # was len(X)
        # X_temp.append(X[i])
        # y_temp.append(y[i])
        if i<=50:
            continue
        combined_array = np.column_stack((X[i:i+50], y[i:i+50]))
        angle = random.randint(1, 360)
        theta = np.radians(angle)  # Convert the angle to radians
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], 
                                    [np.sin(theta), np.cos(theta)]])
        rotated = np.dot(combined_array, rotation_matrix)
        shape = 50
        X_all = combined_array[:, 0 ] # This is for X data and x coordinate value
        Y_all = combined_array[:, 1 ]
        subset.append((filename, i, angle, X_all, Y_all, shape))
    # df = pd.DataFrame(subset, columns=['Filename', 'index', 'angle', 'X', 'y', 'length'])
    # df.to_pickle(f"{new_data}/modified_{filename.split('/')[-1]}.pkl")
    return subset

# def get_subset(X, y, filename):
#     subset = [] # format ([Xi, X2. . . Xf], [yi, y2, . . . yf])
#     for i in tqdm(range(len(X)-11)): # was len(X)
#         if i<=600:
#             continue
#         X_temp, y_temp = [], []
#         for j in range(i+11, len(X)):
#             X_temp.append(X[j])
#             y_temp.append(y[j])
#             combined_array = np.column_stack((X_temp, y_temp))
#             angle = random.randint(1, 360)
#             theta = np.radians(angle)  # Convert the angle to radians
#             rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], 
#                                         [np.sin(theta), np.cos(theta)]])
#             rotated = np.dot(combined_array, rotation_matrix)
#             shape = j-i
#             subset.append((filename, i, angle, rotated[:-1, :-1], rotated[-1, -1], shape))
#     df = pd.DataFrame(subset, columns=['Filename', 'index', 'angle', 'X', 'y', 'length'])
#     df.to_pickle(f"{new_data}/modified_{filename}.pkl")
#     print(subset)

#     return subset


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

        if len(df_stat):
            
            X, y, z, c, t = df_stat[:,0], df_stat[:,1], df_stat[:,2], df_stat[:, 3], df_stat[:, 4]
            print(t.size)
            
            X = X * (xmax-xmin)/(X.max()-X.min())
            X = X- (X[0]-x0)
            y = y- (y[0]-y0)
            c = (c-c.min())/(c.max()-c.min())
            c= c.astype(str)
        
            
            plt.plot(X, y)
            return(filename, X,y) # Just to get the values of the control 
            get_subset(X, y, filename)

def visualize_subsets(filepath):
    with open(filepath, 'rb') as f:
        classification_dict = pkl.load(f)
        print(classification_dict)

#visualize_subsets(f"{new_data}/dummy.pkl")

parkinsons_1 = "data/hw_dataset/parkinson"
parkinsons_2 = "data/new_dataset/parkinson"
control = "data/hw_dataset/control"

# ls = os.listdir(parkinsons_1)
# for i in tqdm(ls):
     
#     read_data(f"./{parkinsons_1}/{i}", label="Parkinson's")

ls2 = os.listdir(parkinsons_2)   
for i in tqdm(ls2):
    read_data(f"./{parkinsons_2}/{i}", label="Parkinson's")

# ls2 = os.listdir(control)   
# for i in tqdm(ls2):
#     read_data(f"./{parkinsons_2}/{i}", label="Parkinson's")

# df = pd.DataFrame(subset, columns=['Filename', 'index', 'angle', 'X', 'y', 'length'])
# df.to_pickle(f"{new_data}/modified_all.pkl")
subset=[]
for i in range(1,10):
    control = "data/hw_dataset/control/C_000"
    arr = read_data(f"./{control}{i}.txt", label=f"control{i}")
    subset.append(arr)
# df = pd.DataFrame(subset, columns=['Filename', 'X', 'y']) # For the intent of making the controls into a pickle file
# df.to_pickle(f"./data/controls.pkl")

print(xmin/8, xmax/8)

plt.legend(loc="upper left")
plt.show()