import numpy as np
import matplotlib.pyplot as plt
import pickle

class RNNModel:
    def __init__(self, model_file="", lam=0.0001, epoch=500):
        self.lam = lam
        self.model_file = model_file
        self.start = ""
        if self.model_file != "":
            self.start = int(model_file.split("_")[-1][:-4])
            val = self.load_data(self.model_file)
            print(val.keys())
            self.wl, self.ws, self.biases = val['wl'], val['ws'], val['b']
        print(self.start)
        self.epoch = epoch

    def soft_max(self, dot):
        temp = np.exp(dot[-1][-1])
        a_N_F = temp / np.sum(temp)
        return a_N_F

    def sum_(self, arr, l, s):
        summ = arr[l][1].copy()
        for i in range(2, s + 1):
            summ += arr[l][i]
        return summ

    def sum_l(self, arr, a, l, s):
        print(f"arr[l][1].shape: {arr[l][1].shape}")
        print(f"a[l-1][1].T.shape: {a[l-1][1].T.shape}")
        summ = arr[l][1].copy() @ (a[l-1][1].T).copy()
        for i in range(2, s + 1):
            temp = arr[l][i] @ (a[l-1][i].T)
            summ += temp
        return summ

    def sum_s(self, arr, a, l, s):
        print(f"arr[l][2].shape: {arr[l][2].shape}")
        print(f"a[l][2-1].T.shape: {a[l][2-1].T.shape}")
        summ = arr[l][2].copy() @ (a[l][2-1].T).copy()
        for i in range(3, s + 1):
            summ += arr[l][i] @ (a[l][i-1].T)
        return summ

    def initialize_weights(self, arc, n_features):
        wl, ws, biases = [None], [None], [None]
        for i in range(len(arc) - 1):
            curr, next = arc[i + 1], arc[i]
            if i == 0:
                next = n_features  # Override for the first layer's input size
            temp = (3 / ((2 * next + curr) / 2)) ** 0.5
            wl.append(temp * 2 * np.random.rand(curr, next) - temp)
            ws.append(temp * 2 * np.random.rand(curr, curr) - temp)
            biases.append(temp * 2 * np.random.rand(curr, 2) - temp)
        return wl, ws, biases

    def rnn_deriv(self, x):
        return 1 / ((np.cosh(x)) ** 2)

    def error(self, x, y):
        return np.sum(np.square(y - x))

    def backprop_rnn2(self, A, grad_A, train, test):
        N = 3
        index = 0
        thresh = int(len(train) * 10)
        for _ in range(self.epoch):
            for val in train:
                if index % thresh == 0 and index != 0:
                    self.store_data(f"model/y_minmax/y_save_{self.start + index}.pkl", self.ws, self.wl, self.biases)
                index += 1

                a = [[None]]
                dot = [[None]]
                grad = {}
                x, y = val[0:49], val[49:]
                for l in range(1, N + 1):
                    a.append([np.zeros((np.shape(self.ws[l])[1], 2))])
                    dot.append([None])

                # Forward Prop
                F = len(x)
                for s in range(1, F + 1):
                    a[0].append(np.array(x[s - 1]).reshape(2, 1))
                    for l in range(1, N + 1):
                        temp = self.wl[l] @ a[l - 1][s] + self.ws[l] @ a[l][s - 1] + self.biases[l]
                        dot[l].append(temp)
                        a[l].append(A(dot[l][s]))

            fin = a[N][F]
            grad[N] = {}
            grad[N][F] = grad_A(dot[N][F]) * (y - a[N][F])
            for s in range(F - 1, 0, -1):
                grad[N][s] = grad_A(dot[N][s]) * (self.ws[N].T @ grad[N][s + 1])
            for i in range(N - 1, 0, -1):
                grad[i] = {}
                grad[i][F] = grad_A(dot[i][F]) * (self.wl[i + 1].T @ grad[i + 1][F])

            for i in range(F - 1, 0, -1):
                for j in range(N - 1, 0, -1):
                    grad[j][i] = grad_A(dot[j][i]) * (self.ws[j].T @ grad[j][i + 1]) + grad_A(dot[j][i]) * (self.wl[j + 1].T @ grad[j + 1][i])
            for i in range(1, N + 1):
                self.biases[i] = self.biases[i] + self.lam * (self.sum_(grad, i, F))
                self.wl[i] = self.wl[i] + self.lam * (self.sum_l(grad, a, i, F))
                self.ws[i] = self.ws[i] + self.lam * (self.sum_s(grad, a, i, F))

            e = self.test_func(np.tanh, test, self.wl, self.ws, self.biases)
            print("Index", index, "Epoch", _, ": MSE:", e)

        return self.wl, self.ws, self.biases

    def test_func(self, step, test, wl, ws, b):
        N, F = 2, 49
        total_error = 0
        for val in test:
            a = [[None]]
            dot = [[None]]
            x, y = val[0:49], val[49:]
            for l in range(1, N + 1):
                a.append([np.zeros((np.shape(ws[l])[1], 1))])
                dot.append([None])

            for s in range(1, F + 1):
                a[0].append(np.array(x[s - 1]).reshape(-1, 1))
                for l in range(1, N + 1):
                    temp = self.wl[l] @ a[l - 1][s] + self.ws[l] @ a[l][s - 1] + self.biases[l]
                    dot[l].append(temp)
                    a[l].append(step(dot[l][s]))

            e = self.error(a[N][F], y)
            if np.isnan(e):
                e
            total_error += e
        return total_error / len(test)

    def predict(self, val, A=np.tanh):
        N = 2
        index = 0
        a = [[None]]
        dot = [[None]]
        grad = {}
        x, y = val[0:49], val[49:]
        for l in range(1, N + 1):
            a.append([np.zeros((np.shape(self.ws[l])[1], 1))])
            dot.append([None])

        # Forward Prop
        F = len(x)
        for s in range(1, F + 1):
            a[0].append(np.array(x[s - 1]).reshape(-1, 1))
            for l in range(1, N + 1):
                temp = self.wl[l] @ a[l - 1][s] + self.ws[l] @ a[l][s - 1] + self.biases[l]
                dot[l].append(temp)
                a[l].append(A(dot[l][s]))

        return a[N][F]



    def graph(self, val):
        y_pred = self.predict(val)

        plt.plot(range(len(val)), val, label='x values (training)', marker='o')
        plt.plot(range(len(val),len(val)+1), y_pred, label='prediction', marker='x')
        plt.show()
        

    def store_data(self, file, ws, wl, b):
        data = {'ws': ws, 'wl': wl, 'b': b}
        with open(file, 'wb') as f:
            pickle.dump(data, f)

    def load_data(self, file):
        with open(file, 'rb') as f:
            return pickle.load(f)

    def generate(self, data_file = "", arc=[1, 6, 1]):
        print(isinstance(self.model_file, str))
        if isinstance(data_file, str)==True: data = self.load_data(data_file)
        else: 
            data = data_file
        if self.model_file !="":
                val = self.load_data(self.model_file)
                self.wl, self.ws, self.biases = val['wl'], val['ws'], val['b']
        else: self.wl, self.ws, self.biases = self.initialize_weights(arc,2)
        self.backprop_rnn2(np.tanh, self.rnn_deriv, data['train'][:1000], data['test'][:500])

import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt
import tqdm as tqdm
import os 
print("here")
read = "G:/My Drive/Sys Lab/modified_data/new"
ls2 = os.listdir(read) 
base = "C:/Users/leahz/OneDrive/Desktop/Quizlet/ATC4/SYSlab"

model = RNNModel(lam=.001,  ) #model_file="./model/y_minmax/y_save_70000.pkl"

import numpy as np

# Generating a 2D array with 50 rows (samples) and 2 columns (features)
n_features = 2   # Number of features
n_samples = 50   # Number of data points (length)

data = {
    "train": [np.random.rand(50, 2) for _ in range(1000)],
    "test": [np.random.rand(50, 2) for _ in range(500)],
}


print("Sample Input Data (50 rows x 2 features):")
model.generate(data, arc=[2, 50,30,2])
#model.generate('data/data_y_minmax.pkl' , arc=[[2, 50, 30, 2]] )#./model/y/y_save_90000.pkl
        
# with open('data/data_y_minmax.pkl', 'rb') as f:
#     data =  pickle.load(f)
# model = RNNModel(model_file="./model/y_minmax/y_save_70000.pkl")
# model.graph(data['train'][100])