import numpy as np
import pickle
import random

class RNNModel:
    def __init__(self, arc, lam=0.0001, epoch=500):
        self.wl, self.ws, self.biases = self.initialize_weights(arc)
        self.lam = lam
        self.epoch = epoch

    def soft_max(self, dot):
        temp = np.exp(dot[-1][-1])
        a_N_F = temp / np.sum(temp)
        return a_N_F

    def sum_(self, arr, l, s):
        summ = arr[l][1].copy()
        for i in range(2, s+1):
            summ += arr[l][i] 
        return summ

    def sum_l(self, arr, a, l, s):
        summ = arr[l][1].copy()@(a[l-1][1].T).copy()
        for i in range(2, s+1):
            temp = arr[l][i]@(a[l-1][i].T)
            summ += temp
        return summ

    def sum_s(self, arr, a, l, s):
        summ = arr[l][2].copy()@(a[l][2-1].T).copy()
        for i in range(3, s+1):
            summ += arr[l][i]@(a[l][i-1].T)
        return summ

    def initialize_weights(self, arc):
        wl = [None]
        ws = [None]
        biases = [None]
        for i in range(len(arc)-1):
            curr, next = arc[i+1], arc[i]
            temp = (3/((2*curr + next)/2))**.5

            wl.append(temp * 2 * np.random.rand(curr, next) - temp)
            ws.append(temp * 2 * np.random.rand(curr, curr) - temp)
            biases.append(temp * 2 * np.random.rand(curr, 1) - temp)
        
        return wl, ws[:3], biases

    def rnn_deriv(self, x):
        return 1/((np.cosh(x))**2)

    def backprop_rnn2(self, A, grad_A, train, test, one_hot):
        N = 2    
        index = 0
        for _ in range(self.epoch):
            for val in train:
                if index % 10000 == 0: 
                    self.store_data("save_" + str(index) + ".pkl", self.ws, self.wl, self.biases)
                    acc, e = self.test_func(np.tanh, test, one_hot)
                    print("Index", index, "Epoch", _, ": MSE:", e, "ACC:", acc)
                index += 1

                a = [[None]]
                dot = [[None]]
                grad = {}
                x, y = val[0:20], one_hot[val[20:]]
                for l in range(1, N+1): 
                    a.append([np.zeros((np.shape(self.ws[l])[1], 1))])
                    dot.append([None])  

                # Forward Prop
                F = len(x)
                for s in range(1, F+1):
                    vec = one_hot[x[s-1]]
                    a[0].append(vec)  
                    for l in range(1, N+1):
                        temp = self.wl[l] @ a[l-1][s] + self.ws[l] @ a[l][s-1] + self.biases[l]
                        dot[l].append(temp) 
                        a[l].append(A(dot[l][s]))

                dot.append((self.wl[3] @ a[2][F]) + self.biases[3])
                a.append(np.exp(dot[3]) / np.sum(np.exp(dot[3])))
                grad[3] = (np.eye(y.shape[0]) - a[3]) @ y
                grad[N] = {}
                grad[N][F] = grad_A(dot[2][F]) * (self.wl[3].T @ grad[3])

                # Backward Propagation
                for s in range(F-1, 0, -1):
                    grad[N][s] = grad_A(dot[N][s]) * (self.ws[N].T @ grad[N][s+1])
                for i in range(N-1, 0, -1):
                    grad[i] = {}
                    grad[i][F] = grad_A(dot[i][F]) * (self.wl[i+1].T @ grad[i+1][F])
                    for s in range(F-1, 0, -1):
                        grad[i][s] = grad_A(dot[i][s]) * (self.ws[i].T @ grad[i][s+1]) + grad_A(dot[i][s]) * (self.wl[i+1].T @ grad[i+1][s])

                # Update Weights
                self.wl[3] += self.lam * (grad[3] @ a[2][F].T)
                self.biases[3] += self.lam * grad[3]
                for i in range(1, N+1):
                    self.biases[i] += self.lam * self.sum_(grad, i, F)       
                    self.wl[i] += self.lam * self.sum_l(grad, a, i, F)
                    self.ws[i] += self.lam * self.sum_s(grad, a, i, F)

            acc, e = self.test_func(np.tanh, test, one_hot)
            print("Epoch", _, ": MSE:", e, "ACC:", acc)

        return self.wl, self.ws, self.biases

    def test_func(self, step, test, one_hot):
        N, F = 2, 50
        total = 0
        correct = 0
        error_all = 0
        for val in test:
            a = [[None]]
            dot = [[None]]
            x, y = val[0:20], one_hot[val[20:]]
            for l in range(1, N+1):  
                a.append([np.zeros((np.shape(self.ws[l])[1], 1))])
                dot.append([None])  

            for s in range(1, len(x)+1):
                vec = one_hot[x[s-1]]
                a[0].append(vec)  
                for l in range(1, N+1):
                    temp = self.wl[l] @ a[l-1][s] + self.ws[l] @ a[l][s-1] + self.biases[l]
                    dot[l].append(temp) 
                    a[l].append(step(dot[l][s]))

            dot.append((self.wl[3] @ a[2][F]) + self.biases[3])
            temp = np.exp(dot[3]/.7)
            a.append(temp/np.sum(temp))
            ind = np.argmax(a[N+1])
            correct += (ind == list(y).index(1))
            total += 1

            error_all += np.sum(-y * np.log(a[N+1]))

        return correct / total, error_all / total

    def store_data(self, file, ws, wl, b):
        data = {'ws': ws, 'wl': wl, 'b': b}
        with open(file, 'wb') as f:
            pickle.dump(data, f)

    def load_data(self, file):
        with open(file, 'rb') as f:
            return pickle.load(f)

    def generate(self, train_file, model_file, arc=[39, 40, 40, 39]):
        data = self.load_data(train_file)
        print(f"Training samples: {len(data['train'])}")
        self.wl, self.ws, self.biases = self.load_data(model_file)
        self.backprop_rnn2(np.tanh, self.rnn_deriv, data['train'], data['test'], data['one_hot'])

    def debug(self, data_file, model_file):
        data = self.load_data(data_file)
        model_data = self.load_data(model_file)
        print(self.test_func(np.tanh, data['test'], model_data['ws'], model_data['b'], data['one_hot']))
