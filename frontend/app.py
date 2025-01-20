import base64
from io import BytesIO
from flask import Flask, render_template, request
from sklearn.model_selection import train_test_split
import numpy as np
from model_file import RNNModel
import pickle
import random 
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
from keras.src.legacy.saving import legacy_h5_format
from matplotlib.gridspec import GridSpec
import pandas as pd
import numpy as np
import json


base = "C:/Users/leahz/OneDrive/Desktop/Quizlet/ATC4/SYSlab"

# with open(f'{base}/data/data_all.pkl', 'rb') as f:
#     data =  pickle.load(f)
data = pd.read_pickle(f'{base}/data/data_all.pkl')
data_premade = pd.read_pickle(f'{base}/data/data_premade.pkl') # This is the data after being preprocessed has index not filename
data_original = pd.read_pickle(f'G:/My Drive/Sys Lab/modified_data/new/modified_all.pkl') # This is the orginal data, it has the filename
data_spirals = pd.read_pickle(f"{base}/data/data_original_spiral.pkl")

model_X = RNNModel(model_file=f"{base}/model/X/X_save_5340000.pkl")
leng = range(len(data))

model_Y = RNNModel(model_file=f"{base}/model/Y/y_save_270000.pkl")

model_premade = load_model(f'{base}/model/my_model.keras')
model_smoothing = load_model(f'{base}/model/smoothing.keras')

app = Flask(__name__)
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/Smooth')
def smooth():
    return render_template('ltsm.html', plot_url=plot_url, error=0 )
@app.route('/LTSM')
def LTSM():
    value = request.args.get('slider_value', 1)
    pers_input = request.args.get('personalized', "")

    plt.clf()
    buf = BytesIO()
    fig, axes = plt.subplots(2, 1, figsize=(24,15)) 
    gs = GridSpec(2, 1, height_ratios=[1, 2]) 
    if pers_input!="":
        data_find = data_original[data_original['Filename'].str.contains(pers_input, regex=False)] # bc I have to get the filename so we have to\
        samp = data_find.sample()
        ind = int(samp.index[0])
        print(type(ind), ind)
        print(data_premade['index'])
        val = data_premade.loc[data_premade['index'] == ind] 
    else: val = data_premade.sample() 
    
    print(val)
    X, y = list(val['X'].iloc[0]), list(val['y'].iloc[0])
    
    # PLOTTING THE PREDICTED VALUES
    axes_0 = fig.add_subplot(gs[0])
    axes_0.plot(X, y, label='values (testing)', marker='o')
    axes_0.plot(val['target'].iloc[0][0], val['target'].iloc[0][1], label='values (testing)', marker='o', color='g')
    
    x_pred = []
    y_pred = []
    for i in range(int(value)):
        together = list(zip(X[i:], y[i:]))
        together += zip(x_pred,y_pred)
        temp = model_premade.predict(np.array(together).reshape(1,49,2))
        x_pred.append(temp[0][0])
        y_pred.append(temp[0][1])

    axes_0.plot(x_pred, y_pred, label='values (testing)', marker='x', color='r')

    # mse = np.mean(([[x_pred[0],y_pred[0]]] - val['target'].iloc[0]) ** 2)

    axes_0.set_title('Values and Prediction')
    axes_0.legend()

    # PLOTTING LOCATION WITHIN SPIRAL
    axes_1 = fig.add_subplot(gs[1])
    row_num = val['index']
     
    original_row = data_original.iloc[row_num]
    fn = original_row['Filename'].iloc[0]

    spiral_row = data_spirals.loc[data_spirals['filename'] == fn]

    X, y = list(spiral_row['X'].iloc[0]), list(spiral_row['y'].iloc[0])
    X_orig, y_orig = original_row['X'].iloc[0], original_row['y'].iloc[0]
    axes_1.plot(X, y, label='values (testing)',  color = 'b')
    axes_1.plot(X_orig, y_orig, label='current location', color = 'r')
    axes_1.set_title('Original Spiral')
    axes_1.legend()


    plt.savefig(buf, format="png")
    buf.seek(0)
    
    plot_url = base64.b64encode(buf.getvalue()).decode('utf8')
    return render_template('ltsm.html', plot_url=plot_url, error=0 )
@app.route('/submit')
def submit():
    #data = request.args.get('txt') HOW TO GET INPUT
    value = request.args.get('slider_value', 1)
    print(value)
    plt.clf()
    buf = BytesIO()
    fig, axes = plt.subplots(3, 1, figsize=(24,15))
    val = data.sample()

    print(list(val['y'].iloc[0]))

    X= list(val['X'].iloc[0])
    y = list(val['y'].iloc[0])

    x_pred = model_X.predict(X[0:49])
    y_pred = model_Y.predict(y[0:49])

    axes[0].plot(range(len(X)), X, label='x values (testing)', marker='o')
    error = np.sum(np.square(x_pred- X[49]))
    axes[0].plot(range(len(X)-1,len(X)), X[49], label='x values (testing)', marker='o', color='g')
    axes[0].plot(range(len(X)-1,len(X)), x_pred, label='x values (testing)', marker='x', color='r')
    
    axes[0].set_title('X Values and Prediction')
    axes[0].legend()

    axes[1].plot(range(len(y)), y, label='y values (testing)', marker='o')
    error = np.sum(np.square(y_pred- y[49]))
    axes[1].plot(range(len(y)-1,len(y)), y[49], label='y values (testing)', marker='o', color='g')
    axes[1].plot(range(len(y)-1,len(y)), y_pred, label='y values (testing)', marker='x', color='r')

    axes[1].set_title('y Values and Prediction')
    axes[1].legend()

    axes[2].plot(X, y, label='values (testing)', marker='o')
    axes[2].plot(X[49], y[49], label='values (testing)', marker='o', color='g')
    axes[2].plot(x_pred, y_pred, label='values (testing)', marker='x', color='r')

    axes[2].set_title('Values and Prediction')
    axes[2].legend()
    plt.savefig(buf, format="png")
    buf.seek(0)
    
    plot_url = base64.b64encode(buf.getvalue()).decode('utf8')
    return render_template('submit.html', plot_url=plot_url, error=error)
@app.route('/draw')
def draw():
    return render_template('draw.html')

@app.route('/drawresults')
def drawresults():
    line_data = json.loads(request.args.get('line_data', np.array([])))
    hasParkinson = request.args.get('hasParkinson', False)
    if hasParkinson == "no": hasParkinson= False
    elif hasParkinson == "yes": hasParkinson = True
    line_data = np.array([[float(point["x"]), float(point["y"])] for point in line_data])
    # Turn line into line likely drawn by Parkinson's Patient using LTSM model 
    print(line_data.shape)
    innerX = line_data[:, 0]
    innerY = line_data[:, 1]
    meanX = np.mean(innerX)
    stdX = np.std(innerX)
    meanY = np.mean(innerY)
    stdY = np.std(innerY)
    line_data[:, 0] = (innerX - meanX) / stdX
    line_data[:, 1] = (innerY - meanY) / stdY
    print(line_data)
    if not hasParkinson:
        pred = model_premade.predict(line_data.reshape(1,line_data.shape[0],2))
        print(pred)
    # Smooth line using smoothing.keras
    
    return render_template("draw_results.html")