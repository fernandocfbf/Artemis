#--IMPORTS MODELO SVM E DATA SCIENCE--#
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import f1_score, recall_score
from sklearn.metrics import confusion_matrix
import numpy as np

#--FUNÇÕES EXTERNAS A PÁGINA--#
dataset = pd.DataFrame(pd.read_csv('creditcard.csv'))
dataset = dataset.sort_values(by = 'Class', ascending = False)
d = dataset[['V3', 'V4', 'V9', 'V10', 'V11', 'V12', 'V14', 'V16', 'V17', 'Class']].iloc[0:1000, :]

#Criando uma seed para mantermos um padrão nos dataset de treino e teste
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

#Sepanrando o dataset apenas com as features necessária para a modelagem
X = d.iloc[:, 0:9]
y = d.iloc[:, 9]

#Separando as bases de dados para teste e treino
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)

#Aplicando o modelo SVM (Support Vector Machine)
model = svm.SVC(kernel = 'linear', gamma = 'scale') #Kernel

#Dando fit no modelo SVM usando o dataset de treino
model.fit(X_train, y_train)

#Criando a lista de predições do modelo
predict = model.predict(X_test)

#--IMPORTS FLASK--#
from flask import Flask, render_template, send_from_directory, request

app = Flask(__name__)

@app.route('/media/<path:filename>')
def page_send_file(filename):
    return send_from_directory('media', filename)

@app.route('/')
def page_index():
    return render_template("index.html", f1 = f1_score(y_test, predict) * 100,
                                         recall = recall_score(y_test, predict) * 100,
                                         v3min = round(d['V3'].min(), 2), v3max = round(d['V3'].max(), 2),
                                         v4min = round(d['V4'].min(), 2), v4max = round(d['V4'].max(), 2),
                                         v9min = round(d['V9'].min(), 2), v9max = round(d['V9'].max(), 2),
                                         v10min = round(d['V10'].min(), 2), v10max = round(d['V10'].max(), 2),
                                         v11min = round(d['V11'].min(), 2), v11max = round(d['V11'].max(), 2),
                                         v12min = round(d['V12'].min(), 2), v12max = round(d['V12'].max(), 2),
                                         v14min = round(d['V14'].min(), 2), v14max = round(d['V14'].max(), 2),
                                         v16min = round(d['V16'].min(), 2), v16max = round(d['V16'].max(), 2),
                                         v17min = round(d['V17'].min(), 2), v17max = round(d['V17'].max(), 2),
                                         predict_value = "No prediction to be shown")

@app.route('/about')
def page_value():
    return "<p>There's no about to be shown</p>"

@app.route('/', methods = ['GET', 'POST'])
def page_predict():
    prediction = "No prediction to be shown"
    X = [request.form['V3'] , request.form['V4'] , request.form['V9'] ,
         request.form['V10'], request.form['V11'], request.form['V12'],
         request.form['V14'], request.form['V16'], request.form['V17']]
    #X = [0, 0, 0]
    if not all(x==X[0] for x  in X):
        X = [X]
        p = model.predict(X)
        if p[0] == 1:
            prediction = "False"
        elif p[0] == 0:
            prediction = "Legit"
    return render_template("index.html", f1 = f1_score(y_test, predict) * 100,
                                         recall = recall_score(y_test, predict) * 100,
                                         v3min = round(d['V3'].min(), 2), v3max = round(d['V3'].max(), 2),
                                         v4min = round(d['V4'].min(), 2), v4max = round(d['V4'].max(), 2),
                                         v9min = round(d['V9'].min(), 2), v9max = round(d['V9'].max(), 2),
                                         v10min = round(d['V10'].min(), 2), v10max = round(d['V10'].max(), 2),
                                         v11min = round(d['V11'].min(), 2), v11max = round(d['V11'].max(), 2),
                                         v12min = round(d['V12'].min(), 2), v12max = round(d['V12'].max(), 2),
                                         v14min = round(d['V14'].min(), 2), v14max = round(d['V14'].max(), 2),
                                         v16min = round(d['V16'].min(), 2), v16max = round(d['V16'].max(), 2),
                                         v17min = round(d['V17'].min(), 2), v17max = round(d['V17'].max(), 2),
                                         predict_value = prediction)
if __name__ == '__main__':
    app.debug = True
    app.run(host = '0.0.0.0', port = 25565)