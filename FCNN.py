%matplotlib inline
# Imports
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix
from keras.models import Sequential 
import matplotlib.pyplot as plt
from keras.layers import Dense
from tensorflow import keras
import tensorflow as tf
import seaborn as sns
import pandas as pd
import numpy as np
import os

# le o arquivo csv
def load_data(csv_path):
 return pd.read_csv(csv_path)

# Caminho do arquivo e leitura
df = load_data(r"C:\Users\.... .csv")
df = df.fillna(0)

# Preparacao para normalizar os dados com Standard Scale
y = df.iloc[:, 94:95].values
x = df.drop(['ID'], axis = 1)
X = x.drop(['Grupo'], axis = 1)

# Normalizando os dados
sc = StandardScaler() 
X = sc.fit_transform(X)
ohe = OneHotEncoder()
y =ohe.fit_transform(y).toarray()

# Separando os dados em teste e treino
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3, random_state=42)

# Gerando Modelo
model = Sequential () 
model.add (Dense (64, input_shape=(93,), activation = 'relu')) 
model.add (Dense (64, activation = 'relu'))
model.add (Dense (64, activation = 'relu'))
model.add (Dense (64, activation = 'relu'))
model.add (Dense (64, activation = 'relu'))
model.add (Dense (64, activation = 'relu'))
model.add (Dense (64, activation = 'relu'))
model.add (Dense (64, activation = 'relu'))
model.add (Dense (64, activation = 'relu'))
model.add (Dense (64, activation = 'relu'))
model.add (Dense (64, activation = 'relu'))
model.add (Dense (64, activation = 'relu'))
model.add (Dense (64, activation = 'relu'))
model.add (Dense (2, activation = 'softmax'))

# Compilando
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# Visualizacao do modelo
model.summary()

# Numero de epocas e tamanho do grupo de treinamento
epochs= 700
batch_size = 32

history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))
model.save("model.h5")

# Mostra a acuracia em relacao ao teste
test_loss, test_acc = model.evaluate(X_test, y_test)
print('Acuracia teste:', test_acc)

#  Acuracia no dataset de treino e o de validacao (provineintes do split), respectivamente, durante as epocas.
history_dict = history.history
val_acc_values = history_dict['val_categorical_accuracy']
epochs = range(1, len(val_acc_values) + 1, 100)
val_1 = list()
val_2= list()
for i in range(1, len(history.history["categorical_accuracy"]) + 1, 100):
    val_1.append(history.history["categorical_accuracy"][i])
    val_2.append(history.history["val_categorical_accuracy"][i])

plt.title("Acurácia")
plt.plot(epochs, val_1, label="acuráca")
plt.plot(epochs, val_2, label="acuráca em validação")
plt.xlabel("Épocas")
plt.ylabel("Porcentagem(%)")
plt.legend()
plt.show()

# Perdas no dataset de treino e o de validacao durante as epocas.
val_1 = list()
val_2= list()
for i in range(1, len(history.history["loss"]) + 1, 100):
    val_1.append(history.history["loss"][i])
    val_2.append(history.history["val_categorical_accuracy"][i])
plt.title("Perda")
plt.plot(epochs, val_1, label="Perda")
plt.plot(epochs, val_2, label="Perda em validação")
plt.xlabel("Épocas")
plt.ylabel("Porcentagem(%)")
plt.legend()
plt.show()

# Confusion matrix
predictions = model.predict(X_test)
conf_matrix= confusion_matrix(np.argmax(y_test, axis=1), np.argmax(predictions, axis=1))
sns.heatmap(conf_matrix, annot = True,fmt='',cbar=False)

# Outras metricas Precision, recall e f1-score
print(classification_report(np.argmax(y_test, axis=1), np.argmax(predictions, axis=1)))