import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_hub as hub

from sklearn.model_selection import train_test_split

#def df_to_dataset (dataframe, shuffle=True, batch_size=1024):



df = pd.read_csv("enron-emails-bag2.csv") 

#df.dropna(subset=["deva","altceva"])
#label = []

#for i in range(len(df.columns)):
#    label.append(df.columns[i])

#print (label)
print(df.head())



X = df[df.columns[:-1]].values
Y = df[df.columns[-1]].values

#print (X)
#print(Y)

X_train, X_temp, Y_train, Y_temp = train_test_split(X,Y,test_size=0.4,random_state=0)
X_valid, X_test, Y_valid, Y_test = train_test_split(X_temp,Y_temp,test_size=0.5,random_state=0)

#from sklearn.preprocessing import LabelEncoder
#lb = LabelEncoder()
#Y = lb.fit_transform(Y)


model = tf.keras.models.Sequential ([
    tf.keras.layers.Dense(16,activation='relu'),
    tf.keras.layers.Dense(16,activation='relu'),
    tf.keras.layers.Dense(16,activation='softmax'),
    tf.keras.layers.Dense(1,activation="sigmoid"),
])



model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
loss=tf.keras.losses.BinaryCrossentropy(),
metrics=['accuracy'])


model.evaluate(X_valid, Y_valid)


#plt.hist(df.points, bins=20)
#plt.title("Points Histogram")
#...
#