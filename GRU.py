from math import sqrt

import numpy as np
from keras.layers import Dense, Dropout
from keras.layers import GRU
from keras.models import Sequential
from matplotlib import pyplot
from sklearn.metrics import mean_squared_error
import time
import math

n = 12000
n1=9000
n2=3000
file = np.load('polution_dataSet.npy')

counter=0;
g=0;

file1 = np.load('polution_dataSet.npy')

t=1
size = len(file1)/t
file = np.zeros((math.ceil(size),8))
while(1):
    file[g] =file1[counter]
    counter += t
    g +=1
    if(counter >= len(file1)):
        break

print ("test size: ", len(file)-n)
a1 = file[:,0:8]
a2 = file[:,0]
b1 = np.zeros((11,8))
b2 = 0
X_Train = np.zeros((n1,11,8))
Y_Train = np.zeros((n1,1))
for  i in range(n1-12):
    for j in range(10):
        b1[j] = a1[j+i]
    b2 = a2[24 + i]
    X_Train[i] = b1
    Y_Train[i] = b2

X_Valid = np.zeros((n2,11,8))
Y_Valid = np.zeros((n2,1))
for  i in range(n2-12):
    idx =  i+n1
    for j in range(10):
        b1[j] = a1[j + idx]
    b2 = a2[11 + idx]
    X_Valid[i] = b1
    Y_Valid[i] = b2


X_Test = np.zeros((len(file)-n,11,8))
Y_Test = np.zeros((len(file)-n,1))
i=0
while  i + n < len(file)-12:
    idx = i+n
    for j in range(10):
        b1[j] = a1[j + idx]
    b2 = a2[11 + idx]
    X_Test[i] = b1
    Y_Test[i] = b2
    i+=1


start = time.time()

model = Sequential()
#model.add(GRU(16,input_shape=(24,8),return_sequences=True,activation='tanh',stateful=True,batch_input_shape=(1, 24, 8)))
#model.add(Dropout(0.3))
model.add(GRU(32,input_shape=(11,8), return_sequences=False,activation='tanh'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])#mean_squared_error   adam   RMSprop    loss mae
print(model.summary())
history = model.fit(X_Train, Y_Train,validation_data=(X_Valid, Y_Valid),verbose=2, epochs=50, shuffle=False)

# Final evaluation of the model
scores = model.evaluate(X_Test, Y_Test, verbose=2)

print("Accuracy: %.2f%%" % (scores[1]*100))

done = time.time()
elapsed = done - start
print("Time of execution:   ",elapsed)

pyplot.figure()
training_loss = history.history['loss']
test_loss = history.history['val_loss']
epoch_count = range(1, len(training_loss)+1)
pyplot.plot(epoch_count, training_loss)
pyplot.plot(epoch_count, test_loss)
pyplot.title('Loss')
pyplot.ylabel('Loss')
pyplot.xlabel('Epochs')
pyplot.legend(['Train', 'Test'], loc='upper right')
pyplot.show()


start = time.time()
trainPredict = model.predict(X_Test)
done = time.time()
elapsed = done - start
print("Time of execution:   ",elapsed)

fig, ax = pyplot.subplots(figsize=(17,8))
ax.set_title('Prediction vs. Actual values after 50 epochs of training in single layer GRU')
ax.plot(Y_Test[:300,], label='True Data', color='green', linewidth='3')
ax.plot(trainPredict[:300,], label='Prediction', color='red', linewidth='2')
pyplot.legend()
pyplot.show()