import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.layers import LSTM
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot
import time
from math import sqrt
from sklearn.preprocessing import MinMaxScaler


n = 12000
file = np.load('polution_dataSet.npy')


a1 = file[:,0:8]
a2 = file[:,0]
b1 = np.zeros((11,8))

train = 9000-11
valid = 3000-11

X_Valid = np.zeros((valid+1,11,8))
Y_Valid = np.zeros((valid+1,1))

X_Train = np.zeros((train+1,11,8))
Y_Train = np.zeros((train+1,1))

for i in range(train+1):
    for j in range(11):
        b1[j] = a1[j + i]
    X_Train[i] = b1
    Y_Train[i] = a2[11 + i]


for  i in range(valid + 1):
        iddx = i + 9000
        for j in range(11):
            b1[j] = a1[j+iddx]
        X_Valid[i] = b1
        Y_Valid[i] = a2[11 + iddx]


X_Test = np.zeros((len(file)-12000-11,11,8))
Y_Test = np.zeros((len(file)-12000-11,1))
for  i in range(len(file)-12000-11):
    idx = 12000 + i
    for j in range(11):
        b1[j] = a1[j + idx]

    X_Test[i] = b1
    Y_Test[i] = a2[11 + idx]

start = time.time()

model = Sequential()

model.add(LSTM(32,input_shape=(11,8),return_sequences=False, dropout=0.1, recurrent_dropout=0.1))
#model.add(Dropout(0.1))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
print(model.summary())
history = model.fit(X_Train, Y_Train, validation_data=(X_Valid, Y_Valid),verbose=2, epochs=100)

# Final evaluation of the model
scores = model.evaluate(X_Test, Y_Test, verbose=0)


print("Accuracy: %.2f%%" % (scores[1]*100))

done = time.time()
elapsed = done - start
print("Time of execution:   ",elapsed)



training_loss = history.history['loss']
test_loss = history.history['val_loss']
epoch_count = range(1, len(training_loss)+1)
pyplot.plot(epoch_count, training_loss)
pyplot.plot(epoch_count, test_loss)
pyplot.ylabel('Loss')
pyplot.xlabel('Epochs')
pyplot.legend(['Train', 'Test'], loc='upper right')
pyplot.show()

training_acc = history.history['accuracy']
test_acc = history.history['val_accuracy']
epoch_count = range(1, len(training_acc)+1)
pyplot.plot(epoch_count, training_acc)
pyplot.plot(epoch_count, test_acc)
pyplot.ylabel('Accuracy')
pyplot.xlabel('Epochs')
pyplot.legend(['Train', 'Test'], loc='upper right')
pyplot.show()


out1 = np.zeros(n)
out2 = np.zeros(n)


trainPredict = model.predict(X_Test)

fig, ax = pyplot.subplots(figsize=(17,8))
ax.set_title('Prediction vs. Actual after 100 epochs of training')
ax.plot(Y_Test[:300,], label='True Data', color='green', linewidth='3')
ax.plot(trainPredict[:300,], label='Prediction', color='red', linewidth='2')
pyplot.legend()
pyplot.show()
