
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers.recurrent import LSTM
from keras.layers.recurrent import SimpleRNN
from keras.layers.merge import concatenate
from matplotlib import pyplot
import numpy as np


# Shared Input Layer
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.utils import plot_model
from keras.layers.merge import average


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

print(Y_Test[len(file)-12000-11-1])
print(file[len(file)-1])



# define input
visible = Input(shape=(11,8))
# feature extraction
extract1 = SimpleRNN(46,return_sequences=False)(visible)

extract2=SimpleRNN(32,return_sequences=False)(visible)

extract3=SimpleRNN(16,return_sequences=False)(visible)
# first interpretation model

merge1 = concatenate([extract1, extract2])

merge = concatenate([merge1, extract3])

output = Dense(1, activation='sigmoid')(merge)


model = Model(inputs=visible, outputs=output)


model.compile(loss = 'mean_squared_error', optimizer = 'RMSprop', metrics = ['accuracy'])

print(model.summary())


history = model.fit([X_Train], Y_Train, validation_data=(X_Valid, Y_Valid),verbose=2, epochs=100)


scores = model.evaluate(X_Test, Y_Test, verbose=0)



training_loss = history.history['loss']
test_loss = history.history['val_loss']
epoch_count = range(1, len(training_loss)+1)
pyplot.plot(epoch_count, training_loss)
pyplot.plot(epoch_count, test_loss)
pyplot.ylabel('Loss')
pyplot.xlabel('Epochs')
pyplot.legend(['Train', 'Test'], loc='upper right')
pyplot.show()



trainPredict = model.predict(X_Test)

fig, ax = pyplot.subplots(figsize=(17,8))
ax.set_title('Prediction vs. Actual after 100 epochs of training')
ax.plot(Y_Test[:300,], label='True Data', color='green', linewidth='3')
ax.plot(trainPredict[:300,], label='Prediction', color='red', linewidth='2')
pyplot.legend()
pyplot.show()
