import numpy as np
from keras.models import Sequential
from keras.layers import *
from keras.callbacks import EarlyStopping
from keras.src.layers import Dense
from matplotlib import pyplot as plt

x_train = np.load('binary_data/horse_human_x_train.npy')
y_train = np.load('binary_data/horse_human_y_train.npy')
x_test = np.load('binary_data/horse_human_x_test.npy')
y_test = np.load('binary_data/horse_human_y_test.npy')
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

model =Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), padding='same',
                 activation='relu', input_shape=(64, 64, 3)))
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
model.add(Conv2D(32, kernel_size=(3, 3), padding='same',
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
model.add(Conv2D(32, kernel_size=(3, 3), padding='same',
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()


model.compile(loss='binary_crossentropy',
              optimizer='adam', metrics=['binary_accuracy'])
early_stopping = EarlyStopping(monitor='val_binary_accuracy', patience=7)
#7회폭 이상 좋아지지 않으면 stopping으로 끝내겠다
fit_hist = model.fit(x_train, y_train, batch_size=32, epochs=100,
                     validation_data=(x_test, y_test), callbacks=[early_stopping])
score = model.evaluate(x_test, y_test)
print('Evaluation loss :', score[0])
print('Evaluation accuracy :', score[1])
model.save('./models/horse_human_model_{}.h5'.format(np.around(score[1], 3)))
plt.plot(fit_hist.history['binary_accuracy'])
plt.plot(fit_hist.history['val_binary_accuracy'])
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.plot(fit_hist.history['loss'])
plt.plot(fit_hist.history['val_loss'])
plt.legend(['train', 'test'], loc='upper left')
plt.show()







