#전처리를 해야하는 작업
from PIL import Image
import glob
import numpy as np
from sklearn.model_selection import train_test_split

import os
from PIL import Image

import numpy as np
from keras.models import Sequential
from keras.layers import *
from keras.callbacks import EarlyStopping
from keras.src.layers import Dense
from matplotlib import pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

from PIL import Image
from keras.models import load_model



# #1. PNG 이미지 jpg로 변경
input_dir = r'./imgs_horse_human/'  # 여기에 PNG 파일들이 들어 있는 폴더 경로를 입력하세요.
output_dir = r'./imgs_horse_human/'  # 변환된 JPG 파일을 저장할 폴더 경로를 입력하세요.
#
# # 출력 폴더가 없다면 생성
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
#
# # 디렉토리 내 PNG 파일들 찾기
for filename in os.listdir(input_dir):
    if filename.endswith('.png'):  # PNG 파일만 찾기
        img_path = os.path.join(input_dir, filename)
        try:
#             # PNG 파일 열기
            img = Image.open(img_path)
#
#             # JPG로 변환
            img = img.convert('RGB')  # PNG는 투명 배경을 가질 수 있으므로, RGB 모드로 변환
#
#             # 출력 파일 경로 설정
            output_path = os.path.join(output_dir, filename.replace('.png', '.jpg'))
#
#             # JPG로 저장
            img.save(output_path, 'JPG')  # JPG로 저장
            print(f"Converted {filename} to JPG.")
        except Exception as e:
            print(f"Error converting {filename}: {e}")
#


#2. 이미지 로드 및 전처리
img_dir = './imgs_horse_human/'  #이미지 통로 경로
categories = ['horse', 'human']
image_w = 64
image_h = 64

pixel = 64 * 64 * 3 #이미지 하나당 픽셀
X = []
Y = []
for idx, category in enumerate(categories):
    for i, img_path in enumerate(glob.glob(img_dir + category + '*.jpg')):
        try:
            img = Image.open(img_path)
            img = img.resize((image_w, image_h))
            img = np.array(img) #사진을 행렬로 바꿔라
            X.append(img)
            Y.append(idx)
            if i % 300 == 0: #작업이 잘 된고 있는지 확인하기 위해 300번째마다 표시해라!
                print(category, ':', img_path)
        except:
            print('error :', category, img_path) #실행을 하다가 에러가 나면 이걸 표시해라!=>에러가 나면 다음으로 넘어갈수 있게.
X = np.array(X)#행렬을 리스트타입으로 다시 바꿔라
Y = np.array(Y)
X = X /255
print(X[0])
print(Y[0])
print(X.shape)
print(Y.shape)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
np.save('binary_data/horse_human_x_train.npy', X_train)
np.save('binary_data/horse_human_y_train.npy', Y_train)
np.save('binary_data/horse_human_x_test.npy', X_test)
np.save('binary_data/horse_human_y_test.npy', Y_test)

#3.말과 사람 이미지 분류 모델을 딥러닝, 평가
#ㄱ데이터 로드
x_train = np.load('binary_data/horse_human_x_train.npy')
y_train = np.load('binary_data/horse_human_y_train.npy')
x_test = np.load('binary_data/horse_human_x_test.npy')
y_test = np.load('binary_data/horse_human_y_test.npy')
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
#
#
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
)

datagen.fit(x_train)



#모델 생성 (CNN모델)
# model =Sequential()
# model.add(Conv2D(32, kernel_size=(3, 3), padding='same',
#                  activation='relu', input_shape=(64, 64, 3)))
# model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
# model.add(Conv2D(32, kernel_size=(3, 3), padding='same',
#                  activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
# model.add(Conv2D(32, kernel_size=(3, 3), padding='same',
#                  activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
#
# #플래튼+Dense 레이어 추가
# model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# model.add(Dense(1, activation='sigmoid')) #말인지 아닌지 분류/2는 말인지 사람인지 분류
# model.summary()

#ㄱ model = Sequential()
model.add(Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu', input_shape=(64, 64, 3)))
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
model.add(Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
model.add(Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#모델 컴파일
# model.compile(loss='binary_crossentropy',
#               optimizer='adam', metrics=['binary_accuracy'])
#  model.compile(loss='binary_crossentropy',
#               optimizer=Adam(learning_rate=0.0001)
               # ,metrics=['binary_accuracy'])



#조기 종료
# early_stopping = EarlyStopping(monitor='val_binary_accuracy', patience=7)

#모델 훈련
# fit_hist = model.fit(x_train, y_train, batch_size=32, epochs=100,
#         validation_data=(x_test, y_test), callbacks=[early_stopping])
#ㄱ fit_hist = model.fit(datagen.flow(x_train, y_train, batch_size=32),
#                      validation_data=(x_test, y_test),
#                      epochs=100)
                     # , callbacks=[early_stopping])

#ㄱ모델 평가
score=model.evaluate(x_test, y_test)
print('Evaluation loss :', score[0])
print('Evaluation accuracy :', score[1])

#ㄱ모델 저장
model.save('./models/horse_human_model_{}.h5'.format(np.around(score[1], 3)))

#ㄱ훈련 과정 시각화
plt.plot(fit_hist.history['binary_accuracy'])
plt.plot(fit_hist.history['val_binary_accuracy'])
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.plot(fit_hist.history['loss'])
plt.plot(fit_hist.history['val_loss'])
plt.legend(['train', 'test'], loc='upper left')
plt.show()

pred = model.predict(img)
print("Raw prediction value:", pred[0][0])  # 예: 0.9998 또는 0.00123
print("Predicted class:", categories[int(np.around(pred))])



# model = load_model('./models/horse_human_model_1.0.h5')
# categories = ['horse', 'human']
#
# img = Image.open(r'D:\AI_\datasets-20250411T005446Z-001\datasets\horse_human_test_img/horse1.jpg')
# img = img.resize((64, 64))
# img = np.array(img)
# img = img / 255
# img = img.reshape(1, 64, 64, 3)
# pred = model.predict(img)
# print(categories[int(np.around(pred))])

