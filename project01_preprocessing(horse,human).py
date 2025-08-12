#전처리를 해야하는 작업
from PIL import Image
import glob
import numpy as np
from sklearn.model_selection import train_test_split

img_dir = './imgs_horse_human/' #이미지 통로 경로
categories = ['horse', 'human']
image_w = 64
image_h = 64

pixel = 64 * 64 * 3 #이미지 하나당 픽셀
X = []
Y = []
for idx, category in enumerate(categories):
    for i, img_path in enumerate(glob.glob(img_dir + category + '*.png')):
        try:
            img = Image.open(img_path).convert('RGB')  # <<== 알파 채널 제거!
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





