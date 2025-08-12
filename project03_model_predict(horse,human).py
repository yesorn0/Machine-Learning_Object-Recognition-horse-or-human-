from PIL import Image
import numpy as np
from keras.models import load_model

model = load_model('./models/horse_human_model_1.0.h5')
categories = ['horse', 'human']

img = Image.open('C:/Users/USER/Downloads/horse01-0.png').convert('RGB')
img = img.resize((64, 64))
img = np.array(img)
img = img / 255
img = img.reshape(1, 64, 64, 3) #(64,64,3)을1개로 묶어줌
pred = model.predict(img)
print(categories[int(np.around(pred))])
print(f"예측값 (확률): {pred[0][0]:.4f}")
print("결과:", categories[int(np.around(pred))])