import sys
import numpy as np
from keras.models import load_model

from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5 import uic
from PIL import Image

form_class = uic.loadUiType('./horse_and_human.ui')[0]

class ExampleApp(QWidget, form_class):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.path = ('project_02/imgs_horse_human/horse.png', '')
        self.btn_open.clicked.connect(self.btn_clicked_slot)#괄호안은 스롯이라고 함. 시그널이 오면 스롯이 활성화 됨.
        self.model = load_model('./models/cat_dog_model_1.0.h5')

    def btn_clicked_slot(self):
        old_path = self.path
        self.path = QFileDialog.getOpenFileName(
            self, 'Open File', './imgs',
            'Image Files (*.jpg *.jpeg *.png);;All Files (*)')
        print(self.path)
        if self.path[0] == '':
            self.path = old_path

        else:
            try:
                pixmap =QPixmap(self.path [0])
                self.lbl_img.setPixmap(pixmap)
                img = pixmap.toImage()

                img = Image.open(self.path [0])
                img = img.convert('RGB')
                img = img.resize((64, 64))
                img = np.array(img)
                img = img / 255
                img = img.reshape(1, 64, 64, 3)
                pred = self.model.predict(img)
                print(pred)
                if pred[0][0] > 0.5:
                    self.lbl_result.setText('사람입니다')
                    print('사람입니다')
                else :
                    self.lbl_result.setText('말입니다')
                    print('말입니다')
            except:
                print('error')

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = ExampleApp()
    main_window.show()
    sys.exit(app.exec_())