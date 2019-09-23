from PyQt5.QtWidgets import (QWidget, QTextEdit, QFileDialog, QApplication, QPushButton, QLineEdit)
from PyQt5.QtGui import QImage,QTextDocument
from PyQt5.QtCore import QUrl
import sys 
from new_demo import M_P

class qt(QWidget):
    def __init__(self):
        super().__init__()
        self.le = QLineEdit(self)
        self.exit_div = QTextEdit(self)
        self.init_ui()
        self.dir_url = None
        self.setFixedSize(self.width(), self.height())

    def init_ui(self):

        self.le.setEnabled(False)
        self.le.setGeometry(150, 30, 380, 30)

        b = QPushButton('选择文件夹', self)
        b.clicked.connect(self.select_dir)
        b.setGeometry(20, 30, 100, 30)

        start_b = QPushButton('点击开始识别', self)
        start_b.setGeometry(210, 90, 100, 30)
        start_b.clicked.connect(self.start)

        self.exit_div.setGeometry(20, 150, 510, 260)
        self.exit_div.setReadOnly(True)

        self.setGeometry(600, 400, 550, 450)
        self.setWindowTitle('数字识别工具')
        # print('elf.exit_div', dir( self.exit_div))
        self.show()

    def select_dir(self):
        dir_name = QFileDialog.getExistingDirectory(None, ' 请选择需要识别图片的文件夹')
        if dir_name:
            self.le.setText(dir_name)
            self.exit_div.append(' new has selected %s' % dir_name)
            self.dir_url = dir_name
        else:
            self.exit_div.append(' 选择取消')
        self.exit_div.moveCursor(self.exit_div.textCursor().End)

    def start(self):

        if self.dir_url:
            self.exit_div.append(' 找到需要识别的文件夹 %s' % self.dir_url)
            self.exit_div.append(' 开始识别...')
            M_P.run_predict(self, self.dir_url)
        else:
            self.exit_div.append(' 还未选择任何需要识别的文件夹')
        self.exit_div.moveCursor(self.exit_div.textCursor().End)

    def predict_show(self, predict_data, predict_path=None):

        print('aaaa')
        cursor = self.exit_div.textCursor()
        if predict_path:
            image = QImage(predict_path)
            document = self.exit_div.document()
            document.addResource(QTextDocument.ImageResource, QUrl("image"), image)
            # cursor.insertImage("image")

        self.exit_div.append(predict_data)

        self.exit_div.moveCursor(cursor.End)
        if predict_path:
            print('-------predict_path--------', predict_path, predict_data)

    def predict_data_format(self, data):
        for i in data:
            self.predict_show(i[0], i[1])
            


app = QApplication(sys.argv)
QT = qt()
sys.exit(app.exec_())