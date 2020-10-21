import sys
from PyQt5.QtWidgets import QMainWindow, QApplication
from print import Ui_MainWindow
from PyQt5.QtCore import pyqtSlot
from printhello import print2
class MyWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MyWindow, self).__init__()
        self.setupUi(self)
        self.s=[]
    @pyqtSlot()
    def on_pushButton_clicked(self):
       #self.n= self.lineEdit.text()
       self.getradio()
       print2(str(self.s))
       #self.n= self.lineEdit.text()
       self.lineEdit.setText(str(self.s[1]))
    def getradio(self):
        n=[]
        if self.radioButton.isChecked():
            n.append(1)
        if self.radioButton_2.isChecked():
            n.append(2)
        if self.radioButton_3.isChecked():
            n.append(3)
        if self.radioButton_4.isChecked():
            n.append(4)
        self.s=n
if __name__ == '__main__':
    app = QApplication(sys.argv)
    mywindow = MyWindow()
    mywindow.show()
    sys.exit(app.exec_())