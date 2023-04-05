from PyQt5 import QtWidgets, uic, QtGui, QtCore


class AboutDialog(QtWidgets.QDialog):
    def __init__(self):
        super(AboutDialog, self).__init__()
        uic.loadUi("GUI/AboutDialog.ui", self)

        self.setFixedSize(self.size())