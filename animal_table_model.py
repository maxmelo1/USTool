import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt

from animal import Cattle

class AnimalTableModel(QtCore.QAbstractTableModel):
    def __init__(self, data):
        super(AnimalTableModel, self).__init__()
        self._data = data

    def data(self, index, role):
        if role == Qt.DisplayRole:
            # See below for the nested-list data structure.
            # .row() indexes into the outer list,
            # .column() indexes into the sub-list
            #return self._data[index.row()][index.column()]
            if index.column() == 0:
                return self._data[index.row()].id
            elif index.column() == 1:
                return self._data[index.row()].ribeye.size
            elif index.column() == 2:
                return self._data[index.row()].egs
            elif index.column() == 3:
                return self._data[index.row()].picanha

    def setData(self, index, value, role):
        if role == Qt.EditRole:
            #self._data[index.row()][index.column()] = value
            #TODO validate fields
            if index.column() == 0:
                self._data[index.row()].id = value
            elif index.column() == 1:
                self._data[index.row()].ribeye.size = value
            elif index.column() == 2:
                self._data[index.row()].egs = value
            elif index.column() == 3:
                self._data[index.row()].picanha = value

            return True

    def addAnimal(self, obj):
        self._data.append(obj)
    
    def removeAnimal(self, idx):
        del self._data[idx]

    def getAnimal(self, index, role):
        if role == Qt.DisplayRole:
            # See below for the nested-list data structure.
            # .row() indexes into the outer list,
            # .column() indexes into the sub-list
            return self._data[index.row()]
    
    def flags(self, index):
        return Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsEditable

    def rowCount(self, index):
        # The length of the outer list.
        return len(self._data)

    def columnCount(self, index):
        # The following takes the first sub-list, and returns
        # the length (only works if all rows are an equal length)
        #return len(self._data[0])
        return Cattle.getSize()

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole and orientation == Qt.Horizontal:
            return Cattle.getHeader()[section]
        return QtCore.QAbstractTableModel.headerData(self, section, orientation, role)

    def keyPressEvent(self, event): #Reimplement the event here, in your case, do nothing
        print("oi")
        return

    
    