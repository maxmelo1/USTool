from PyQt5 import QtCore
from PyQt5 import QtGui
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QMessageBox

from animal import Cattle

class AnimalTableView(QtWidgets.QTableView):
    # def __init__(self, tmodel, parent=None):
    #     """Create the view.
    #     """
    #     super(AnimalTableView, self).__init__(parent)
    #     self.setModel(tmodel)
    #     self.tmodel = tmodel
    #


    def keyPressEvent(self, event):
        idx = self.currentIndex()
        if event.key() == QtCore.Qt.Key_Down:
            atual = idx.row()
            c = self.model().rowCount(idx)-1
            print(atual, c)

            if atual == c:
                print('adding line')
                a = Cattle()
                self.model().addAnimal(a)
                self.model().layoutChanged.emit()

        elif event.key() == QtCore.Qt.Key_Delete:
            print("DELETING ROW")
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Information)
            msg.setText("Tem certeza que deseja deletar o registro? Não será possível recuperá-lo!")
            msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
            retval = msg.exec_()

            if retval == QMessageBox.Ok:
                self.model().removeRow(idx.row())
                self.model().removeAnimal(idx.row())
                self.model().layoutChanged.emit()
                
            
        
        super(AnimalTableView, self).keyPressEvent(event) 

    