from PyQt5.QtWidgets import QMainWindow, QApplication, QPushButton, QTextEdit, QMessageBox, QDialog, QComboBox, QLineEdit, QLabel, QDialogButtonBox, QWidget,QGraphicsView, QTableView, QAction, QFileDialog, QShortcut
from PyQt5.QtGui import QImage, QPixmap, qRgb, QPen, QPolygonF, QStandardItemModel, QKeySequence
from PyQt5.QtCore import QTimer, Qt, QRectF, QPointF, QThread, pyqtSlot
from PyQt5 import uic, QtGui, QtCore, QtWidgets
import sys
import os
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import threading
import logging
import time
from threading import Thread

from lxml import etree, objectify

from tensorflow.keras.layers import ReLU
from keras.models import load_model

from models.loss import *
from overlay import overlay_mask
from utils.utils2 import pixel_area

import GUI.res.resources
from graphicsview import GraphicsScene
from animal import *
from animal_table_model import AnimalTableModel
from animal_table_view import AnimalTableView
from about import AboutDialog
from VideoStream import VideoStream, ProcessWorker

from enum import Enum

from functools import partial

import pandas as pd


import matplotlib
matplotlib.use('TkAgg')

APP_NAME = "Captura de Ultrassom "
APP_VERSION = "0.1.0"




class TimerMessageBox(QMessageBox):
    def __init__(self, timeout=4, parent=None):
        super(TimerMessageBox, self).__init__(parent)
        self.setWindowTitle("Aguarde...")
        self.time_to_wait = timeout
        self.setText('Atualizando informações (Fechará automaticamente em alguns instantes)')
        self.setStandardButtons(QMessageBox.NoButton)
        self.timer = QTimer(self)
        self.timer.setInterval(100)
        self.timer.timeout.connect(self.changeContent)
        self.timer.start()

    def changeContent(self):
        self.setText('Atualizando informações (Fechará automaticamente em alguns instantes)')
        if(self.time_to_wait>0):
            self.time_to_wait -= 1
        if self.time_to_wait <= 0:
            self.close()

    def closeEvent(self, event):
        self.timer.stop()
        event.accept()


class NewFile(QDialog):
    '''
    Maintains the TableModel with Cattle information. 
    '''
    #TODO Reformulate this class to mantain multiple cattle trait information per animal. 
    def __init__(self):
        super(NewFile, self).__init__()
        uic.loadUi("GUI/NewFileDialog.ui", self)

        #print(AnimalTableView())
        self.tbl_animais = AnimalTableView(self.findChild(QTableView, "tblDados"))
        

        self.animal = [Cattle()]

        #model = QStandardItemModel()
        #model.setHorizontalHeaderLabels(['Id', 'AOL', 'EGS', 'Picanha'])
        model = AnimalTableModel(self.animal)

        #self.tbl_animais = AnimalTableView(model)
        self.tbl_animais.setModel(model)
        self.tbl_animais.setGeometry(QtCore.QRect(10, 20, 470, 320))
        self.tbl_animais.setColumnWidth(1, 120)
        #self.tbl_animais.resizeColumnsToContents()

        header = self.tbl_animais.horizontalHeader()
        header.setDefaultAlignment(Qt.AlignHCenter)

        self.tbl_animais.doubleClicked.connect(self.tableDoubleClicked)
    

    def tableDoubleClicked(self):
        row = self.tbl_animais.selectionModel().currentIndex().row()
        idx = self.tbl_animais.selectionModel().currentIndex()
        d = self.tbl_animais.model().data(idx, Qt.DisplayRole)
        ob = self.tbl_animais.model().getAnimal(idx, Qt.DisplayRole)
        # print(type(ob))
        #id_us = self.tbl_animais.model().index(row, 0)
        #o = id_us
        #print(o)
        #print ("index : " + id_us)

        


class SettingsDialog(QDialog):
    '''
    Miscellaneous project configurations
    '''

    def __init__(self):
        super(SettingsDialog, self).__init__()
        uic.loadUi("GUI/Settings.ui", self)

        self.cb_us = self.findChild(QComboBox, "cbUs")
        self.txt_us = self.findChild(QLineEdit, "txtUs")
        self.cb_pipeline = self.findChild(QComboBox, "cbPipeline")
        self.button_box = self.findChild(QDialogButtonBox, "buttonBox")

        self.buttonBox.accepted.connect(self.accept2)

        self.selected_model = None
        self.model_size = None
        self.model = 0

        self.device = self.listDevices()
        self.listPipelines()


    def load_trained_model(self):
        '''
        Loads the previously selected segmentation model.
        '''
        model_path = 'GUI/models/'+self.selected_model+'/model.h5'

        model = load_model(model_path, custom_objects={
            'dice': dice,
            'relu6': ReLU})

        #print(model)
        
        self.model = model

    
    def accept2(self):
        self.selected_model = self.model_list[self.cb_pipeline.currentIndex()]
        self.model_size = int(self.selected_model.split('_')[1])
        
        self.setWindowTitle('Atualizando configurações, aguarde...')
        
        self.device = self.cb_us.currentText()
        ###############################################################################

        t = threading.Thread(target=self.load_trained_model, args=())
        
        t.start()
        t.join()

        return super().accept()

    def listPipelines(self):
        '''
        Feeds the available models to use in segmentation. Default model was trained with standard UNet model.
        '''
        self.models = {}
        self.model_list = os.listdir('GUI/models')
        model_names = list()

        for m in self.model_list:
            key, val = m.split('_')
            if key not in self.models:
                self.models[key] = list()
            self.models[key].append(val)
            model_names.append( f'{key.capitalize()} - Resolução {val}x{val} px' )

        self.cb_pipeline.clear()
        self.cb_pipeline.addItems(model_names)

        #print(models)

    def listDevices(self):
        '''
        Lists all video capture devices recognized by the OS.
        '''
        devs = ["/dev/"+el for el in list(filter( lambda x: x.startswith('video'), os.listdir('/dev')))]
        print(devs)

        devs = sorted(devs)
        
        self.cb_us.clear()
        devs.insert(0, '-')
        self.cb_us.addItems(devs)

        return devs


class ActionStates(Enum):
    '''
    This enumeration is used to control the application mode. 
    In Ribeye mode, the segmentation model is enabled. In IMF, only the image capture is enabled.
    '''
    #TODO train a model to predict the line of IMF. 
    #TODO create a menu for IMF image capture.
    Segm_Ribeye = 0
    Segm_IMF = 1

class UI(QMainWindow):
    PIXEL_SIZE = {'128': 0.01326, '256':0.00331, '512':0.00083}

    def __init__(self):
        super(UI, self).__init__()
        uic.loadUi("GUI/FrameGraber.ui", self)

        self.project_file_name = "Projeto em branco"
        
        self.setWindowTitle(f'{APP_NAME} - {APP_VERSION} - {self.project_file_name}')

        self.model_name = None
        self.buffer = None
        self.vs = None

        self.settings = SettingsDialog()
        self.settings.setWindowTitle("Configurações de Sistema")

        self.new_file = NewFile()
        self.new_file.setWindowTitle("Adicionar/editar um animal")

        self.about = AboutDialog()
 
        # find the widgets in the xml file
        self.btn_apply = self.findChild(QPushButton, "btnSegmentar")
        self.btn_settings = self.findChild(QPushButton, "btnSettings")
        self.btn_new = self.findChild(QPushButton, "btnNew")
        self.btn_save = self.findChild(QPushButton, "btnSalvar")
        #self.lbl_stream = self.findChild(QLabel, "lblStream")
        self.lbl_stream = self.findChild(QGraphicsView, "stream")

        ag = QtWidgets.QActionGroup(self)

        self.action_aol_menu = self.findChild(QAction, "actionSegmentar_AOL")
        self.action_imf_menu = self.findChild(QAction, "actionSegmentar_EGS")
        self.about_menu = self.findChild(QAction, "action_Sobre")
        self.open_menu = self.findChild(QAction, "actionAbrir")
        self.new_menu = self.findChild(QAction, "actionNovo")
        self.save_menu = self.findChild(QAction, "actionSalvar")
        self.import_menu = self.findChild(QAction, "actionImportarPlanilha")
        self.import_image_menu = self.findChild(QAction, "actionImportarImagem")
        self.export_menu = self.findChild(QAction, "actionExportarPlanilha")
        self.export_image_menu = self.findChild(QAction, "actionExportarImagem")
        self.start_grabbing_menu = self.findChild(QAction, "actionIniciarCaptura")
        self.settings_menu = self.findChild(QAction, "actionConfiguracoes")

        #self.start_grabbing_menu.setShortcut(QKeySequence("Ctrl+i"))

        ag.addAction(self.action_aol_menu)
        ag.addAction(self.action_imf_menu)

        self.actionState = ActionStates.Segm_Ribeye

        #change to graphics obj
        #self.lbl_stream = GraphicsView(self.lbl_stream)
        self.scene = GraphicsScene(self)
        self.lbl_stream.setScene(self.scene)
        
        self.lbl_stream.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)

        self.btn_apply.clicked.connect(self.applySegmentation)
        self.btn_settings.clicked.connect(self.openSettings)
        self.btn_new.clicked.connect(self.newFile)
        self.btn_save.clicked.connect(self.saveFile)
        
        self.about_menu.triggered.connect(self.openAboutDialog)
        self.open_menu.triggered.connect(self.openFile)
        self.save_menu.triggered.connect(self.saveFile)
        self.new_menu.triggered.connect(self.newFile)
        self.export_menu.triggered.connect(self.exportFile)
        self.export_image_menu.triggered.connect(self.exportImage)
        self.import_menu.triggered.connect(self.importFile)
        self.import_image_menu.triggered.connect(self.importImage)
        self.start_grabbing_menu.triggered.connect(self.startGrabbing)
        self.settings_menu.triggered.connect(self.openSettings)
        self.action_aol_menu.triggered.connect(partial(self.setCurrentAction, ActionStates.Segm_Ribeye))
        self.action_imf_menu.triggered.connect(partial(self.setCurrentAction, ActionStates.Segm_IMF))

        self.setup()
        self.settings.load_trained_model()
 
        self.show()

    def importImage(self):
        
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"Carregar arquivo", "","Imagem PNG (*.png);;Imagem JPG (*.jpg);;Imagem BMP (*.bmp);;All Files (*)", options=options)
        if fileName:
            new_size = self.lbl_stream.size()

            self.buffer = cv2.imread(fileName, cv2.IMREAD_COLOR)
            im = cv2.resize(self.buffer, (new_size.width(), new_size.height()))
            image = QImage( im.data , new_size.width(), new_size.height(), im.strides[0], QImage.Format_RGB888)#.rgbSwapped()
            self.setImage(image)
            self.new_file.animal[0].ribeye.image_path = fileName
            self.new_file.tbl_animais.model()._data[0].ribeye.image_path = fileName
            self.new_file.tbl_animais.model().layoutChanged.emit()

    def exportImage(self):
        if self.buffer is None:
            QMessageBox.about(self, "Erro", "Nenhum arquivo de imagem carregado.")
        else:
            options = QFileDialog.Options()
            options |= QFileDialog.DontUseNativeDialog
            fileName, fi = QFileDialog.getSaveFileName(self,"Salvar arquivo","","Imagem PNG (*.png);;Imagem JPG (*.jpg);;Imagem BMP (*.bmp);;All Files (*)", options=options)
            
            if fileName:
                ext = {"Imagem PNG (*.png)":'.png', 'Imagem JPG (*.jpg)': '.jpg', 'Imagem BMP (*.bmp)': '.bmp'}
                name = ([ext[el] for el in ext if el == fi])[0]
                name = fileName+name if name not in fileName else fileName
                cv2.imwrite(name, self.buffer)
                QMessageBox.about(self, "Sucesso", "Imagem salva com sucesso.")

    @pyqtSlot(QImage)
    def setImage(self, image):
        pixmap = QPixmap.fromImage(image)
        self.scene.item.setPixmap(pixmap)
        self.updateView()


    def startGrabbing(self):
        #if self.buffer is not None:
        msg = QMessageBox()
        msg.setWindowTitle("Sobreposição de arquivo")
        msg.setText("Descartar o arquivo de imagem existente?\nNão será possível recuperá-lo posteriormente!")
        msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
        retval = msg.exec_()

        if retval == QMessageBox.Ok:
            
            if self.vs.stopped:
                self.vs = self.vs.start()
                # self.updateImageWidgetThread()
                if not hasattr(self, 'workerThread'):
                    self.workerThread = QThread()
                if hasattr(self, 'worker'):
                    del self.worker
                self.worker = ProcessWorker(self)
                self.worker.moveToThread(self.workerThread)
                self.workerThread.finished.connect(self.worker.deleteLater)
                self.workerThread.started.connect(self.worker.grab)
                self.worker.imageChanged.connect(self.setImage)
                self.workerThread.start()
                print('starting grab process. Device: ', self.vs.device)
                self.workerThread.quit()
            else:
                print('stopping')
                self.buffer = self.vs.read()
                #self.vs.stop()
                self.vs.stopped = True

        else:
            QMessageBox.about(self, "Ajuda", "Para salvar a imagem, vá em\nArquivo>Exportar>Imagem")


    def dfToAnimal(self, df):
        lan = list()
        for i in df.index:
            an = Cattle()
            for j, idx in enumerate(list(df.columns)):
                an.setValue(j, list(df[idx])[i])

            lan.append(an)
        
        self.new_file.animal = lan
        #self.new_file.model._data = lan
        self.new_file.tbl_animais.model()._data = lan
        self.new_file.tbl_animais.model().layoutChanged.emit()

    def importFile(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"Abrir arquivo", "","All Files (*);;Arquivos de planilha (*.xlsx)", options=options)
        if fileName:
            self.project_file_name = fileName
            self.setWindowTitle(f'{APP_NAME} - {APP_VERSION} - {self.project_file_name}')

            elems = pd.read_excel(fileName, 'Sheet1', engine='openpyxl')
            
            self.dfToAnimal(elems)
            QMessageBox.about(self, "Abrindo planilha", "Arquivo importado com sucesso!")

    def openFile(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"Abrir arquivo", "","Arquivos de projeto (*.pro);;All Files (*)", options=options)
        if fileName:
            try:
                an = Cattle()
                tree = etree.parse(fileName)
                root = tree.getroot()

                #print( root[0].text )
                print( root.findall('name')[0].text )
                self.project_file_name = root.findall('name')[0].text
                an.id = root.findall('cattle/id')[0].text
                an.aol = root.findall('cattle/aol/size')[0].text
                an.ribeye.size = root.findall('cattle/aol/size')[0].text
                an.ribeye.image_path = root.findall('cattle/aol/path')[0].text
                an.egs = 0
                an.picanha = 0

                lan = [an]
                self.new_file.animal = lan
                self.new_file.tbl_animais.model()._data = lan
                self.new_file.tbl_animais.model().layoutChanged.emit()
                self.setWindowTitle(f'{APP_NAME} - {APP_VERSION} - {self.project_file_name}')

                if os.path.exists(self.new_file.tbl_animais.model()._data[0].ribeye.image_path):
                    image = cv2.imread(self.new_file.tbl_animais.model()._data[0].ribeye.image_path, cv2.IMREAD_COLOR)
                    self.buffer = image
                    im = cv2.resize(self.buffer, (self.image_size, self.image_size))
                    image = QImage( im.data , self.image_size, self.image_size, im.strides[0], QImage.Format_RGB888)#.rgbSwapped()
                    self.setImage(image)

                    xml_pts = root.findall('cattle/aol/points')
                    
                    l_points = list()
                    for elem in xml_pts:
                        pts = elem.findall('point')
                        for pt in pts:
                            vals = pt.text.split(',')
                            p = [[int(float(vals[0])), int(float(vals[1])) ]]
                            l_points.append(p)
                    np_points = np.array(l_points)
                    
                    pts = self.toQPoint(np_points)
                    self.scene.points = pts

                    for pt in pts:
                        self.scene.polygon_item.addPoint(pt)

                    self.updateView()

                QMessageBox.about(self, "Carregar projeto", "Projeto carregado com sucesso!")

            except OSError:
                QMessageBox.about(self, "Carregar projeto", "Erro em carregar o projeto.")

    def toDict(self):
        elems = dict((el,[]) for el in Cattle.getHeader())

        for a in self.new_file.animal:
            al = a.toList()
            for i,k in enumerate(elems):
                elems[k].append( al[i] )
        return elems


    def exportFile(self):
        if len(self.new_file.animal)>0:
            elems = self.toDict()

            df = pd.DataFrame(elems)

            options = QFileDialog.Options()
            options |= QFileDialog.DontUseNativeDialog
            fileName, _ = QFileDialog.getSaveFileName(self,"Salvar arquivo","","All Files (*);;Text Files (*.txt)", options=options)

            if fileName:
                print(fileName.find('.xls') > -1 or fileName.find('.xlsx') > -1)
                
                name = fileName if fileName.find('.xls') > -1 or fileName.find('.xlsx') > -1 else fileName+".xlsx"
                print(name)
                writer = pd.ExcelWriter(name, engine='xlsxwriter')
                df.to_excel(writer, sheet_name='Sheet1', index=False)

                writer.save()
                QMessageBox.about(self, "Salvando arquivos", "Arquivo salvo com sucesso!")

    def writeProjectXML(self, location):
        '''
        TODO adjust for multi cattle files
        '''
        root = etree.Element("app")
        n = etree.SubElement(root, 'name')
        n.text = self.project_file_name

        animal = etree.SubElement(root, 'cattle')
        
        app_id = etree.SubElement(animal, 'id')
        app_id.text = str(self.new_file.animal[0].id)
        
        aol = etree.SubElement(animal, 'aol')
        
        img = etree.SubElement(aol, 'path')
        img.text = self.new_file.animal[0].ribeye.image_path
        size = etree.SubElement(aol, 'size')
        size.text = str(self.new_file.animal[0].ribeye.size)

        #TODO save the points in project xml file.
        if self.scene.polygon_item.m_points and len(self.scene.polygon_item.m_points)>0:
            pts = etree.SubElement(aol, 'points')
            print('points:')
            
            for pt in self.scene.polygon_item.m_points:
                p = etree.SubElement(pts, 'point')
                p.text = f'{pt.x()},{pt.y()}'

        tree = etree.ElementTree(root)
        tree.write(location, pretty_print=True, xml_declaration=True,   encoding="utf-8")  

    def saveFile(self):
        if len(self.new_file.animal)>0:
            
            options = QFileDialog.Options()
            options |= QFileDialog.DontUseNativeDialog
            fileName, _ = QFileDialog.getSaveFileName(self,"Salvar arquivo","","Project Files (*.pro);;All Files (*)", options=options)

            if fileName:
                name = fileName if fileName.find('.pro') > -1  else fileName+".pro"
                self.project_file_name = name
                self.writeProjectXML(name)

                self.setWindowTitle(f'{APP_NAME} - {APP_VERSION} - {self.project_file_name}')
                QMessageBox.about(self, "Salvando arquivos", "Arquivo salvo com sucesso!")
        else:
            QMessageBox.about(self, "Erro", "Não há dados a salvar")

    def setCurrentAction(self, newState):
        self.actionState = newState

    def setup(self):
        '''
        Method that sets internal parameters:
        - List of capture devices.
        - List of Rib eye segmentation models.
        - Detect model input image size.
        '''
        self.scene.item = self.scene.addPixmap(QPixmap())
        try:
            tree = etree.parse("GUI/config/conf.xml")
            root = tree.getroot()

            #print( root[0].text )
            print( root.findall('model/size')[0].text )
            self.settings.device = root.findall('us')[0].text
            self.model_name = root.findall('model/name')[0].text
            self.settings.model_size = int(root.findall('model/size')[0].text)
            self.settings.selected_model = self.model_name + '_' + str(self.settings.model_size)
            self.image_size = self.settings.model_size

        except OSError:
            if not os.path.exists('GUI/config/'):
                os.makedirs("GUI/config/", exist_ok=True)

            root = etree.Element("system")
            us = etree.SubElement(root, 'us')
            us.text = '/dev/video0'
            
            model = etree.SubElement(root, 'model')
            name = etree.SubElement(model, 'name')
            name.text = 'unet'
            size = etree.SubElement(model, 'size')
            size.text = '128'

            tree = etree.ElementTree(root)
            tree.write('GUI/config/conf.xml', pretty_print=True, xml_declaration=True,   encoding="utf-8")  

        try:
            self.vs = VideoStream(device=int(self.settings.device[-1]))
            print('started')
        except Exception as e: 
            print("Não foi possível iniciar o dispositivo de captura: ", e)

    
    def openAboutDialog(self):
        self.about.exec()

    def openSettings(self):
        ret_val = self.settings.exec()
        
        #if ret_val == QDialogButtonBox.Ok:
        #     print(settings.model_list[settings.cb_pipeline.currentIndex()])
        self.model_name = self.settings.selected_model
        self.image_size = self.settings.model_size
        print(self.model_name)
        print(self.image_size)

        print('settings', self.settings.device)
        self.vs.changeDevice(int(self.settings.device[-1]))

        #print(self.lbl_stream.height())
        #print(self.lbl_stream.width())


    def applySegmentation(self):
        '''
        Aplies Rib eye segmentation. The segmentation mask generates a polygon that feeds the polygon object in GraphicsView.
        '''
        if self.model_name is not None:
            #image = cv2.imread("GUI/res/icons/410A_AOL-AOriginal.bmp", cv2.IMREAD_COLOR)

            new_size = self.lbl_stream.size()

            print('size::::', self.buffer.shape)
            image = self.buffer
            orig_im = image.copy()
            image = cv2.resize(image, (self.image_size, self.image_size))
            
            im_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = image / 255.
            image = np.expand_dims(image, axis=0)

            print('shape', image.shape)         

            result = self.settings.model.predict(image)
            result = result > 0.5

            print('pred shape', result[0].squeeze().shape)

            areaPrediction, _ = pixel_area(result[0].squeeze()*255 > 0, result[0].squeeze() > 0, cl=range(2))
            
            px_size = self.PIXEL_SIZE[str(self.settings.model_size)]
            print(f'pixel area: {areaPrediction[1]}, aol area: {areaPrediction[1]*px_size} cm^2')
            
            self.new_file.animal[0].ribeye.size = float(areaPrediction[1]*px_size)
            self.new_file.tbl_animais.model()._data[0].ribeye.size = float(areaPrediction[1]*px_size)
            self.new_file.tbl_animais.model().layoutChanged.emit()


            #im = np.require(result[0].squeeze()*255, np.uint8, 'C') 
            #rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB )
            im = self.overlay(image, result[0].squeeze()*255)

            im = cv2.resize(im, (new_size.width(), new_size.height()))
            result = cv2.resize(result.squeeze(axis=0).astype(np.uint8), (new_size.width(), new_size.height()))
            result = np.expand_dims(result, axis=0)
            result = np.expand_dims(result, axis=-1)

            gray_color_table = [qRgb(i, i, i) for i in range(256)]

            copy = im.copy()

            imgray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
            ret, thresh = cv2.threshold(np.array(result[0].squeeze()*255, dtype='uint8'), 127, 255, 0)

            
            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(copy, contours, -1, (255, 0, 0), 2)

            
            # cv2.imshow('contours', copy)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            # plt.imshow(result[0].squeeze()*255)
            # plt.show()
            #dst_image = cv2.applyColorMap(result[0].squeeze()*255, cv2.COLORMAP_AUTUMN)

            
            orig_im = cv2.resize(orig_im, (new_size.width(), new_size.height()))
            
            qImg = QImage( orig_im.data , new_size.width(), new_size.height(), copy.strides[0], QImage.Format_RGB888)#.rgbSwapped()
            #qImg.setColorTable(gray_color_table)
            #qImg = qImg.scaledToWidth(self.lbl_stream.width()*0.9)

            pix = QPixmap(qImg)
            #self.lbl_stream.setPixmap(QPixmap.fromImage(qImg))
            if not hasattr(self.scene, 'item'):
                #print(self.item)
                self.scene.item = self.scene.addPixmap(pix)
            else:
                self.scene.item.setPixmap(pix)

            pts = self.toQPoint(contours[0])
            #aolp = QPolygonF(pts)
            #pen = QPen(Qt.white, 1.5, Qt.SolidLine)
            #self.scene.aol = self.scene.addPolygon(aolp, pen)
            self.scene.points = pts

            for pt in pts:
                self.scene.polygon_item.addPoint(pt)

            self.updateView()
    
    def toQPoint(self, npoints):
        points = list()

        for p in npoints:
            points.append(QPointF(p[0,0], p[0,1]))
        return points
            
    def updateView(self):
        scene = self.lbl_stream.scene()
        r = scene.sceneRect()
        self.lbl_stream.fitInView(r, Qt.KeepAspectRatio)

    def resizeEvent(self, event):
        self.updateView()

    def showEvent(self, event):
        # ensure that the update only happens when showing the window
        # programmatically, otherwise it also happen when unminimizing the
        # window or changing virtual desktop
        if not event.spontaneous():
            #print('show event')
            self.updateView()

    def overlay(self, x, result):
        #green mask
        colors = [  [0, 255, 0] ]
        print(x.shape)
        
        overlay = overlay_mask(img=x*255, masks=[result], colorMap=colors)

        #plt.axis('off')
        #plt.suptitle('overlayed predicted')
        #plt.imshow(overlay[0])
        #plt.show()
        #plt.savefig(self.save_path+'/out_overlayed_predicted_ss_'+str(index)+'.png', dpi=300)

        return overlay[0]

    def newFile(self):

        ret_val = self.new_file.exec()
 
    
 
def main(): 
    app = QApplication(sys.argv)
    window = UI()
    app.exec_()

if __name__ == '__main__':
    main()