from PyQt5 import QtGui, QtWidgets
from PyQt5.QtGui import QPainter, QBrush, QPen, QPolygon, QPolygonF, QPainterPath
from PyQt5.QtWidgets import QWidget, QGraphicsScene, QApplication, QGraphicsPolygonItem, QGraphicsPathItem
from PyQt5.QtCore import QLineF, QPointF, Qt
from enum import Enum
from functools import partial
import sys
#from point2d import Point2D #replaced with QLineF

from PyQt5 import QtCore

class GripItem(QGraphicsPathItem):
    circle = QtGui.QPainterPath()
    circle.addEllipse(QtCore.QRectF(-0.5, -0.5, 1, 1))
    square = QtGui.QPainterPath()
    square.addRect(QtCore.QRectF(-2, -2, 4, 4))

    def __init__(self, annotation_item, index):
        super(GripItem, self).__init__()
        self.m_annotation_item = annotation_item
        self.m_index = index

        self.setPath(GripItem.circle)
        self.setBrush(QtGui.QColor("green"))
        self.setPen(QtGui.QPen(QtGui.QColor("green"), 2))
        self.setFlag(QtWidgets.QGraphicsItem.ItemIsSelectable, True)
        self.setFlag(QtWidgets.QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QtWidgets.QGraphicsItem.ItemSendsGeometryChanges, True)
        self.setAcceptHoverEvents(True)
        self.setZValue(11)
        self.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
    
    def hoverEnterEvent(self, event):
        self.setPath(GripItem.square)
        self.setBrush(QtGui.QColor("red"))
        super(GripItem, self).hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        self.setPath(GripItem.circle)
        self.setBrush(QtGui.QColor("green"))
        super(GripItem, self).hoverLeaveEvent(event)

    def mouseReleaseEvent(self, event):
        self.setSelected(False)
        super(GripItem, self).mouseReleaseEvent(event)

    def itemChange(self, change, value):
        if change == QtWidgets.QGraphicsItem.ItemPositionChange and self.isEnabled():
            self.m_annotation_item.movePoint(self.m_index, value)
        return super(GripItem, self).itemChange(change, value)

class PolygonAnnotation(QtWidgets.QGraphicsPolygonItem):
    def __init__(self, parent=None):
        super(PolygonAnnotation, self).__init__(parent)
        self.m_points = []
        self.setZValue(10)
        self.setPen(QtGui.QPen(QtGui.QColor("green"), 2))
        self.setBrush(QtGui.QColor(0, 255, 0, 80))
        self.setAcceptHoverEvents(True)

        self.setFlag(QtWidgets.QGraphicsItem.ItemIsSelectable, True)
        self.setFlag(QtWidgets.QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QtWidgets.QGraphicsItem.ItemSendsGeometryChanges, True)

        self.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))

        self.m_items = []

    def number_of_points(self):
        return len(self.m_items)

    def addPoint(self, p):
        self.m_points.append(p)
        self.setPolygon(QtGui.QPolygonF(self.m_points))
        item = GripItem(self, len(self.m_points) - 1)
        self.scene().addItem(item)
        self.m_items.append(item)
        item.setPos(p)

    def removeLastPoint(self):
        if self.m_points:
            self.m_points.pop()
            self.setPolygon(QtGui.QPolygonF(self.m_points))
            it = self.m_items.pop()
            self.scene().removeItem(it)
            del it

    def movePoint(self, i, p):
        #print('moving PA: ', i, p)
        if 0 <= i < len(self.m_points):
            self.m_points[i] = self.mapFromScene(p)
            self.setPolygon(QtGui.QPolygonF(self.m_points))

    def move_item(self, index, pos):
        if 0 <= index < len(self.m_items):
            item = self.m_items[index]
            item.setEnabled(False)
            item.setPos(pos)
            item.setEnabled(True)

    def itemChange(self, change, value):
        if change == QtWidgets.QGraphicsItem.ItemPositionHasChanged:
            for i, point in enumerate(self.m_points):
                self.move_item(i, self.mapToScene(point))
        return super(PolygonAnnotation, self).itemChange(change, value)

    def hoverEnterEvent(self, event):
        self.setBrush(QtGui.QColor(255, 0, 0, 100))
        super(PolygonAnnotation, self).hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        self.setBrush(QtGui.QColor(0, 255, 0, 80))
        super(PolygonAnnotation, self).hoverLeaveEvent(event)


class Instructions(Enum):
    No_Instruction = 0
    Polygon_Instruction = 1




class GraphicsScene(QGraphicsScene):
    def __init__(self, parent=None):
        QGraphicsScene.__init__(self, parent)
        #self.setSceneRect(-100, -100, 200, 200)
        
        self.points = list()
        
        self.aol = None
        self.setCurrentInstruction(Instructions.No_Instruction)


    def setCurrentInstruction(self, instruction):
        self.current_instruction = instruction
        self.polygon_item = PolygonAnnotation()
        self.addItem(self.polygon_item)
    
    
    # def mouseReleaseEvent(self, event):
    #     xa = event.scenePos().x()
    #     ya = event.scenePos().y()

    #     print('GraphicsScene::MouseReleaseEvent', xa, ya)

    #     app = QApplication.instance()
    #     #app.instance().restoreOverrideCursor()

    def mousePressEvent(self, event):
        pen = QPen(QtCore.Qt.white)
        brush = QBrush(QtCore.Qt.white)
        x = event.scenePos().x()
        y = event.scenePos().y()
        
        app = QApplication.instance()

        #app.instance().setOverrideCursor(Qt.OpenHandCursor)

        print('GraphicsScene::MousePressEvent', x, y)
        
        if self.current_instruction == Instructions.Polygon_Instruction:
            self.polygon_item.removeLastPoint()
            self.polygon_item.addPoint(event.scenePos())
            # movable element
            self.polygon_item.addPoint(event.scenePos())

        super(GraphicsScene, self).mousePressEvent(event)
    
    def mouseMoveEvent(self, event):
        if self.current_instruction == Instructions.Polygon_Instruction:
            self.polygon_item.movePoint(self.polygon_item.number_of_points()-1, event.scenePos())
        super(GraphicsScene, self).mouseMoveEvent(event)