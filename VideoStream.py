import cv2
from queue import Queue
from threading import Thread
import numpy as np
# import v4l2capture

from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QObject, QThread, pyqtSignal


class VideoStream:
    def __init__(self, device=0, size=300):
        self.stream = cv2.VideoCapture(device, cv2.CAP_V4L)
        self.device = device
        self.size = size
        #self.stream = cv2.VideoCapture(device)
        #self.stream = cv2.VideoCapture('v4l2:///dev/video0')
        #self.stream.set(cv2.CAP_FFMPEG,True)
        #self.stream.set(cv2.CAP_PROP_FPS,30)
        #self.stream.set(cv2.CAP_PROP_FRAME_WIDTH,320)
        #self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT,240)
        # self.stream = v4l2capture.Video_device(device)

        # size_x, size_y = video.set_format(320, 240, fourcc='MJPG')
        # self.stream.create_buffers(size)
        # self.stream.queue_all_buffers()
        # self.stream.start()

        #self.stream.set(cv2.CAP_PROP_FPS, 10)
        #os.system("v4l2-ctl -d /dev/video0 -i 0 -s 5 --set-fmt-video=width=320,height=240,pixelformat=4")
        
        print('selected device:', device)
        self.stopped = True
        self.queue = Queue(maxsize=size)
    
    def changeDevice(self, device):
        self.stream = cv2.VideoCapture(device, cv2.CAP_V4L)

        self.stopped = True
        self.queue = Queue(maxsize=self.size)
    
    def start(self):
        if not self.stream.isOpened():
            self.stream.open(self.device)
        thread = Thread(target=self.update, args=())
        self.stopped = False
        thread.daemon = True
        thread.start()
        return self

    def update(self):
        while self.stopped is False:
            if not self.queue.full():
                #grabbed = False
                #while not grabbed:
                (self.grabbed, frame) = self.stream.read()
                
            # if not grabbed:
            #     self.stop()
            #     #print('ishiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii')
            #     return
            
            self.queue.put(frame)

    def read(self):
        return self.queue.get()

    def check_queue(self):
        return self.queue.qsize() > 0

    def stop(self):
        self.stopped = True
        self.stream.release()



class ProcessWorker(QObject):
    imageChanged = pyqtSignal(QImage)

    def __init__(self, mw):
        super(ProcessWorker, self).__init__()
        self.mw = mw
        self.lastFrame = None

    def grab(self):
        
        print('grabber thread started', self.mw.vs.stopped )
        while not self.mw.vs.stopped:
            #print('queue size: ', self.vs.check_queue())
            if self.mw.vs.check_queue():
                frame = self.mw.vs.read()
                
                image = frame.copy()
                self.lastFrame = frame.copy()
                
                new_size = self.mw.lbl_stream.size()
                image = cv2.resize(image, (new_size.width(), new_size.height()))

                im_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                
                
                #img = QImage( im_gray.data , self.mw.image_size, self.mw.image_size, im_gray.strides[0], QImage.Format_RGB888)#.rgbSwapped()
                img = QImage( im_gray.data , new_size.width(), new_size.height(), im_gray.strides[0], QImage.Format_Grayscale8)#.rgbSwapped()

                self.imageChanged.emit(img)
                QThread.msleep(1)
        print('grabber thread finished')
        

    def onfinish(self):
        self.finish()