import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

#TODO criar método para remontar uma imagem a partir das regiões
#TODO ajustar o mainUnet para remontar as imagens
class Partition:
    def __init__(self, path='test'):
        self.path = path

    def convertRegions(self, path, img):
        img = cv2.imread( path+'/'+img , cv2.IMREAD_COLOR)

        h, w, c = img.shape

        self.mH = h // 2
        self.mW = w // 2

        imgA = img[0:128, 0:128, :] #top left
        imgB = img[0:128, w-128:w, :] #top right
        imgC = img[h-128:h, 0:128, :] #lower left
        imgD = img[h-128:h, w-128:w, :] #lower right 
        #imgE = img[self.mH-64:self.mH+64, self.mW-64:self.mW+64, :] # center
        imgE = img[0:128, self.mW-64:self.mW+64, :] # top center
        imgF = img[h-128:h, self.mW-64:self.mW+64] # lower center
        

        return [imgA, imgB, imgC, imgD, imgE, imgF]

    def createImg(self, path, imgName):
        '''
        salva a imagem em disco
        '''
        newImg = imgName[0:-4]

        

        regions = self.convertRegions(path, imgName)

        save = path.replace(self.path, f'{self.path}_converted')

        newPath = f'{save}/{newImg}'
        print(newPath)

        if not os.path.exists(newPath):
            os.makedirs(newPath)

        
        for i,r in enumerate(regions):
            cv2.imwrite(newPath+'/'+str(i)+'.bmp', r)
        

    def convertDataset(self):

        for i in range(10):
            newPath = [f'{self.path}/folder_{i}/train', f'{self.path}/folder_{i}/test']
            
            for el in newPath:
                p = f'{el}/anotada2'
                ids = next(os.walk(p))[2]

                for img in ids:
                    self.createImg(p,img)
                
                
                p = f'{el}/original'
                ids = next(os.walk(p))[2]

                for img in ids:                    
                    self.createImg(p,img)
 

    def reassembleImage(self, img):
        '''
            recebe 6 imagens por parâmetro a as remonta em uma, pela média das regiões
        '''        
        resultado = np.zeros((309, 213)).T
        w = 309
        h = 213
        mW = 309 // 2
        mH = 213 // 2 

        a,b,c,d,e,f = img

        resultado[0:128,0:128] += a
        resultado[85:213,0:128] += c

        resultado[0:128,181:309] += b
        resultado[85:213,181:309] += d

        resultado[85:128,0:128] /= 2 # y left
        resultado[85:128,181:309] /= 2  # y right


        resultado[0:128,mW-64:mW+64] += e
        resultado[0:85,mW-64:128] /= 2 # y upper left
        resultado[85:128,mW-64:128] /= 2 # y mid left
        resultado[0:85,181:mW+64] /= 2 # y upper right
        resultado[85:128,181:mW+64] /= 2 # y mid right

        resultado[h-128:h,mW-64:mW+64] += f
        resultado[128:h,mW-64:128] /= 2 # y lower left
        resultado[85:128,mW-64:128] /= 2 # y mid left
        resultado[128:h,181:mW+64] /= 2 # y lower right
        resultado[85:128,181:mW+64] /= 2 # y mid right

        resultado[h-128:128,128:w-128] /= 2 # y center
        
        return resultado



p = Partition('stage4_train')
p.convertDataset()

# a = np.ones((128, 128))
# b = np.ones((128, 128))
# c = np.ones((128, 128))
# d = np.ones((128,128))
# e = np.ones((128, 128))
# f = np.ones((128, 128))
# g = np.ones((128, 128))

# w = 309
# h = 213
# mW = 309 // 2
# mH = 213 // 2 

# a *= 128
# c *= 128

# b *= 128
# d *= 128

# e *= 128
# f *= 128

#res2 = p.reassembleImage([a,b,c,d,e,f])

#plt.figure(figsize=[15,5])
#plt.imshow(res2, cmap='gray', vmin=0, vmax=255)
#plt.show()

#p.reassembleImage('101A_AOL-AOriginal.bmp')