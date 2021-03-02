class RibEye:
    def __init__(self):
        self.size = 0.0
        self.image_path = ''

class Cattle():
    def __init__(self):
        self.id = 0
        self.ribeye = RibEye()
        self.aol = ''
        self.egs = ''
        self.picanha = ''

    def toList(self):
        l = list()
        l.append(self.id)
        l.append(self.ribeye.size)
        l.append(self.ribeye.image_path)
        l.append(self.egs)
        l.append(self.picanha)
        
        return l
    
    def setValue(self, pos, val):

        assert(pos>=0 and pos <=3, "Erro ao criar os elementos de modelo de tabela: posição inválida!")

        if pos == 0:
            self.id = val
        elif pos == 1:
            self.ribeye.size = val
        elif pos == 2:
            self.egs = val
        elif pos == 3:
            self.picanha = val
    
    #column count
    @staticmethod
    def getSize():
        return 4

    @staticmethod
    def getHeader():
        return ['Id', 'AOL', 'EGS', 'Picanha']
    
    def __str__(self):
        return f"ID: {self.id}\nAOL: {self.ribeye.size}\nEGS:{self.egs}\nPICANHA: {self.picanha}\n"