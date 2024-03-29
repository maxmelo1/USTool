# configuration vars we want to set in one place

# imshape should have 3 channels for rgb input images
# (height, width)
imshape = (256, 256, 3)
#imshape = (3888, 5184, 3)
# set your classification mode (binary or multi)
mode = 'multi'
# model_name (unet or fcn_8)
model_name = 'unet_'+mode
# log dir for tensorboard
logbase = 'logs'

# classes are defined in hues
# background should be left out
hues = {'tendao': 0,
        'gordura': 60,
        'musculo': 90,
        'osso': 30}

labels = sorted(hues.keys())

if mode == 'binary':
    n_classes = 1

elif mode == 'multi':
    n_classes = len(labels) + 1

assert imshape[0]%32 == 0 and imshape[1]%32 == 0,\
    "imshape should be multiples of 32. comment out to test different imshapes."
