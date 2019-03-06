import numpy as np
from skimage.transform import resize
from scipy.ndimage import zoom


def resize3d(volume, standard=(224, 224, 128)):
    ''' To fed the whole volume into the network '''
    H, W, D = volume.shape
    vm = np.zeros((standard[0], standard[1], D))
    for d in range(D):
        vm[:, :, d] = resize(volume[:, :, d], (standard[0], standard[1]));
    vm = zoom(vm, (1, 1, standard[2]/D))
    return vm
