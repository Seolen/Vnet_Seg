import numpy as np
import nrrd

'''
According to the file "train/test_volume.txt", merge segments predictions to original size.
+ create a large volume -> waiting all segments orderly, write to the large volume -> save the large volume.
'''

'''
Requirement: input: 1-vcode, 2-shape of the whole volume, 3-lefttop coordinate, 4-shape of segments
'''


class Volume():
    def __init__(self, vcode, vshape, seg_size=(64, 64, 32), savedir='tmp/'):
        pass
        self.vol = np.zeros(vshape)
        self.vcode = vcode
        self.dx, self.dy, self.dz = seg_size
        self.savedir = savedir

    def receive(self, pred, vcode, vshape, lefttop):
        ''' It needs to decide whether clear the last volume '''
        if vcode == self.vcode:
            x, y, z = lefttop
            self.vol[x: x + self.dx, y:y + self.dy, z:z + self.dz] = pred
        else:
            self.save()

            self.vcode = vcode
            self.reset(vshape)
            x, y, z = lefttop
            self.vol[x: x + self.dx, y:y + self.dy, z:z + self.dz] = pred

    def save(self):
        nrrd.write(self.savedir + "%d.nrrd" % self.vcode, self.vol)

    def terminate(self):
        self.save()

    def reset(self, vshape):
        self.vol = np.zeros(vshape)

    def get_volume(self):
        return self.vol

if __name__ == '__main__':
    shape1 = (17, 17, 17)
    volume = Volume(1, shape1, seg_size=(4, 4, 4)) # n

    pred = np.random.randint(0, 2, (4, 4, 4))
    lefttop = (0, 0, 0)
    volume.receive(pred, 1, shape1, lefttop) # n

    pred = np.random.randint(5, 7, (4, 4, 4))
    lefttop = (13, 13, 13)
    volume.receive(pred, 1, shape1, lefttop)
    print(volume.get_volume())

    shape2 = (19, 19, 19)
    pred = np.random.randint(10, 12, (4, 4, 4))
    lefttop = (0, 0, 0)
    volume.receive(pred, 2, shape2, lefttop)

    print(volume.get_volume())
    volume.save()   # n