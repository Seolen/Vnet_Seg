import torch
from torch.utils.data.dataset import Dataset
import os.path as osp
import collections
import nibabel as nib
import numpy as np

''' 
1. Large volume (128, 128, ~288) 
2. Batch_size = 1
'''


class SegThor2(Dataset):
    def __init__(self, root, split='train', use_truncated=False, volume_shape=(128, 128), partial_train=False):
        self.root = root
        self.split = split
        self.volume_shape = volume_shape
        self.partial_train = partial_train

        self.data = collections.defaultdict(list)

        self.boundary = [161, 413, 85, 401]  # (252, 316)
        self.mean, self.std = 0.456, 0.224

        # Notice
        if use_truncated:
            print('Data Info: Truncated dataset is employed!')
        if partial_train:
            print('Data Info: Partial Train (class 1,3) is employed!')

        # load train/val.txt, offer index-dir convertion

        imgsets_file = osp.join(root, '%s_large_volume.txt' % split)
        seolen_count = 0
        for line in open(imgsets_file):
            line = line.strip()
            _dir, _zsize, _zstart, _x, _y = line.split(' ')
            self.data[split].append({
                'dir': int(_dir), 'zsize': int(_zsize), 'zstart': int(_zstart), 'xy': (int(_x), int(_y))
            })
            seolen_count += 1
            if use_truncated and seolen_count > 2:
                break

    def __getitem__(self, index):
        pass
        ## index to dir -> img/label path -> load data
        #  Read lefttop index -> load a sub volume -> load data

        filemap = self.data[self.split][index]
        dir_name, zsize, zstart, xy = 'Patient_%02d' % filemap['dir'], filemap['zsize'], filemap['zstart'], filemap[
            'xy']
        _x, _y = xy
        _x, _y = _x + self.boundary[0], _y + self.boundary[2]
        _dx, _dy = self.volume_shape

        _dir = osp.join(self.root, 'train/%s' % dir_name)
        img_3d = nib.load(osp.join(_dir, '%s.nii.gz' % dir_name))
        lbl_3d = nib.load(osp.join(_dir, 'GT.nii.gz'))
        img = img_3d.get_fdata()
        img = img[_x:_x + _dx, _y:_y + _dy, zstart:zstart + zsize]

        lbl = lbl_3d.get_data()
        lbl = lbl[_x:_x + _dx, _y:_y + _dy, zstart:zstart + zsize]

        # transform
        img, lbl = self._transform(img, lbl)
        # img_ = np.array([img, img, img])
        img_ = np.expand_dims(img, axis=0)
        img_ = torch.from_numpy(img_).float()

        if self.partial_train:
            lbl[lbl == 2] == 0
            lbl[lbl == 4] == 0
            lbl[lbl == 3] == 2
        lbl_ = torch.from_numpy(lbl).long()

        ## check img_, lbl_ 's shapes.
        del img_3d
        del lbl_3d
        return {'image': img_, 'label': lbl_}

    def _transform(self, img, mask, low_range=-200, high_range=200, ):
        # thershold [-200, 200] -> normalize -> crop
        _img = img.copy()
        _img[img > high_range] = high_range
        _img[img < low_range] = low_range

        # _img /= 255.0
        _img = (_img + 200) / 400
        _img -= self.mean
        _img /= self.std

        # top, bottom, left, right = self.boundary
        # return _img[top:bottom, left:right], mask[top:bottom, left:right]
        return _img, mask

    def __len__(self):
        return len(self.data[self.split])


if __name__ == '__main__':
    '''
    from torch.utils.data import DataLoader
    import nrrd

    root = '/Users/seolen/Desktop/DATASET/0108_SegTHOR'
    train_set = SegThor(root, split='train', use_truncated=True)
    train_loader = DataLoader(train_set, batch_size=1, shuffle=False)

    for batch_idx, sample in enumerate(train_loader):
        img, target = sample['image'], sample['label']
        # print(batch_idx, img.numpy().shape)
        # if img.numpy().shape != (1, 64, 64, 32) or img.numpy().shape != target.numpy().shape:
        #     print("Error Info: %d th volume is wrong" % batch_idx)

        if batch_idx in list(range(18,22)):
            nrrd.write('check_data/%di.nrrd' % batch_idx, np.squeeze(img.numpy(), axis=0))
            nrrd.write('check_data/%dt.nrrd' % batch_idx, np.squeeze(target.numpy(), axis=0))
    '''
