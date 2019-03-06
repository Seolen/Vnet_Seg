import torch
from torch.utils.data.dataset import Dataset
import os.path as osp
import collections
import nibabel as nib
import numpy as np

import numpy as np
from scipy.misc import imresize
from scipy.ndimage import zoom

'''
1. The whole volume, resize to a fixed size
2. Evaluation on the original mask
'''


class SegThor3(Dataset):
    def __init__(self, root, split='train', use_truncated=False):
        self.root = root
        self.split = split

        self.data = collections.defaultdict(list)
        self.mean, self.std = 0.456, 0.224

        # Notice
        if use_truncated:
            print('Data Info: Truncated used!')

        # load train/val.txt, offer index-dir convertion

        seolen_count = 0
        for pid in range(1, 33):
            pdir = 'Patient_%02d' % pid
            _dir = osp.join(root, pdir)
            self.data['train'].append({'pdir': _dir, 'pid': pid})

            seolen_count += 1
            if use_truncated and seolen_count>4:
                break;
        for pid in range(33, 41):
            pdir = 'Patient_%02d' % pid
            _dir = osp.join(root, pdir)
            self.data['test'].append({'pdir': _dir, 'pid': pid})

            seolen_count += 1
            if use_truncated and seolen_count > 4:
                break;

    def __getitem__(self, index):
        pass
        ## index to dir -> img/label path -> load data
        #  Read lefttop index -> load a sub volume -> load data
        _dict = self.data[self.split][index]
        pdir, pid = _dict['pdir'], _dict['pid']
        img_3d = nib.load(osp.join(pdir, 'Patient_%02d.nii.gz' % pid))
        lbl_3d = nib.load(osp.join(pdir, 'GT.nii.gz'))
        img = img_3d.get_fdata()
        lbl = lbl_3d.get_data()

        # resize
        # st = (224, 224, 128) # OOM when loss.backward
        # st = (192, 192, 128)
        st = (144, 144, 288)  # 12G
        # st = (240, 240, 160)    # 24G
        img = self.resize3d(img, st)
        lbl = self.resize3d(lbl, st, interpolation='nearest')        # Error: lbl must be integer! but not float
        # import ipdb; ipdb.set_trace()

        # transform
        img, lbl = self._transform(img, lbl)
        # img_ = np.array([img, img, img])
        img_ = np.expand_dims(img, axis=0)
        img_ = torch.from_numpy(img_).float()

        lbl_ = torch.from_numpy(lbl).long()

        del img_3d
        del lbl_3d
        return {'image': img_, 'label': lbl_}

    def _transform(self, img, mask, low_range=-200, high_range=200, ):
        # thershold [-200, 200] -> normalize -> crop
        _img = img.copy()
        _img[img > high_range] = high_range
        _img[img < low_range] = low_range

        _img = (_img + 200) / 400
        _img -= self.mean
        _img /= self.std

        # top, bottom, left, right = self.boundary
        # return _img[top:bottom, left:right], mask[top:bottom, left:right]
        return _img, mask

    def resize3d(self, volume, standard=(208, 208, 128), interpolation='bilinear'):
        ''' To fed the whole volume into the network '''
        crop_params = [181, 392, 106, 399]

        H, W, D = volume.shape
        # Assumption: D mod 16 = 0
        D = D - D % 16
        vm = np.zeros((standard[0], standard[1], D))

        for d in range(D):
            vm[:, :, d] = imresize(volume[crop_params[0]:crop_params[1], crop_params[2]:crop_params[3], d], (standard[0], standard[1]), interp=interpolation);
            # vm[:, :, d] = imresize(volume[:, :, d], (standard[0], standard[1]), interp=interpolation)
        # vm = zoom(vm, (1, 1, standard[2] / D), mode='nearest')    # zoom cannot be employed, try another alternative!
        return vm

    def __len__(self):
        return len(self.data[self.split])


if __name__ == '__main__':

    from torch.utils.data import DataLoader
    import nrrd

    root = '/Users/seolen/Desktop/DATASET/0108_SegTHOR/train'
    train_set = SegThor3(root, split='train', use_truncated=True)
    train_loader = DataLoader(train_set, batch_size=1, shuffle=False)

    for batch_idx, sample in enumerate(train_loader):
        img, target = sample['image'], sample['label']
        # print(batch_idx, img.numpy().shape)
        # if img.numpy().shape != (1, 64, 64, 32) or img.numpy().shape != target.numpy().shape:
        #     print("Error Info: %d th volume is wrong" % batch_idx)
        # import ipdb;
        # ipdb.set_trace()
        print(img.numpy().shape, target.numpy().shape)
        nrrd.write('check_data/s3_%di.nrrd' % batch_idx, np.squeeze(img.numpy(), axis=(0,1))*255)
        nrrd.write('check_data/s3_%dt.nrrd' % batch_idx, np.squeeze(target.numpy(), axis=0))
        break

