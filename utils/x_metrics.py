import numpy as np
from scipy.spatial.distance import directed_hausdorff


class Evaluator(object):
    '''
    Relationship between IoU and Dice:
        https://medium.com/datadriveninvestor/deep-learning-in-medical-imaging-3c1008431aaf
        IOU = DICE/(2-DICE);
        DICE = 2*IOU/(1+IOU);
    '''

    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,) * 2)
        self.gt_image = self.pre_image = None

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Acc = np.nanmean(Acc)
        return Acc

    def Hausdorff(self):
        ''' https://stackoverflow.com/questions/30706079/hausdorff-distance-between-3d-grids '''
        vol_a = self.gt_image
        vol_b = self.pre_image
        dist_lst = []
        for idx in range(len(vol_a)):
            dist_min = 1000.0
            for idx2 in range(len(vol_b)):
                dist = np.linalg.norm(vol_a[idx] - vol_b[idx2])
                if dist_min > dist:
                    dist_min = dist
            dist_lst.append(dist_min)
        return np.max(dist_lst)
        # return self.hausdorff(self.gt_image, self.pre_image)

    def Dice_coefficient(self):
        return np.diag(self.confusion_matrix) * 2 / \
               (np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0))

    def Mean_Intersection_over_Union(self, ignore_index=None):
        MIoU = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix))
        if ignore_index is not None:
            MIoU = np.delete(MIoU, ignore_index, 0)
        MIoU = np.nanmean(MIoU)
        return MIoU

    def Intersection_over_Union(self):
        IoU = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix))
        return IoU

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        self.gt_image = gt_image
        self.pre_image = pre_image
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class ** 2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

    def bbox(self, array, point, radius):
        a = array[np.where(np.logical_and(array[:, 0] >= point[0] - radius, array[:, 0] <= point[0] + radius))]
        b = a[np.where(np.logical_and(a[:, 1] >= point[1] - radius, a[:, 1] <= point[1] + radius))]
        c = b[np.where(np.logical_and(b[:, 2] >= point[2] - radius, b[:, 2] <= point[2] + radius))]
        return c

    def hausdorff(self, surface_a, surface_b):

        # Taking two arrays as input file, the function is searching for the Hausdorff distane of "surface_a" to "surface_b"
        dists = []

        l = len(surface_a)

        for i in range(l):

            # walking through all the points of surface_a
            dist_min = 1000.0
            radius = 0
            b_mod = np.empty(shape=(0, 0, 0))

            # increasing the cube size around the point until the cube contains at least 1 point
            while b_mod.shape[0] == 0:
                b_mod = self.bbox(surface_b, surface_a[i], radius)
                radius += 1

            # to avoid getting false result (point is close to the edge, but along an axis another one is closer),
            # increasing the size of the cube
            b_mod = self.bbox(surface_b, surface_a[i], radius * math.sqrt(3))

            for j in range(len(b_mod)):
                # walking through the small number of points to find the minimum distance
                dist = np.linalg.norm(surface_a[i] - b_mod[j])
                if dist_min > dist:
                    dist_min = dist

            dists.append(dist_min)

        return np.max(dists)


if __name__ == '__main__':
    logits = np.random.randint(0, 5, size=(30, 30, 30))
    pred = np.random.randint(0, 5, size=(30, 30, 30))
    # evaluator
    eva = Evaluator(num_class=5)
    eva.reset()
    eva.add_batch(logits, pred)
    print(eva.Hausdorff())
    print(eva.Dice_coefficient())

    eva.reset()
    eva.add_batch(pred, logits)
    print(eva.Hausdorff())
    print(eva.Dice_coefficient())
