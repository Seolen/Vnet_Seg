import warnings


class DefaultConfig(object):
    # visualize
    env = 'Vnet_00'
    notes = "segthor"

    # data
    dataset = 'SegThor3'
    num_classes = 5
    datadir = '/Users/seolen/Desktop/DATASET/0108_SegTHOR' # '../Datasets/SegTHOR'  # m0

    num_workers = 0
    batch_size = 1
    val_batch_size = 1

    # learning
    balance_weight = [1, 17.10, 1, 22.40, 3.96]

    use_pretrained = False
    pretrained_name = "deeplab-resnet.pth.tar"
    predict_name = 'segthor_0_01_best.pth'
    # model = "ResNet1024"   # "FCDenseNet103"  # "SegNet"  # "FCN8sAtOnce"  # "UNet" # "ResUnet2048"   #
    epoch = 100
    lr = 1e-4
    momentum = 0.99
    weight_decay = 5e-4

    use_init = False
    use_stepLR = False
    lr_decay = 0.995

    # path
    prefix = "results/checkpoints/"
    pretrain = "results/pretrain/"

    # control setting
    use_truncated = False  # For testing, only few data loaded
    use_parallel = True
    use_perparam = False

    use_balance_weight = False
    use_dice_loss = False


def parse(self, kwargs):
    '''
    update parser parameters, by dict 'kwargs'
    '''
    print('User config:')
    for k, v in kwargs.items():
        if not hasattr(self, k):
            warnings.warn("Warning: opt has not attribut %s" % k)
        setattr(self, k, v)

        print(k, getattr(self, k))

    # print('User config:')
    # for k, v in self.__class__.__dict__.items():
    #     if not k.startswith('__'):
    #         print(k, getattr(self, k))


DefaultConfig.parse = parse
opt = DefaultConfig()
# opt.parse = parse
