import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision
import numpy as np

from tqdm import tqdm
from torchnet import meter

import data
from models import DiceLoss, VNet
from utils import Visualizer, Evaluator
from config import opt
import ipdb


class Trainer(object):
    def __init__(self, **kwargs):
        ''' externar -> init param -> data prepare -> model load -> learning def'''

        opt.parse(kwargs)
        self.env = opt.env
        self.vis = Visualizer(opt.env)
        self.vis.text(opt.notes)

        self.evaluator = Evaluator(opt.num_classes)
        self.best_acc = self.best_epoch = -1

        self.train_loader, self.val_loader = self.data_process()
        self.model = self.set_model()
        self.criterion, self.optimizer, self.scheduler = self.learning(self.model)

    def forward(self):
        ''' train and val '''

        vis, evaluator = self.vis, self.evaluator
        train_loader, val_loader, model = self.train_loader, self.val_loader, self.model
        criterion, optimizer, scheduler = self.criterion, self.optimizer, self.scheduler

        for epoch_i in tqdm(range(opt.epoch)):
            # adjust lerning rate
            if opt.use_stepLR:
                scheduler.step()
            else:  # poly_lr_scheduler
                power = 0.9
                new_lr = opt.lr * (1 - epoch_i / opt.epoch) ** power
                for param_group in optimizer.param_groups:
                    param_group['lr'] = new_lr

            train_loss, train_mean_ious, train_ious, train_dice, train_hausdorff = self.train(model, train_loader, criterion, optimizer,
                                                                            evaluator)
            test_loss, test_mean_ious, test_ious, test_dice, test_hausdorff = self.val(model, val_loader, criterion, evaluator)
            vis.plot('loss', [train_loss, test_loss])
            vis.plot('mIOU', [train_mean_ious, test_mean_ious])
            vis.plot('train_IoU', train_ious[1:])
            vis.plot('test_IoU', test_ious[1:])
            vis.plot('train_dice', train_dice[1:])
            vis.plot('test_dice', test_dice[1:])
            vis.plot('hausdorff_distance', [train_hausdorff, test_hausdorff])

            if self.acc_update(test_mean_ious):
                self.model_save(model, epoch_i, test_mean_ious, name=opt.env + '_best.pth')
        vis.text('Best accuracy %f' % self.best_acc)
        print('Best accuracy', self.best_acc)

    def train(self, model, dataloader, criterion, optimizer, evaluator):
        model.train();
        evaluator.reset()
        loss_meter = meter.AverageValueMeter()

        for batch_idx, sample in tqdm(enumerate(dataloader)):
            img, target = sample['image'], sample['label']
            # print(img.data.shape, target.data.shape)

            img, target = Variable(img.cuda()), Variable(target.cuda())

            optimizer.zero_grad()
            output = model(img)
            loss = criterion(output, target)

            loss.backward()
            optimizer.step()

            # metrics, prediction
            loss_meter.add(loss.data.item())
            pred = output.data.cpu().argmax(1)  # shape(1, 64, 64, 32)
            evaluator.add_batch(target.data.cpu().numpy(), pred.numpy())

            '''
            # For Classification Prediction
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()
            '''

        return loss_meter.value()[0], evaluator.Mean_Intersection_over_Union(ignore_index=0), evaluator.Intersection_over_Union(), \
               evaluator.Dice_coefficient(), evaluator.Hausdorff()

    def val(self, model, dataloader, criterion, evaluator):

        model.eval();
        evaluator.reset()
        loss_meter = meter.AverageValueMeter()

        for batch_idx, sample in tqdm(enumerate(dataloader)):
            img, target = sample['image'], sample['label']
            # print(img.data.shape)
            img, target = Variable(img.cuda()), Variable(target.cuda())

            with torch.no_grad():
                output = model(img)
            loss = criterion(output, target)

            # metrics, prediction
            loss_meter.add(loss.data.item())
            pred = output.data.cpu().argmax(1)
            evaluator.add_batch(target.data.cpu().numpy(), pred.numpy())

            '''
            # For Classification Prediction
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()
            '''

        return loss_meter.value()[0], evaluator.Mean_Intersection_over_Union(ignore_index=0), evaluator.Intersection_over_Union(), \
               evaluator.Dice_coefficient(), evaluator.Hausdorff()


    def data_process(self):
        dataset = getattr(data, opt.dataset)
        trainset = dataset(opt.datadir, split='train', use_truncated=opt.use_truncated)
        valset = dataset(opt.datadir, split='test', use_truncated=opt.use_truncated)
        train_loader = DataLoader(trainset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)
        val_loader = DataLoader(valset, batch_size=opt.val_batch_size, shuffle=False, num_workers=opt.num_workers)
        print('Data processed.')

        return train_loader, val_loader

    def set_model(self):
        model = VNet(num_classes=opt.num_classes)
        if opt.use_init:
            model.weights_init()
        model.cuda()

        if opt.use_parallel:
            model = nn.DataParallel(model)
        if opt.use_pretrained:
            pt = torch.load(opt.pretrain + opt.pretrained_name)['state_dict']
            model.load_state_dict(pt)

        return model

    def learning(self, model):
        if opt.use_balance_weight:
            weights = torch.FloatTensor(opt.balance_weight).cuda()
        else:
            weights = None
        if not opt.use_dice_loss:
            criterion = nn.CrossEntropyLoss(weight=weights)
        else:
            criterion = DiceLoss()
        if not opt.use_perparam:
            optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=opt.momentum,
                                        weight_decay=opt.weight_decay)
        else:
            train_params = [{'params': model.get_1x_lr_params(), 'lr': opt.lr},
                            {'params': model.get_10x_lr_params(), 'lr': opt.lr * 10}]
            optimizer = optim.SGD(train_params, momentum=opt.momentum, weight_decay=opt.weight_decay)
        # optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

        return criterion, optimizer, scheduler

    def acc_update(self, cur_acc):
        if cur_acc > self.best_acc:
            self.best_acc = cur_acc
            return True
        return False

    def model_init(self, model):
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                # m.bias.data.zero_()

    def model_save(self, model, epoch, metric, name=opt.env + '_best.pth'):
        prefix = 'results/checkpoints/'
        torch.save({
            'epoch': epoch,
            'state_dict': model.module.state_dict(),
            'metric': metric
        }, prefix + name)


def training(**kwargs):
    trainer = Trainer(**kwargs)
    trainer.forward()

def evaluating(**kwargs):
    trainer = Trainer(**kwargs)
    trainer.evaluate()


if __name__ == '__main__':
    import fire

    fire.Fire()
