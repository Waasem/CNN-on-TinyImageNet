import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import argparse
# import numpy as np
import torch.utils.data
# import logging
import os
import wandb
import time
from PIL import Image
from Resnet import ResNet

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


def parseClasses(file):
    # Validation data loader for TinyImageNet class
    classes = []
    filenames = []
    with open(file) as f:
        lines = f.readlines()
    lines = [x.strip() for x in lines]
    for x in range(0, len(lines)):
        tokens = lines[x].split()
        classes.append(tokens[1])
        filenames.append(tokens[0])
    return filenames, classes


class TinyImageNet(torch.utils.data.Dataset):
    """TinyImageNet 200 validation dataloader."""

    def __init__(self, img_path, gt_path, class_to_idx=None, transform=None):
        self.img_path = img_path
        self.gt_path = gt_path
        self.transform = transform
        self.class_to_idx = class_to_idx
        self.classidx = []
        self.imgs, self.classnames = parseClasses(gt_path)
        for classname in self.classnames:
            self.classidx.append(self.class_to_idx[classname])

    def __getitem__(self, index):
        """ inputs: Index, returns: tuple(im, label) """
        img = None
        with open(os.path.join(self.img_path, self.imgs[index]), 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
            if self.transform is not None:
                img = self.transform(img)
        label = self.classidx[index]
        return img, label

    def __len__(self):
        return len(self.imgs)


# class Model(nn.Module):
#     def __init__(self, args):
#         super(Model, self).__init__()
#         self.num_classes = 200
#         self.dropout_rate = args.dropout
#
#         self.conv1 = nn.Conv2d(3, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#         self.batchnorm1 = nn.BatchNorm2d(128)
#         self.relu1 = nn.ReLU(inplace=True)
#
#         self.conv2 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#         self.batchnorm2 = nn.BatchNorm2d(128)
#         self.relu2 = nn.ReLU(inplace=True)
# #         self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=1)
#         # g2nn: ker_size=3, stride=2
#
#         self.conv3 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#         self.batchnorm3 = nn.BatchNorm2d(128)
#         self.relu3 = nn.ReLU(inplace=True)
#
#         self.conv4 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#         self.batchnorm4 = nn.BatchNorm2d(128)
#         self.relu4 = nn.ReLU(inplace=True)
# #         self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=1)
#
#         self.conv5 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#         self.batchnorm5 = nn.BatchNorm2d(128)
#         self.relu5 = nn.ReLU(inplace=True)
# #         self.maxpool5 = nn.MaxPool2d(kernel_size=2, stride=1)
#
#         self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
#         self.dropout1 = nn.Dropout(self.dropout_rate)
#         self.fc1 = nn.Linear(in_features=128, out_features=400, bias=True)
#         self.dropout2 = nn.Dropout(self.dropout_rate)
#         self.fc2 = nn.Linear(in_features=400, out_features=400, bias=True)
#         self.dropout3 = nn.Dropout(self.dropout_rate)
#         self.fc3 = nn.Linear(in_features=400, out_features=200, bias=True)
#
#     def forward(self, x):
#         out = self.conv1(x)
#         out = self.batchnorm1(out)
#         out = self.relu1(out)
#
#         out = self.conv2(out)
#         out = self.batchnorm2(out)
#         out = self.relu2(out)
# #         out = self.maxpool2(out)
# #         print("#### MAXPOOL1 #####", out.shape)
#
#         out = self.conv3(out)
#         out = self.batchnorm3(out)
#         out = self.relu3(out)
#
#         out = self.conv4(out)
#         out = self.batchnorm4(out)
#         out = self.relu4(out)
# #         out = self.maxpool4(out)
# #         print("#### MAXPOOL2 #####", out.shape)
#
#         out = self.conv5(out)
#         out = self.batchnorm5(out)
#         out = self.relu5(out)
# #         out = self.maxpool5(out)
# #         print("#### MAXPOOL3 #####", out.shape)
#
# #         print(out.size())
# #         batch_size, channels, height, width = out.size()
# #         out = F.avg_pool2d(out, (1,1))
# #         out = torch.squeeze(out)
#
#         out = self.avg_pool(out)
#         out = out.view(out.size(0), -1)
#         out = self.dropout1(out)
#         out = self.fc1(out)
#
#         out = self.dropout2(out)
#         out = self.fc2(out)
#
#
#         out = self.dropout3(out)
#         out = self.fc3(out)
#         return out


def adjust_lr(optimizer, cur_epoch, args):
    """Set the learning rate to inital LR, then decay it by 10 for every 30 epochs."""
    # cosine_lr =  0.5 * args.lrate * (1.0 + np.cos(np.pi * cur_epoch / args.epochs))
    # warmup_factor = 0.1 * (1.0 - alpha) + alpha
    # cosine_lr *= warmup_factor
    lr = args.lrate * (0.1 ** (cur_epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train_epoch(model, train_loader, optimizer, criterion, cur_epoch, args):
    """Performs one epoch of training"""
    
    print("Training for epoch {}".format(cur_epoch))
    # enable the training mode
    model.train()
    train_loss = 0
    train_acc = 0
    step = 0
    for cur_iter, (data, target) in enumerate(train_loader):
        # adjust the learning rate for this epoch
        adjust_lr(optimizer, cur_epoch=cur_epoch, args=args)
        # transfer the data to the current GPU
        data, target = data.to(device), target.to(device)
        # perform the forward pass
        preds = model(data)
        # compute the loss
        loss = criterion(preds, target)
        # perform the backward pass
        optimizer.zero_grad()
        loss.backward()
        # update the parameters
        optimizer.step()
        # compute the losses and accuracy
        train_loss += loss.item()
        y_pred = preds.data.max(1)[1]
        acc = float(y_pred.eq(target.data).sum()) / len(data) * 100.
        train_acc += acc
        
        step += 1
#         if step % 10 == 0:
        print("\n[Epoch {:4d}: step {:4d}] Loss: {:2.3f}, Acc: {:.3f}%".format(cur_epoch, step, loss.data, acc), end='')
#         for param_group in optimizer.param_groups:
#             print(",  Current learning rate is: {}".format(param_group['lr']))
            
    length = len(train_loader.dataset) // args.bs
    return train_loss/length, train_acc/length


@torch.no_grad()
def eval_epoch(model, test_loader, criterion):
    """Evaluates the model on the validation set."""

    # enable eval mode
    model.eval()
    correct = 0
    test_loss = 0
    ctr = 0
    for cur_iter, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
        preds = model(data)
        prediction = preds.data.max(1)[1]
        correct += prediction.eq(target.data).sum()
        loss = criterion(preds, target)
        test_loss += loss.item()
        ctr += 1
    test_acc = 100. * float(correct) / len(test_loader.dataset)
    test_loss /= ctr
    return test_loss, test_acc


def main():
    
    wandb.init(project='Deterministic-ResNet18-TinyImagenet')
    parser = argparse.ArgumentParser(description="Train a deterministic CNN on CIFAR10 dataset")
    
    parser.add_argument('--epochs', type=int, default=2, help='Number of epochs (Ideally: 100)')
    parser.add_argument('--bs', type=int, default=256, help='Training batch size (Ideally: 256)')
    parser.add_argument('--lrate', type=float, default=1e-1, help='Learning rate')
    parser.add_argument('--wd', type=float, default=0.03, help='Weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='Optimizer momentum')
    parser.add_argument('--dropout', type=float, default=0.5, help='Network dropout')
    parser.add_argument('--is_train', type=bool, default=True, help='True if training, False is testing')
    parser.add_argument('--train_dataset', type=str, default='tinyimagenet200', help='Training datasets, cifar10 or tinyimagenet200')

    args = parser.parse_args()
    wandb.config.update(args)

    print('Config: Training= {}, epochs = {}, batch_size = {}, learning_rate = {}, weight_decay = {}, dropout_rate = {}'.format(args.is_train, args.epochs, args.bs, args.lrate, args.wd, args.dropout))

    # Build loaders
    # get and set the dataset paths
    data_path = '/data/tinyimagenet200'
    traindir = os.path.join(data_path, 'train')
    valdir = os.path.join(data_path, 'val', 'images')
    valgtdir = os.path.join(data_path, 'val', 'val_annotations.txt')
    print('train_directory={}, val_directory={}, val_ground_truth_directory={}'.format(traindir, valdir, valgtdir))

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # create training dataset and loader
    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(traindir, transforms.Compose([
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.bs,
        shuffle=True,
        num_workers=12,
        pin_memory=True)

    # create validation dataset
    test_dataset = TinyImageNet(
        valdir, valgtdir,
        class_to_idx=train_loader.dataset.class_to_idx.copy(),
        transform=transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            normalize]))

    # create validation loader
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.bs,
        shuffle=False,
        num_workers=12,
        pin_memory=True,
        drop_last=False)


    model = ResNet(args)
    if torch.cuda.device_count() > 1:
        print('Using ', torch.cuda.device_count(), 'GPUs')
        model = torch.nn.DataParallel(model)
        model = model.to(device)
    else:
        print('Using ', torch.cuda.device_count(), 'GPU')
        model = model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lrate, momentum=args.momentum, weight_decay=args.wd, nesterov=True)
    criterion = nn.CrossEntropyLoss().to(device)
    
    wandb.watch(model)
    start_time = time.time()
    print("Training starting now!")

    for cur_epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, cur_epoch, args)
        test_loss, test_acc = eval_epoch(model, test_loader, criterion)

        wandb.log({
            "Test Accuracy :": test_acc,
            "Train Accuracy :": train_acc,
            "Test Loss :": test_loss,
            "Train Loss :": train_loss})

        print("\n Epoch Number {} => Train Acc: {:.2f}%, Test Acc: {:.2f}%, Train Loss: {:.4f}, Test Loss: {:.4f}"
            .format(cur_epoch, train_acc, test_acc, train_loss, test_loss))

    print("Total time = {:.1f} seconds. Time per epoch = {:.1f}".format(time.time() - start_time,
                                                                        (time.time() - start_time)/args.epochs))


if __name__ == '__main__':
    main()