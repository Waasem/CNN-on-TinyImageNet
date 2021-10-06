import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
# import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import argparse
# import numpy as np
import torch.utils.data
# import logging
import os
# import wandb
import time
import torch.multiprocessing as mp
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torchvision.models as models


# from Resnet import ResNet

gpu_ids = "0, 1, 2, 3, 4, 5, 6, 7"

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
n_gpus = torch.cuda.device_count()

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


# import torch
# import torch.nn.functional as F
# import torch.nn as nn
# import argparse

def setup(rank, world_size):
    # Setup rendezvous
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '120922'

    # Initialize the process group
    torch.distributed.init_process_group(
        backend="nccl",
        init_method="env://",
        rank=rank,
        world_size=world_size)

def gather_variable(world_size, variable):
    # Initialize lists
    variable_list = [variable.clone() for _ in range(world_size)]
    # Gather all variables
    dist.all_gather(variable_list, variable)
    # Convert from list to single tensor
    return torch.stack(variable_list)


def gather_acc_and_loss(rank, world_size, correct, total_tested, total_loss):
    # Gather values from all machines (ranks)
    correct_list = gather_variable(world_size, correct)
    total_tested_list = gather_variable(world_size, total_tested)
    total_loss_list = gather_variable(world_size, total_loss)
    dist.barrier()

    # Calculate final metrics
    if rank == 0:
        acc = correct_list.sum() / total_tested_list.sum()
        loss = (total_tested_list * total_loss_list).sum() / total_tested_list.sum()
        return acc.item(), loss.item()
    else:
        return None, None


class ResBlock(nn.Module):
    def __init__(self, in_channels, stride, kernel_num=64, bias=False, padding=1):
        super(ResBlock, self).__init__()

        self.in_channels = in_channels
        self.kernel_num = kernel_num
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.kernel_size = 3

        self.conv1 = nn.Conv2d(self.in_channels, self.kernel_num, self.kernel_size, self.stride, padding=self.padding,
                               bias=self.bias)
        self.conv2 = nn.Conv2d(self.kernel_num, self.kernel_num, self.kernel_size, padding=self.padding, bias=self.bias)
        self.conv = nn.Conv2d(self.in_channels, self.kernel_num, 1)
        self.relu = nn.ReLU()
        self.batchnorm1 = nn.BatchNorm2d(self.kernel_num)

    def forward(self, x):
        mu = self.conv1(x)
        mu = self.batchnorm1(mu)
        mu = self.relu(mu)

        mu = self.conv2(mu)
        mu = self.batchnorm1(mu)

        #         if x[1].shape!=mu[1].shape:
        #             x = self.conv(x)

        mu_o = mu + x
        return mu_o


class ResNet1(nn.Module):
    def __init__(self, args, kernel_num):
        super(ResNet1, self).__init__()

        self.kernel_num = kernel_num
        self.outputsize = args.output_size
        self.drop = args.dropout

        self.dropout = nn.Dropout2d(self.drop)
        self.conv1 = nn.Conv2d(3, self.kernel_num, kernel_size=7, stride=2, padding=3, bias=False)
        self.batchnorm1 = nn.BatchNorm2d(self.kernel_num)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.resblock1 = ResBlock(in_channels=64, kernel_num=64, stride=1)
        self.resblock2 = ResBlock(in_channels=64, kernel_num=64, stride=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(128)
        self.resblock3 = ResBlock(in_channels=128, kernel_num=128, stride=1)
        self.resblock4 = ResBlock(in_channels=128, kernel_num=128, stride=1)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1, bias=False)
        self.batchnorm3 = nn.BatchNorm2d(256)
        self.resblock5 = ResBlock(in_channels=256, kernel_num=256, stride=1)
        self.resblock6 = ResBlock(in_channels=256, kernel_num=256, stride=1)
        self.resblock7 = ResBlock(in_channels=256, kernel_num=256, stride=1)
        #         self.resblock8 = ResBlock(args, in_channels=512, kernel_num=512, stride=1)

        #         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.maxpool2 = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(256 * 28 * 28, self.outputsize)

    def forward(self, x):
        mu = self.conv1(x)
        mu = self.batchnorm1(mu)
        mu = self.relu(mu)
        mu = self.maxpool(mu)
        #         print("mu after 1st maxpool", mu.shape)

        mu = self.resblock1(mu)
        mu = self.relu(mu)
        mu = self.resblock2(mu)
        mu = self.relu(mu)
        mu = self.dropout(mu)

        mu = self.conv2(mu)
        mu = self.batchnorm2(mu)
        mu = self.relu(mu)

        mu = self.resblock3(mu)
        mu = self.relu(mu)
        mu = self.resblock4(mu)
        mu = self.relu(mu)
        mu = self.dropout(mu)

        mu = mu = self.conv3(mu)
        mu = self.batchnorm3(mu)
        mu = self.relu(mu)

        mu = self.resblock5(mu)
        mu = self.relu(mu)
        mu = self.resblock6(mu)
        mu = self.relu(mu)
        mu = self.resblock7(mu)
        mu = self.relu(mu)
        mu = self.dropout(mu)
        #         mu = self.resblock8(mu)
        #         print("mu after resblock 7", mu.shape)

        #         mu = self.avgpool(mu)
        mu = self.maxpool2(mu)
        #         print("################## mu after max pool2:", mu.shape)
        mu_flat = torch.flatten(mu, start_dim=1)
        muf = self.fc(mu_flat)

        return muf


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


def adjust_lr(optimizer, cur_epoch, args):
    """Set the learning rate to inital LR, then decay it by 10 for every 30 epochs."""
    # cosine_lr =  0.5 * args.lrate * (1.0 + np.cos(np.pi * cur_epoch / args.epochs))
    # warmup_factor = 0.1 * (1.0 - alpha) + alpha
    # cosine_lr *= warmup_factor
    lr = args.lrate * (0.1 ** (cur_epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train_epoch(net, train_loader, optimizer, criterion, cur_epoch, args, rank):
    """Performs one epoch of training"""

    # if rank ==0:
    #     print("Training for epoch {}".format(cur_epoch))
    # enable the training mode
    # net.train(rank)
    step = 0
    correct = 0
    len_data = 0
    total_step = len(train_loader)
    epoch_correct, epoch_train_tested, epoch_train_loss = [torch.tensor(0).float().to(rank) for _ in range(3)]
    train_loader.sampler.set_epoch(cur_epoch)
    for cur_iter, (data, target) in enumerate(train_loader):
        # adjust the learning rate for this epoch
        adjust_lr(optimizer, cur_epoch=cur_epoch, args=args)
        # transfer the data to the current GPU
        data, target = data.to(rank, non_blocking=True), target.to(rank, non_blocking=True)
        # perform the forward pass
        preds = net(data)
        # compute the loss
        loss = criterion(preds, target)
        # set the parameter gradients to zero
        optimizer.zero_grad()
        # perform the backward pass
        loss.backward()
        # update the parameters
        optimizer.step()

        # y_pred = preds.data.max(1)[1]
        # correct += y_pred.eq(target.data).sum()
        _, predictions = torch.max(preds, 1)
        epoch_correct += (predictions == target).sum()
        # compute the losses and accuracy
        epoch_train_loss += loss.item()
        epoch_train_tested += target.size(0)
        len_data += len(data)
        # print("length of data:", len_data)
        epoch_train_acc = 100. * float(correct) / len_data

        step += 1
        print("\n[Epoch [{}/{}], step [{}/{}]]".format(cur_epoch, args.epochs, step, total_step))
        # if step % 50 == 0:
        #     print("\n[Epoch [{}/{}], step [{}/{}]], Loss: {:.4f}, Acc: {:.3f}%".format(cur_epoch, args.epochs, step,
        #                                                                                total_step, loss.item(),
        #                                                                                epoch_train_acc), end='')

    # print("Epoch: ", cur_epoch, "\tTrain Loss: ", round(epoch_train_loss / len(train_loader.dataset), 5))
    #         for param_group in optimizer.param_groups:
    #             print(",  Current learning rate is: {}".format(param_group['lr']))

    #     length = len(train_loader.dataset) // args.bs
    # length = len(train_loader.dataset)
    dist.barrier()
    return epoch_correct, epoch_train_tested, epoch_train_loss


@torch.no_grad()
def eval_epoch(net, test_loader, criterion, rank):
    """Evaluates the model on the validation set."""

    # enable eval mode
    net.eval()
    ctr = 0
    # len_data = 0
    correct, total_tested, total_loss = [torch.tensor(0).float().to(rank) for _ in range(3)]
    for cur_iter, (data, target) in enumerate(test_loader):
        data, target = data.to(rank, non_blocking=True), target.to(rank, non_blocking=True)
        preds = net(data)
        # prediction = preds.data.max(1)[1]
        _, predictions = torch.max(preds, 1)
        # correct += prediction.eq(target.data).sum()
        correct += (predictions == target).sum()
        loss = criterion(preds, target)
        total_loss += loss.item()
        total_tested += target.size(0)
        ctr += 1
    dist.barrier()
    # test_acc = correct / total_tested
    # test_loss = (total_loss / len(test_loader.dataset))
    return correct, total_tested, total_loss


def train(rank, world_size, args):
    """Starts training process."""
    print("Training on Rank", rank)
    setup(rank, world_size)

    # build the model and assign to the gpu rank.
#     model = ResNet1(args, kernel_num=64).to(rank)
    torch.cuda.set_device(rank)
    torch.cuda.empty_cache()
    model = models.resnet18().to(rank)

    # # if checkpoint exists, load state dictionary.
    # state_dict = torch.load("models/pretrained/???.pt", map_location=torch.device('cpu'))
    # model.load_state_dict(state_dict)

    net = DDP(model, device_ids=[rank])
#     net = DDP(model, device_ids=[rank], find_unused_parameters=True)
    optimizer = optim.SGD(net.parameters(), lr=args.lrate, momentum=args.momentum, weight_decay=args.wd,
                          nesterov=True)
    # criterion = nn.CrossEntropyLoss().to(rank)
    criterion = nn.CrossEntropyLoss()
    # Build loaders
    # get and set the dataset paths
    data_path = '/data/tiny-imagenet-200'
    traindir = os.path.join(data_path, 'train')
    valdir = os.path.join(data_path, 'val', 'images')
    valgtdir = os.path.join(data_path, 'val', 'val_annotations.txt')
    # if rank == 0:
    #     print('train_directory={}, val_directory={}, val_ground_truth_directory={}'.format(traindir, valdir, valgtdir))

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # create training dataset and loader for multiple GPUs
    # load the dataset
    train_dataset = datasets.ImageFolder(traindir, transforms.Compose([
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    # create distributed sampler pinned to the rank
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,
                                                                    num_replicas=world_size,
                                                                    rank=rank,
                                                                    shuffle=True)
    # wrap train dataset into DataLoader
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.bs,
                                               shuffle=False,
                                               num_workers=8,
                                               pin_memory=True,
                                               sampler=train_sampler)

    # create validation dataset
    test_dataset = TinyImageNet(
        valdir, valgtdir,
        class_to_idx=train_loader.dataset.class_to_idx.copy(),
        transform=transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            normalize]))

    # create distributed sampler pinned to the rank
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset,
                                                                    num_replicas=world_size,
                                                                    rank=rank,
                                                                    shuffle=False)
    # create validation loader
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=args.bs,
                                              shuffle=False,
                                              num_workers=8,
                                              pin_memory=True,
                                              sampler=test_sampler,
                                              drop_last=False)

    # wandb.watch(model)
    start_time = time.time()

    # total_step = len(train_loader)
    for cur_epoch in range(1, args.epochs + 1):
        epoch_correct, epoch_train_tested, epoch_train_loss = train_epoch(net, train_loader, optimizer, criterion, cur_epoch, args, rank)
        # epoch_train_loss, epoch_train_acc = train_epoch(net, train_loader, optimizer, criterion, cur_epoch, args, rank)
        val_correct, val_total_tested, val_total_loss = eval_epoch(net, test_loader, criterion, rank)
        # test_loss, test_acc = eval_epoch(net, test_loader, criterion, rank)

        # wandb.log({
        #     "Test Accuracy :": test_acc,
        #     "Train Accuracy :": train_acc,
        #     "Test Loss :": test_loss,
        #     "Train Loss :": train_loss})

        epoch_acc, epoch_loss = gather_acc_and_loss(rank, world_size, epoch_correct, epoch_train_tested, epoch_train_loss)
        val_acc, val_loss = gather_acc_and_loss(rank, world_size, val_correct, val_total_tested, val_total_loss)

        if rank == 0:
            print("Epoch: ", cur_epoch, "\tTrain Loss: ", round(epoch_loss/len(train_dataset), 5),
                  "\tVal Loss: ", round(val_loss, 5),
                  "\tTrain Acc: {:.2f}%".format(epoch_acc),
                  "\tVal Acc: {:.2f}%".format(val_acc))
        dist.barrier()

    print("Total time = {:.1f} seconds. Time per epoch = {:.1f}".format(time.time() - start_time,
                                                                        (time.time() - start_time) / args.epochs))

    print("Final Evaluation on GPU ", rank)
    val_correct, val_total_tested, val_total_loss = eval_epoch(net, test_loader, criterion, rank)
    val_acc, val_loss = gather_acc_and_loss(rank, world_size, val_correct, val_total_tested, val_total_loss)

    if rank == 0:
        print("\nSaving Model")
        pretrained_path = "models/pretrained/tinyimagenet"
        os.makedirs(pretrained_path, exist_ok=True)
        filename = "resnet" + "_val_acc_" + str(int(round(val_acc * 100, 3))) + ".pt"
        torch.save(net.state_dict(), pretrained_path + "/" + filename)

    # close all processes
    dist.barrier()
    dist.destroy_process_group()
    print("closed")

def multi_proc_run(world_size, fun, args):
    """RUns a function in a multi-process setting."""

    # Spawn processes on gpus
    mp.spawn(fun, args=(world_size, args), nprocs=world_size, join=True)
    print("Run completed")


def main():
    # wandb.init(project='Deterministic-ResNet18-TinyImagenet')
    parser = argparse.ArgumentParser(description="Train a deterministic CNN on CIFAR10 dataset")

    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs (Ideally: 100)')
    parser.add_argument('--bs', type=int, default=100, help='Training batch size (Ideally: 256)')
    parser.add_argument('--lrate', type=float, default=0.005, help='Learning rate')
    parser.add_argument('--wd', type=float, default=0.05, help='Weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='Optimizer momentum')
    parser.add_argument('--dropout', type=float, default=0.5, help='Network dropout')
    parser.add_argument('--is_train', type=bool, default=True, help='True if training, False is testing')
    parser.add_argument('--train_dataset', type=str, default='tinyimagenet200',
                        help='Training datasets, cifar10 or tinyimagenet200')
    parser.add_argument('--output_size', type=float, default=200, help='Output classes')

    args = parser.parse_args()
    # wandb.config.update(args)

    print(
        'Config: Training= {}, epochs = {}, batch_size = {}, learning_rate = {}, weight_decay = {}, dropout_rate = {}'.format(
            args.is_train, args.epochs, args.bs, args.lrate, args.wd, args.dropout))

    if n_gpus > 1:
        assert n_gpus >= 2, f"Requires at least 2 GPUs to run, instead got {n_gpus}"
        print('Using ', torch.cuda.device_count(), 'GPUs')
        multi_proc_run(world_size=n_gpus, fun=train, args=args)
    else:
        print("ERROR! This code runs on DDP. Try using > 1 GPUs.")
        # print('Using ', torch.cuda.device_count(), 'GPU')
        # train(world_size=1, args)


if __name__ == '__main__':
    main()
