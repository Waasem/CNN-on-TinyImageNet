import math

import torch.nn as nn
# import torch

# Stage depths for an ImageNet model {model depth -> (d2, d3, d4, d5)}
_IN_MODEL_STAGE_DS = {
    16: (1, 2, 3, 1),
    18: (2, 2, 2, 2),
    34: (3, 4, 6, 3),
    50: (3, 4, 6, 3),
    101: (3, 4, 23, 3),
    152: (3, 8, 36, 3),
}

DIM_LIST = [64, 128, 256, 512]


def init_weights(m):
    """Performs ResNet style weight initialization."""
    if isinstance(m, nn.Conv2d):
        # Note that there is no bias due to BN
        fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(mean=0.0, std=math.sqrt(2.0 / fan_out))
    # elif isinstance(m, TalkConv2d):
    #     # Note that there is no bias due to BN
    #     ### uniform init
    #     fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels * m.params_scale
    #     ### node specific init
    #     # fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
    #     m.weight.data.normal_(mean=0.0, std=math.sqrt(2.0 / fan_out))
    #     # m.weight.data = m.weight.data*m.init_scale
    elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
        zero_init_gamma = (
                hasattr(m, 'final_bn') and m.final_bn)
        m.weight.data.fill_(0.0 if zero_init_gamma else 1.0)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(mean=0.0, std=0.01)
        if m.bias is not None:
            m.bias.data.zero_()


class ResStem(nn.Module):
    """Stem of ResNet."""

    def __init__(self, dim_in, dim_out):
        super(ResStem, self).__init__()
        # if args.train_dataset == 'cifar10':
        #     self._construct_cifar(dim_in, dim_out)
        # else:
        self._construct_tinyimagenet(dim_in, dim_out)

    # def _construct_cifar(self, dim_in, dim_out):
    #     # 3x3, BN, ReLU
    #     # self.conv = nn.Conv2d(
    #     #     dim_in, dim_out, kernel_size=3,
    #     #     stride=1, padding=1, bias=False
    #     # )
    #     self.conv = nn.Conv2d(
    #         dim_in, dim_out, kernel_size=7,
    #         stride=1, padding=3, bias=False
    #     )
    #     self.bn = nn.BatchNorm2d(dim_out, eps=cfg.BN.EPS, momentum=cfg.BN.MOM)
    #     self.relu = nn.ReLU(cfg.MEM.RELU_INPLACE)

    def _construct_tinyimagenet(self, dim_in, dim_out):
        # 7x7, BN, ReLU, pool
        self.conv = nn.Conv2d(
            dim_in, dim_out, kernel_size=(7, 7),
            stride=(2, 2), padding=3, bias=False
        )
        self.bn = nn.BatchNorm2d(dim_out, eps=1e-5, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
            return x


class BasicTransform(nn.Module):
    """Basic transformation: 3x3, 3x3"""

    def __init__(self, dim_in, dim_out, stride):
        # self.seed = seed
        super(BasicTransform, self).__init__()
        self._construct_class(dim_in, dim_out, stride)

    def _construct_class(self, dim_in, dim_out, stride):
        # 3x3, BN, ReLU
        self.a = nn.Conv2d(
            dim_in, dim_out, kernel_size=(3, 3),
            stride=stride, padding=1, bias=False
        )
        self.a_bn = nn.BatchNorm2d(dim_out, eps=1e-5, momentum=0.1)
        self.a_relu = nn.ReLU(inplace=True)

        # 3x3, BN
        self.b = nn.Conv2d(
            dim_in, dim_out, kernel_size=(3, 3),
            stride=stride, padding=1, bias=False
        )
        self.b_bn = nn.BatchNorm2d(dim_out, eps=1e-5, momentum=0.1)
        self.b_bn.final_bn = True

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x


class ResBlock(nn.Module):
    """Residual block: x + F(x)"""

    def __init__(
            self, dim_in, dim_out, stride, trans_fun):
        super(ResBlock, self).__init__()
        self._construct_class(dim_in, dim_out, stride, trans_fun)

    def _add_skip_proj(self, dim_in, dim_out, stride):
        self.proj = nn.Conv2d(
                dim_in, dim_out, kernel_size=(1, 1),
                stride=stride, padding=0, bias=False
                 )
        self.bn = nn.BatchNorm2d(dim_out, eps=1e-5, momentum=0.1)

    def _construct_class(self, dim_in, dim_out, stride, trans_fun):
        # Use skip connection with projection if dim or res change
        self.proj_block = (dim_in != dim_out) or (stride != 1)
        if self.proj_block:
            self._add_skip_proj(dim_in, dim_out, stride)
        self.f = trans_fun(dim_in, dim_out, stride)
        self.act = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        if self.proj_block:
            x = self.bn(self.proj(x)) + self.f(x)
        else:
            x = x + self.f(x)
        x = self.act(x)
        x = self.dropout(x)
        return x


class ResStage(nn.Module):
    """Stage of ResNet."""

    def __init__(
            self, dim_in, dim_out, stride, num_bs):
        super(ResStage, self).__init__()
        self._construct_class(dim_in, dim_out, stride, num_bs)
        # dim_in = 64, dim_out = dim_list[0], stride = 1, num_bs = d2

    def _construct_class(self, dim_in, dim_out, stride, num_bs):
        for i in range(num_bs):
            # Stride and dim_in apply to the first block of the stage
            b_stride = stride if i == 0 else 1
            b_dim_in = dim_in if i == 0 else dim_out
            # Retrieve the transformation function
            trans_fun = BasicTransform
            # Construct the block
            res_block = ResBlock(
                b_dim_in, dim_out, b_stride, trans_fun)
            self.add_module('b{}'.format(i + 1), res_block)
            for j in range(0):
                trans_fun = (BasicTransform + '1x1')
                # Construct the block
                res_block = ResBlock(
                    dim_out, dim_out, 1, trans_fun)
                self.add_module('b{}_{}1x1'.format(i + 1, j + 1), res_block)

    def forward(self, x):
        for block in self.children():
            x = block(x)
        return x


class ResHead(nn.Module):
    """ResNet head."""

    def __init__(self, dim_in, num_classes):
        super(ResHead, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(dim_in, num_classes, bias=True)

    def forward(self, x):
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


class ResNet(nn.Module):
    """ResNet model."""

    def __init__(self, args):
        assert args.train_dataset in ['cifar10', 'tinyimagenet200'], \
            'Training ResNet on {} is not supported'.format(args.train_dataset)
        # assert args.train_dataset in ['cifar10', 'cifar100', 'tinyimagenet200', 'imagenet'], \
        #     'Testing ResNet on {} is not supported'.format(cfg.TEST.DATASET)
        # assert cfg.TRAIN.DATASET == cfg.TEST.DATASET, \
        #     'Train and test dataset must be the same for now'
        super(ResNet, self).__init__()
        if args.train_dataset == 'cifar10':
            self.num_classes = 10
            self._construct_cifar()
        else:
            self.num_classes = 200
            self._construct_tinyimagenet()
        self.apply(init_weights)
        self.args = args

    # # ##### basic transform
    # def _construct_cifar(self):
    #     assert (cfg.MODEL.DEPTH - 2) % 6 == 0, \
    #         'Model depth should be of the format 6n + 2 for cifar'
    #     logger.info('Constructing: ResNet-{}, cifar10'.format(cfg.MODEL.DEPTH))
    #
    #     # Each stage has the same number of blocks for cifar
    #     num_blocks = int((cfg.MODEL.DEPTH - 2) / 6)
    #     # length = num of stages (excluding stem and head)
    #     dim_list = cfg.RGRAPH.DIM_LIST
    #     # Stage 1: (N, 3, 32, 32) -> (N, 16, 32, 32)*8
    #     # self.s1 = ResStem(dim_in=3, dim_out=16)
    #     self.s1 = ResStem(dim_in=3, dim_out=64)
    #     # Stage 2: (N, 16, 32, 32) -> (N, 16, 32, 32)
    #     # self.s2 = ResStage(dim_in=16, dim_out=dim_list[0], stride=1, num_bs=num_blocks)
    #     self.s2 = ResStage(dim_in=64, dim_out=dim_list[0], stride=1, num_bs=num_blocks)
    #     # Stage 3: (N, 16, 32, 32) -> (N, 32, 16, 16)
    #     self.s3 = ResStage(dim_in=dim_list[0], dim_out=dim_list[1], stride=2, num_bs=num_blocks)
    #     # Stage 4: (N, 32, 16, 16) -> (N, 64, 8, 8)
    #     self.s4 = ResStage(dim_in=dim_list[1], dim_out=dim_list[2], stride=2, num_bs=num_blocks)
    #     # Head: (N, 64, 8, 8) -> (N, num_classes)
    #     self.head = ResHead(dim_in=dim_list[2], num_classes=cfg.MODEL.NUM_CLASSES)

    # smaller imagenet
    def _construct_tinyimagenet(self):
        print('Constructing: ResNet-18, TinyImagenet')
        # Retrieve the number of blocks per stage (excluding base)
        (d2, d3, d4, d5) = _IN_MODEL_STAGE_DS[18]
        # Compute the initial inner block dim
        dim_list = DIM_LIST
        print(d2, d3, d4, d5)
        print(dim_list[0], dim_list[1], dim_list[2], dim_list[3])
        # Stage 1: (N, 3, 224, 224) -> (N, 64, 56, 56)
        self.s1 = ResStem(dim_in=3, dim_out=64)
        # Stage 2: (N, 64, 56, 56) -> (N, 256, 56, 56)
        self.s2 = ResStage(
            dim_in=64, dim_out=dim_list[0], stride=1, num_bs=d2
        )
        # Stage 3: (N, 256, 56, 56) -> (N, 512, 28, 28)
        self.s3 = ResStage(
            dim_in=dim_list[0], dim_out=dim_list[1], stride=2, num_bs=d3
        )
        # Stage 4: (N, 512, 56, 56) -> (N, 1024, 14, 14)
        self.s4 = ResStage(
            dim_in=dim_list[1], dim_out=dim_list[2], stride=2, num_bs=d4
        )
        # Stage 5: (N, 1024, 14, 14) -> (N, 2048, 7, 7)
        self.s5 = ResStage(
            dim_in=dim_list[2], dim_out=dim_list[3], stride=2, num_bs=d5
        )
        # Head: (N, 2048, 7, 7) -> (N, num_classes)
        self.head = ResHead(dim_in=dim_list[3], num_classes=self.num_classes)

    def forward(self, x):
        for module in self.children():
            print("module :{}, x.shape :{}".format(module, x.shape))
            x = module(x)
        return x
