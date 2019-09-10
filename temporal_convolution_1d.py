import torch
from torch import nn
from torch.nn.utils import weight_norm


class Chomp1d(nn.Module):
    "Expects (_, _, T) input. Chomps off the end of the sequence dim T."
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
        # Select forward() method. (Do nothing if chomp_size is 0)
        self.forward = self.chomp if chomp_size else self.skip

    def chomp(self, X):
        "Usual forward operation."
        return X[:, :, :-self.chomp_size].contiguous()

    def skip(self, X):
        "Don't try to chomp, [:-0] returns empty Tensor."
        return X


class Bite1d(nn.Module):
    "Expects (_, _, T) input. Bites off the start of the sequence dim T."
    def __init__(self, bite_size):
        super(Bite1d, self).__init__()
        self.bite_size = bite_size

    def forward(self, X):
        return X[:, :, self.bite_size:].contiguous()


class CausalConv1d(nn.Module):
    """
    Simple bundling of a conv layer and a chomp layer to achieve basic causal
    convolution with none of the frills included in the Temporal CNN
    implementation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups=1, bias=True):
        super(CausalConv1d, self).__init__()

        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, stride=stride,
            padding=padding, dilation=dilation, groups=groups, bias=bias
        )
        self.chomp = Chomp1d(padding)

    def forward(self, X):
        return self.chomp(self.conv(X))


class TemporalBlock1d(nn.Module):
    """
    Pair of weight normalized causal (in time) 1d convolutional layers with
    optional dropout, and a residual skip connection. T padding is double
    needed for 'same' output, while the Chomp3d module removes all of the
    padding from the end of the sequential time dimension. Dimensionality
    reduction is applied to the skip connection if the number of input
    channels does not match the output channels.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 dilation, padding, groups=1, dropout=0, activation=nn.ReLU):
        super(TemporalBlock1d, self).__init__()

        # in_channels -> out_channels convolution, activation, and dropout
        conv1 = weight_norm(
            nn.Conv1d(
                in_channels, out_channels, kernel_size, stride=stride,
                padding=padding, dilation=dilation, groups=groups
            )
        )
        chomp1 = Chomp1d(padding)
        activation1 = activation()
        dropout1 = nn.Dropout(dropout)

        # out_channels -> out_channels convolution, activation, and dropout
        conv2 = weight_norm(
            nn.Conv1d(
                out_channels, out_channels, kernel_size, stride=stride,
                padding=padding, dilation=dilation, groups=groups
            )
        )
        chomp2 = Chomp1d(padding)
        activation2 = activation()
        dropout2 = nn.Dropout(dropout)

        # package the main block as a sequential model
        self.main_branch = nn.Sequential(
            conv1, chomp1, activation1, dropout1,
            conv2, chomp2, activation2, dropout2
        )

        # sz 1 kernel convolution to adjust dimensionality of skip connection
        self.downsample = nn.Conv1d(
            in_channels, out_channels, 1
        ) if in_channels != out_channels else None
        self.activation = activation()

        self.init_weights()

    def init_weights(self):
        "Re-initialize weight standard deviation; 0.1 -> .01"
        self.main_branch[0].weight.data.normal_(0, 0.01)  # conv1
        self.main_branch[4].weight.data.normal_(0, 0.01)  # conv2
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, X):
        "Residual connection dimensionality reduced/increased to match main."
        out = self.main_branch(X)
        res = X if self.downsample is None else self.downsample(X)
        return self.activation(out + res)


class TemporalConv1dStack(nn.Module):
    """
    Build a temporal convolutional network by stacking exponentionally dilated
    causal convolutional residual blocks (2 conv layers with skip connections).
    """
    def __init__(self, in_channels, block_channels, kernel_size=2, groups=1,
                 dropout=0, activation=nn.ReLU):
        super(TemporalConv1dStack, self).__init__()
        blocks = []
        for i, out_channels in enumerate(block_channels):
            dilation = 2**i
            padding = (kernel_size-1)*dilation  # causal padding
            blocks.append(
                TemporalBlock1d(
                    in_channels, out_channels, kernel_size, stride=1,
                    dilation=dilation, padding=padding, groups=1,
                    dropout=dropout
                )
            )
            in_channels = out_channels

        self.network = nn.Sequential(*blocks)

    def forward(self, X):
        return self.network(X)


class CausalTranspose1d(nn.Module):
    '''
    ConvTranspose outputs are GROWN by the kernel rather than shrank, and the
    padding parameter serves to cancel it out (rather than add to it). Thus
    to achieve causality in the temporal dimension, depth "anti-padding" is
    set to 0, and the implicit transpose convolution padding is "chomped" off.
    '''
    def __init__(self, in_channels, out_channels, kernel, stride, groups=1,
                 bias=True, dilation=1):
        super(CausalTranspose1d, self).__init__()

        # No symmetrical padding in temporal dimension
        padding = 0
        # asymmetrical padding to achieve 'same' dimensions despite upsampling
        out_padding = stride//2

        self.network = nn.Sequential(
            nn.ConvTranspose1d(
                in_channels, out_channels, kernel, stride, padding,
                out_padding, groups, bias, dilation,
            ),
            # remove all implicit T padding from the end (therefore causal)
            Chomp1d((kernel-1)*dilation)
        )

    def forward(self, X):
        return self.network(X)


class CausalPool1d(nn.Module):
    '''
    Provides causal padding for 1d pooling operations (op='avg' or 'max').
    Perform rolling avg (or max) operation in time, before strided
    downsampling on offset sequence. Offset with Bite1d allows
    sampling t=[1, 3, 5, ..., T]; rather than t=[0, 2, 4, ..., T-1].

    TODO: To make it possible to use kernels > 2 in time, need to do padding
    of my own before pooling since Pytorch pooling layers will not allow pad
    to be greater than half of the kernel size. (not a rush since I use kT=2)
    '''
    def __init__(self, op, kernel, stride=None):
        super(CausalPool1d, self).__init__()

        stride = kernel if stride is None else stride
        padding = kernel-1

        # rolling stride=1 operation on time
        if op == 'avg':
            pool = nn.AvgPool1d(kernel, 1, padding, count_include_pad=False)
        elif op == 'max':
            pool = nn.MaxPool1d(kernel, 1, padding)
        chomp = Chomp1d(padding)  # remove end padding

        # temporal downsampling
        bite = Bite1d(padding)  # offset sequence for sampling
        downsample = nn.AvgPool1d(1, stride)

        # package as sequential network
        self.network = nn.Sequential(pool, chomp, bite, downsample)

    def forward(self, X):
        return self.network(X)


if __name__ == '__main__':
    # use GPU if available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # generate random data and build a Temporal CNN
    data = torch.randn(5, 15, 60).to(device)  # (N, C, T)
    tcnn = TemporalConv1dStack(
        15,  # input channels
        [30, 60, 30],  # channels for each block
        kernel_size=2,
        groups=1,
        dropout=0,
        activation=nn.ReLU
    ).to(device)

    # run data through Temporal Convolution network
    out = tcnn(data)

    # causal pooling
    pool = CausalPool1d('avg', 2).to(device)
    out = pool(out)

    # convert data and outputs into numpy
    data = data.detach().cpu().numpy()
    out = out.detach().cpu().numpy()

    # input
    print('input dimensions (N, C, T)')
    print('input shape:', data.shape, end='\n\n')
    # sequence outputs
    print('output dimensions (N, C, T)')
    print('output shape:', out.shape)
