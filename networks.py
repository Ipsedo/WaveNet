import torch as th
import torch.nn as nn


class CausalConvolution(nn.Module):
    def __init__(
            self, in_channels: int, out_channels: int,
            kernel_size: int, dilation: int = 1, **kwargs
    ):
        super(CausalConvolution, self).__init__()

        self.__conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            dilation=dilation,
            padding=(kernel_size - 1) * dilation,
            **kwargs
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        out_conv = self.__conv(x)
        out_pad = out_conv[:, :, :-self.__conv.padding[0]]
        return out_pad


class GatedActivationUnit(nn.Module):
    def __init__(
            self, in_channels: int, out_channels: int,
            kernel_size: int, dilation: int = 1,
            **kwargs
    ):
        super(GatedActivationUnit, self).__init__()

        self.__filter_conv = CausalConvolution(
            in_channels, out_channels,
            kernel_size, dilation=dilation,
            **kwargs
        )
        self.__gate_conv = CausalConvolution(
            in_channels, out_channels,
            kernel_size, dilation=dilation,
            **kwargs
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        out_f = self.__filter_conv(x)
        out_g = self.__gate_conv(x)

        return th.tanh(out_f) * th.sigmoid(out_g)


class ResidualSkipConnections(nn.Module):
    def __init__(
            self, in_channels: int, out_channels: int,
            kernel_size: int, dilation: int = 1,
            **kwargs
    ):
        super(ResidualSkipConnections, self).__init__()

        self.__gated_act_unit = GatedActivationUnit(
            in_channels, out_channels, kernel_size,
            dilation=dilation, **kwargs
        )

        self.__conv = nn.Conv1d(
            out_channels, out_channels,
            1, dilation=1, padding=0
        )

    def forward(self, x: th.Tensor) -> (th.Tensor, th.Tensor):
        out = self.__gated_act_unit(x)
        out = self.__conv(out)

        residual = out + x

        return out, residual


class WaveNet(nn.Module):
    def __init__(
            self, num_block: int, num_layer: int,
            in_channels: int, hidden_channels: int,
            n_class: int
    ):
        super(WaveNet, self).__init__()

        self.__n_class = n_class

        self.__layers = nn.ModuleList()

        for b_idx in range(num_block):
            for l_idx in range(num_layer):
                self.__layers.append(
                    ResidualSkipConnections(
                        in_channels if (b_idx == 0) & (l_idx == 0)
                        else hidden_channels,
                        hidden_channels,
                        2, dilation=2 ** l_idx
                    )
                )

        self.__conv1 = nn.Conv1d(
            hidden_channels, hidden_channels,
            kernel_size=1, padding=0, dilation=1
        )

        self.__conv2 = nn.Conv1d(
            hidden_channels, n_class,
            kernel_size=1, padding=0, dilation=1
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        skipped = []

        for res_conv in self.__layers:
            out, x = res_conv(x)
            skipped.append(out)

        skipped = th.stack(skipped, dim=1)
        skipped = skipped.sum(dim=1)

        out = th.relu(skipped)
        out = self.__conv1(out)
        out = th.relu(out)
        out = self.__conv2(out)

        return out

    @property
    def n_class(self) -> int:
        return self.__n_class


if __name__ == '__main__':
    conv_1 = CausalConvolution(1, 1, 2, dilation=2)

    x = th.zeros(1, 1, 10)
    x[0, 0, 1] = 1
    # x[0, 0, -1] = 1

    wavenet = WaveNet(4, 10, 1, 32, 10)

    o = wavenet(x)
    print(o.size())
    print(o)
