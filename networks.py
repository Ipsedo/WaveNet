import torch as th
import torch.nn as nn
import torch.nn.functional as F


class CausalConvolution(nn.Module):
    def __init__(
            self, in_channels: int, out_channels: int,
            kernel_size: int, dilation: int = 1
    ):
        super(CausalConvolution, self).__init__()

        self.__conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            dilation=dilation,
            padding=0
        )

        self.__padding = (kernel_size - 1) * dilation

    def forward(self, x: th.Tensor) -> th.Tensor:
        x_padded = F.pad(x, (self.__padding, 0), mode="constant", value=0.)
        out_conv = self.__conv(x_padded)
        return out_conv


class GatedActivationUnit(nn.Module):
    def __init__(
            self, in_channels: int, out_channels: int,
            kernel_size: int, dilation: int = 1
    ):
        super(GatedActivationUnit, self).__init__()

        self.__filter_conv = CausalConvolution(
            in_channels, out_channels,
            kernel_size, dilation=dilation
        )
        self.__gate_conv = CausalConvolution(
            in_channels, out_channels,
            kernel_size, dilation=dilation
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        out_f = self.__filter_conv(x)
        out_g = self.__gate_conv(x)

        return th.tanh(out_f) * th.sigmoid(out_g)


class ResidualSkipConnections(nn.Module):
    def __init__(
            self, in_channels: int, res_channels: int,
            skip_channels: int, kernel_size: int,
            dilation: int = 1
    ):
        super(ResidualSkipConnections, self).__init__()

        self.__gated_act_unit = GatedActivationUnit(
            in_channels, res_channels, kernel_size,
            dilation=dilation
        )

        self.__skip_conv = nn.Conv1d(
            res_channels, skip_channels,
            1, dilation=1, padding=0
        )

        self.__res_conv = nn.Conv1d(
            res_channels, res_channels,
            1, dilation=1, padding=0
        )

    def forward(self, x: th.Tensor) -> (th.Tensor, th.Tensor):
        out = self.__gated_act_unit(x)

        out_skip = self.__skip_conv(out)
        out_res = self.__res_conv(out)

        out_res = out_res + x

        return out_skip, out_res


class WaveNet(nn.Module):
    def __init__(
            self, num_block: int, num_layer: int,
            in_channels: int, residual_channel: int,
            skip_channel: int, hidden_channel: int,
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
                        else residual_channel,
                        residual_channel, skip_channel,
                        2, dilation=2 ** l_idx
                    )
                )

        self.__conv1 = nn.Conv1d(
            skip_channel, hidden_channel,
            kernel_size=1, padding=0, dilation=1
        )

        self.__conv2 = nn.Conv1d(
            hidden_channel, n_class,
            kernel_size=1, padding=0, dilation=1
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        skipped = []

        for res_conv in self.__layers:
            out, x = res_conv(x)
            skipped.append(out)

        skipped = th.stack(skipped, dim=1)
        skipped = skipped.sum(dim=1)

        out = F.relu(skipped)
        out = self.__conv1(out)
        out = F.relu(out)
        out = self.__conv2(out)

        return out

    @property
    def n_class(self) -> int:
        return self.__n_class


if __name__ == '__main__':
    conv_1 = CausalConvolution(1, 1, 2, dilation=2)

    data = th.zeros(1, 1, 10)
    data[0, 0, 1] = 1
    # x[0, 0, -1] = 1

    o = conv_1(data)
    print(o)

    wavenet = WaveNet(4, 10, 1, 32, 64, 10)

    o = wavenet(data)
    print(o.size())
    print(o)
