import math
import torch
import torch.nn as nn
from torch import Tensor

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p
class SC_conv3(nn.Module):

    def __init__(self, dim, n_div, forward):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)

        if forward == 'slicing':
            self.forward = self.forward_slicing
        elif forward == 'split_cat':
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError

    def forward_slicing(self, x: Tensor) -> Tensor:
        # only for inference
        x = x.clone()   # !!! Keep the original input intact for the residual connection later
        x[:, :self.dim_conv3, :, :] = self.partial_conv3(x[:, :self.dim_conv3, :, :])

        return x

    def forward_split_cat(self, x: Tensor) -> Tensor:
        # for training/inference
        chunklist = list(torch.chunk(x, 4, dim=1))
        chunklist[0] = self.partial_conv3(chunklist[0])
        x = torch.cat(chunklist, 1)

        return x


class PWConv(nn.Module):
    """Ghost Convolution https://github.com/huawei-noah/ghostnet."""

    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        from ultralytics.nn.modules import Conv
        """Initializes the GhostConv object with input channels, output channels, kernel size, stride, groups and
        activation.
        """
        super().__init__()
        #c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c2, k, s, None, g, act=act)

    def forward(self, x):
        """Forward propagation through a Ghost Bottleneck layer with skip connection."""
        return self.cv1(x)
class Partial_conv3(nn.Module):

    def __init__(self, dim, n_div, forward):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)

        if forward == 'slicing':
            self.forward = self.forward_slicing
        elif forward == 'split_cat':
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError

    def forward_slicing(self, x: Tensor) -> Tensor:
        # only for inference
        x = x.clone()   # !!! Keep the original input intact for the residual connection later
        x[:, :self.dim_conv3, :, :] = self.partial_conv3(x[:, :self.dim_conv3, :, :])

        return x

    def forward_split_cat(self, x: Tensor) -> Tensor:
        # for training/inference
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)

        return x

class SCF(nn.Module):
    default_act = nn.SiLU()  # default activation
    def __init__(self, c1, c2, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        #SC_conv3
        #self.pwg = PWConv(c1,c1,g=c1)
        #self.pc = Partial_conv3(c1, 4, 'split_cat')
        self.pc = SC_conv3(c1, 4, 'split_cat')
        self.pw = PWConv(c1, c2)
        self.pc1 = Partial_conv3(c2, 4, 'split_cat')

        # self.bn = nn.BatchNorm2d(c2)
        # self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        #return self.act(self.bn(self.pc1(self.pw(self.pc(x)))))
        #return self.act(self.bn(self.pw(self.pc(x)))) # (1) noways
        #return self.act(self.bn(self.pc1(self.pw(x)))) ok (2) 经过和（1）的对比验证。现在是第一层是PC卷积 tensorrt 转换就是会报错
        #return self.act(self.bn(self.pc1(self.pw(self.pc(self.pwg(x))))))(3) # 效果降低
        #return self.pc1(self.pw(self.pc(self.pwg(x)))) #(4) 比（3）的效果好 （3）才47太差了没记录 这次记录了
        return self.pc1(self.pw(self.pc(x))) #(5)
#lzy 构建模块使用fasterConv 和PWConv进行 特征融合。 命名为PFPConv
class FPF(nn.Module):
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv_1 = nn.Conv2d(c1, c2, 1, 1, bias=False)
        self.fastConv_2 = Partial_conv3(c2, 4, "split_cat")
        self.conv_3 = nn.Conv2d(c2, c2, 1, 1, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv_3(self.fastConv_2(self.conv_1(x)))))

#lzy 构建模块使用fasterConv 和PWConv进行 特征融合。 命名为PFPConv
class ResFPF(nn.Module):
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv_1 = nn.Conv2d(c1, c2, 1, 1, bias=False)
        self.fastConv_2 = Partial_conv3(c2, 4, "split_cat")
        self.conv_3 = nn.Conv2d(c2, c2, 1, 1, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return x+self.act(self.bn(self.conv_3(self.fastConv_2(self.conv_1(x)))))
class ABFFSPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1, c2, k=5,attention='ELA'):
        from ultralytics.nn.modules import Conv
        """
        Initializes the SPPF layer with given input/output channels and kernel size.

        This module is equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.pw1 = PWConv(c2, c2//2);
        self.pc = Partial_conv3(c2//2, 4, 'split_cat')
        self.pw2 = PWConv(c2//2, c2);
        if attention=='ELA':
            self.attention = ELA(c2);
        elif attention=='EMA':
            self.attention = EMA(c2);
        elif attention=='LSKA':
            self.attention = LSKA(c2,7);
        elif attention=='OD':
            self.attention = ODConv_3rd(c2,c2)
        elif attention=='SE':
            self.attention = SEAttention(c2);
        elif attention=='CA':
            self.attention = CA(c2);
        elif attention=='N':
            self.attention = attention;
        else:
            print(attention)



    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(3))
        if(self.attention=='N'):
            return self.pw2(self.pc(self.pw1(self.cv2(torch.cat(y, 1)))))

        return self.attention(self.pw2(self.pc(self.pw1(self.cv2(torch.cat(y, 1))))))

#lzy 引入ELA注意力机制
class ELA(nn.Module):
    def __init__(self, channel, kernel_size=7):
        super(ELA, self).__init__()
        self.pad = kernel_size // 2
        self.conv = nn.Conv1d(channel, channel, kernel_size=kernel_size, padding=self.pad, groups=channel, bias=False)
        self.gn = nn.GroupNorm(16, channel)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()

        # 处理高度维度
        x_h = torch.mean(x, dim=3, keepdim=True).view(b, c, h)
        x_h = self.sigmoid(self.gn(self.conv(x_h))).view(b, c, h, 1)

        # 处理宽度维度
        x_w = torch.mean(x, dim=2, keepdim=True).view(b, c, w)
        x_w = self.sigmoid(self.gn(self.conv(x_w))).view(b, c, 1, w)

        # print(x_h.shape, x_w.shape)
        # 在两个维度上应用注意力
        return x * x_h * x_w


class EMA(nn.Module):
    def __init__(self, channels, factor=8):
        super(EMA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)


class LSKA(nn.Module):
    def __init__(self, dim, k_size):
        super().__init__()

        self.k_size = k_size

        if k_size == 7:
            self.conv0h = nn.Conv2d(dim, dim, kernel_size=(1, 3), stride=(1, 1), padding=(0, (3 - 1) // 2), groups=dim)
            self.conv0v = nn.Conv2d(dim, dim, kernel_size=(3, 1), stride=(1, 1), padding=((3 - 1) // 2, 0), groups=dim)
            self.conv_spatial_h = nn.Conv2d(dim, dim, kernel_size=(1, 3), stride=(1, 1), padding=(0, 2), groups=dim,
                                            dilation=2)
            self.conv_spatial_v = nn.Conv2d(dim, dim, kernel_size=(3, 1), stride=(1, 1), padding=(2, 0), groups=dim,
                                            dilation=2)
        elif k_size == 11:
            self.conv0h = nn.Conv2d(dim, dim, kernel_size=(1, 3), stride=(1, 1), padding=(0, (3 - 1) // 2), groups=dim)
            self.conv0v = nn.Conv2d(dim, dim, kernel_size=(3, 1), stride=(1, 1), padding=((3 - 1) // 2, 0), groups=dim)
            self.conv_spatial_h = nn.Conv2d(dim, dim, kernel_size=(1, 5), stride=(1, 1), padding=(0, 4), groups=dim,
                                            dilation=2)
            self.conv_spatial_v = nn.Conv2d(dim, dim, kernel_size=(5, 1), stride=(1, 1), padding=(4, 0), groups=dim,
                                            dilation=2)
        elif k_size == 23:
            self.conv0h = nn.Conv2d(dim, dim, kernel_size=(1, 5), stride=(1, 1), padding=(0, (5 - 1) // 2), groups=dim)
            self.conv0v = nn.Conv2d(dim, dim, kernel_size=(5, 1), stride=(1, 1), padding=((5 - 1) // 2, 0), groups=dim)
            self.conv_spatial_h = nn.Conv2d(dim, dim, kernel_size=(1, 7), stride=(1, 1), padding=(0, 9), groups=dim,
                                            dilation=3)
            self.conv_spatial_v = nn.Conv2d(dim, dim, kernel_size=(7, 1), stride=(1, 1), padding=(9, 0), groups=dim,
                                            dilation=3)
        elif k_size == 35:
            self.conv0h = nn.Conv2d(dim, dim, kernel_size=(1, 5), stride=(1, 1), padding=(0, (5 - 1) // 2), groups=dim)
            self.conv0v = nn.Conv2d(dim, dim, kernel_size=(5, 1), stride=(1, 1), padding=((5 - 1) // 2, 0), groups=dim)
            self.conv_spatial_h = nn.Conv2d(dim, dim, kernel_size=(1, 11), stride=(1, 1), padding=(0, 15), groups=dim,
                                            dilation=3)
            self.conv_spatial_v = nn.Conv2d(dim, dim, kernel_size=(11, 1), stride=(1, 1), padding=(15, 0), groups=dim,
                                            dilation=3)
        elif k_size == 41:
            self.conv0h = nn.Conv2d(dim, dim, kernel_size=(1, 5), stride=(1, 1), padding=(0, (5 - 1) // 2), groups=dim)
            self.conv0v = nn.Conv2d(dim, dim, kernel_size=(5, 1), stride=(1, 1), padding=((5 - 1) // 2, 0), groups=dim)
            self.conv_spatial_h = nn.Conv2d(dim, dim, kernel_size=(1, 13), stride=(1, 1), padding=(0, 18), groups=dim,
                                            dilation=3)
            self.conv_spatial_v = nn.Conv2d(dim, dim, kernel_size=(13, 1), stride=(1, 1), padding=(18, 0), groups=dim,
                                            dilation=3)
        elif k_size == 53:
            self.conv0h = nn.Conv2d(dim, dim, kernel_size=(1, 5), stride=(1, 1), padding=(0, (5 - 1) // 2), groups=dim)
            self.conv0v = nn.Conv2d(dim, dim, kernel_size=(5, 1), stride=(1, 1), padding=((5 - 1) // 2, 0), groups=dim)
            self.conv_spatial_h = nn.Conv2d(dim, dim, kernel_size=(1, 17), stride=(1, 1), padding=(0, 24), groups=dim,
                                            dilation=3)
            self.conv_spatial_v = nn.Conv2d(dim, dim, kernel_size=(17, 1), stride=(1, 1), padding=(24, 0), groups=dim,
                                            dilation=3)

        self.conv1 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0h(x)
        attn = self.conv0v(attn)
        attn = self.conv_spatial_h(attn)
        attn = self.conv_spatial_v(attn)
        attn = self.conv1(attn)
        return u * attn

# YOLOV8改进-添加动态ODConv




class ODConv2d_3rd(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True,
                 K=4, r=1 / 16, save_parameters=False,
                 padding_mode='zeros', device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.K = K
        self.r = r
        self.save_parameters = save_parameters

        super().__init__(in_channels, out_channels, kernel_size, stride,
                         padding, dilation, groups, bias, padding_mode)

        del self.weight
        self.weight = nn.Parameter(torch.empty((
            K,
            out_channels,
            in_channels // groups,
            *self.kernel_size,
        ), **factory_kwargs))

        if bias:
            del self.bias
            self.bias = nn.Parameter(torch.empty(K, out_channels, **factory_kwargs))

        hidden_dim = max(int(in_channels * r), 16)  # 设置下限为16
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.reduction = nn.Linear(in_channels, hidden_dim)
        self.fc = nn.Conv2d(in_channels, hidden_dim, 1, bias=False)
        self.bn = nn.BatchNorm2d(hidden_dim)
        self.act = nn.ReLU(inplace=True)
        # self.act = nn.SiLU(inplace=True)

        self.fc_f = nn.Linear(hidden_dim, out_channels)
        if not save_parameters or self.kernel_size[0] * self.kernel_size[1] > 1:
            self.fc_s = nn.Linear(hidden_dim, self.kernel_size[0] * self.kernel_size[1])
        if not save_parameters or in_channels // groups > 1:
            self.fc_c = nn.Linear(hidden_dim, in_channels // groups)
        if not save_parameters or K > 1:
            self.fc_w = nn.Linear(hidden_dim, K)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        fan_out = self.kernel_size[0] * self.kernel_size[1] * self.out_channels // self.groups
        for i in range(self.K):
            self.weight.data[i].normal_(0, math.sqrt(2.0 / fan_out))
        if self.bias is not None:
            self.bias.data.zero_()

    def extra_repr(self):
        return super().extra_repr() + f', K={self.K}, r={self.r:.4}'

    def get_weight_bias(self, context):
        B, C, H, W = context.shape

        if C != self.in_channels:
            raise ValueError(
                f"Expected context{[B, C, H, W]} to have {self.in_channels} channels, but got {C} channels instead")

        # x = self.gap(context).squeeze(-1).squeeze(-1)  # B, c_in
        # x = self.reduction(x)  # B, hidden_dim
        x = self.gap(context)
        x = self.fc(x)
        if x.size(0) > 1:
            x = self.bn(x)
        x = x.squeeze(-1).squeeze(-1)
        x = self.act(x)

        attn_f = self.fc_f(x).sigmoid()  # B, c_out
        attn = attn_f.view(B, 1, -1, 1, 1, 1)  # B, 1, c_out, 1, 1, 1
        if hasattr(self, 'fc_s'):
            attn_s = self.fc_s(x).sigmoid()  # B, k * k
            attn = attn * attn_s.view(B, 1, 1, 1, *self.kernel_size)  # B, 1, c_out, 1, k, k
        if hasattr(self, 'fc_c'):
            attn_c = self.fc_c(x).sigmoid()  # B, c_in // groups
            attn = attn * attn_c.view(B, 1, 1, -1, 1, 1)  # B, 1, c_out, c_in // groups, k, k
        if hasattr(self, 'fc_w'):
            attn_w = self.fc_w(x).softmax(-1)  # B, n
            attn = attn * attn_w.view(B, -1, 1, 1, 1, 1)  # B, n, c_out, c_in // groups, k, k

        weight = (attn * self.weight).sum(1)  # B, c_out, c_in // groups, k, k
        weight = weight.view(-1, self.in_channels // self.groups, *self.kernel_size)  # B * c_out, c_in // groups, k, k

        bias = None
        if self.bias is not None:
            if hasattr(self, 'fc_w'):
                bias = attn_w @ self.bias
            else:
                bias = self.bias.tile(B, 1)
            bias = bias.view(-1)  # B * c_out

        return weight, bias

    def forward(self, input, context=None):
        B, C, H, W = input.shape

        if C != self.in_channels:
            raise ValueError(
                f"Expected input{[B, C, H, W]} to have {self.in_channels} channels, but got {C} channels instead")

        weight, bias = self.get_weight_bias(context or input)

        output = nn.functional.conv2d(
            input.view(1, B * C, H, W), weight, bias,
            self.stride, self.padding, self.dilation, B * self.groups)  # 1, B * c_out, h_out, w_out
        output = output.view(B, self.out_channels, *output.shape[2:])

        return output

    def debug(self, input, context=None):
        B, C, H, W = input.shape

        if C != self.in_channels:
            raise ValueError(
                f"Expected input{[B, C, H, W]} to have {self.in_channels} channels, but got {C} channels instead")

        output_size = [
            ((H, W)[i] + 2 * self.padding[i] - self.dilation[i] * (self.kernel_size[i] - 1) - 1) // self.stride[i] + 1
            for i in range(2)
        ]

        weight, bias = self.get_weight_bias(context or input)

        weight = weight.view(B, self.groups, self.out_channels // self.groups,
                             -1)  # B, groups, c_out // groups, c_in // groups * k * k

        unfold = nn.functional.unfold(
            input, self.kernel_size, self.dilation, self.padding, self.stride)  # B, c_in * k * k, H_out * W_out
        unfold = unfold.view(B, self.groups, -1,
                             output_size[0] * output_size[1])  # B, groups, c_in // groups * k * k, H_out * W_out

        output = weight @ unfold  # B, groups, c_out // groups, H_out * W_out
        output = output.view(B, self.out_channels, *output_size)  # B, c_out, H_out * W_out

        if bias is not None:
            output = output + bias.view(B, self.out_channels, 1, 1)

        return output
class ODConv_3rd(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, kerNums=1, g=1, p=None,
                 act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = ODConv2d_3rd(c1, c2, k, s, autopad(k, p), groups=g, K=kerNums)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))

class SEAttention(nn.Module):

    def __init__(self, channel=512,reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


##CA
# class h_sigmoid_CA(nn.Module):
#     def __init__(self, inplace=True):
#         super(h_sigmoid_CA, self).__init__()
#         self.relu = nn.ReLU6(inplace=inplace)
#
#     def forward(self, x):
#         return self.relu(x + 3) / 6
#
#
# class h_swish_CA(nn.Module):
#     def __init__(self, inplace=True):
#         super(h_swish_CA, self).__init__()
#         self.sigmoid = h_sigmoid_CA(inplace=inplace)
#
#     def forward(self, x):
#         return x * self.sigmoid(x)
class CA(nn.Module):
    def __init__(self, inp, reduction=32):
        super(CA, self).__init__()
        self.relu = nn.ReLU6(inplace=True)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        #self.act = h_swish_CA()

        self.conv_h = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        #y = self.act(y)
        y = y*(self.relu(y + 3) / 6)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h
