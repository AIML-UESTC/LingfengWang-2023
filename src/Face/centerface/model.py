from torch import nn

def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        if stride not in [1, 2]:
            raise AssertionError

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self,width_mult=1.0,round_nearest=8,):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1], # 0
            [6, 24, 2, 2], # 1
            [6, 32, 3, 2], # 2
            [6, 64, 4, 2], # 3
            [6, 96, 3, 1], # 4
            [6, 160, 3, 2],# 5
            [6, 320, 1, 1],# 6
        ]
        self.feat_id = [1,2,4,6]
        self.feat_channel = []

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        features = [ConvBNReLU(3, input_channel, stride=2)]

        # building inverted residual blocks
        for id,(t, c, n, s) in enumerate(inverted_residual_setting):
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
            if id in self.feat_id  :
                self.__setattr__("feature_%d"%id,nn.Sequential(*features))
                #self.__setattr__("features",nn.Sequential(*features))
                self.feat_channel.append(output_channel)
                features = []

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        y = []
        for id in self.feat_id:
            x = self.__getattr__("feature_%d"%id)(x)
            y.append(x)
        return y

class Fuse(nn.Module):
    """
    fuses two feature maps(fusion through addition)
    """
    def __init__(self, out_dim, channel):
        super(Fuse, self).__init__()
        self.out_dim = out_dim
        
        self.deconv_layers = nn.Sequential(
                    nn.ConvTranspose2d(
                        out_dim, out_dim, kernel_size=2, stride=2, padding=0,
                        output_padding=0, bias=False),
                    nn.BatchNorm2d(out_dim,eps=0.001,momentum=0.1),
                    nn.ReLU())
        
        self.lateral_layers =  nn.Sequential(
                    nn.Conv2d(channel, out_dim,
                              kernel_size=1, stride=1, bias=False),
                    nn.BatchNorm2d(out_dim,eps=0.001,momentum=0.1),
                    nn.ReLU(inplace=True))

    def forward(self, layers):
        layers = list(layers)
        
        x = self.deconv_layers(layers[0])
        y = self.lateral_layers(layers[1])
        out = x + y
        return out

class FPN(nn.Module):
    """
    constructs FPN module on input feature maps
    """
    def __init__(self, channels, out_dim = 24):
        super(FPN, self).__init__()
        channels =  channels[::-1]

        self.conv =  nn.Sequential(
                    nn.Conv2d(channels[0], out_dim,
                              kernel_size=1, stride=1, bias=False),
                    nn.BatchNorm2d(out_dim,eps=0.001,momentum=0.1),
                    nn.ReLU(inplace=True))
        
        self.p3 = Fuse(out_dim, channels[1])
        self.p4 = Fuse(out_dim, channels[2])
        self.p5 = Fuse(out_dim, channels[3])

        self.conv_last =  nn.Sequential(
                    nn.Conv2d(out_dim,out_dim,
                              kernel_size=3, stride=1, padding=1 ,bias=False),
                    nn.BatchNorm2d(out_dim,eps=1e-5,momentum=0.01),
                    nn.ReLU(inplace=True))

    def forward(self, layers):
        layers = list(layers)
        if len(layers) <= 1:
            raise AssertionError
        x = self.conv(layers[-1])

        x = self.p3([x, layers[-2]])
        x = self.p4([x, layers[-3]])
        x = self.p5([x, layers[-4]])
        
        x = self.conv_last(x)
        return x

class CenterFace(nn.Module):
    """
    four heads
    heatmap, heatmap offsets, scale, landmarks
    """
    def __init__(self, base_name='mobilenetv2',
                        heads=None,
                        head_conv=24):
        if heads is None:
            heads = {'hm':1, 'hm_offset':2, 'wh':2, 'landmarks':10}
        super(CenterFace, self).__init__()
        self.heads = heads
        
        if base_name=='mobilenetv2':
            self.backbone = MobileNetV2(width_mult=1.0)

        channels = self.backbone.feat_channel #[24, 32, 96, 320]
        self.fpn = FPN(channels, out_dim=head_conv)

        self.hm  = nn.Sequential(
                    nn.Conv2d(head_conv, heads['hm'],
                              kernel_size=1, stride=1,
                              padding=0, bias=True),
                    nn.Sigmoid()
                )
        
        self.hm_offset = nn.Conv2d(head_conv, heads['hm_offset'],
                              kernel_size=1, stride=1,
                              padding=0, bias=True)

        self.wh    = nn.Conv2d(head_conv, heads['wh'],
                              kernel_size=1, stride=1,
                              padding=0, bias=True)
        
        self.landmarks = nn.Conv2d(head_conv, heads['landmarks'],
                              kernel_size=1, stride=1,
                              padding=0, bias=True)

    def forward(self, x):
        x = self.backbone(x)
        x = self.fpn(x)
        return self.hm(x), self.wh(x), self.hm_offset(x), self.landmarks(x)
'''
import torch
input = torch.zeros([1,3,32,32])
#input = torch.zeros([1,3,320,320])
model = CenterFace()
model.eval()
res = model(input)
for key in res[0]:
    print(res[0][key].shape)
print(res[0].keys())
#print(res[0].shape)
'''
