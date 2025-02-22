import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, build_norm_layer
from mmseg.ops import resize
from .decode_head import BaseDecodeHead
from ..builder import HEADS


class ZeroInitializedLayer(nn.Module):
    def __init__(self, in_channels, out_channels, norm_cfg):
        super(ZeroInitializedLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        if norm_cfg is not None:
            self.bn = build_norm_layer(norm_cfg, out_channels)[1]
        else:
            self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

        self.init_weights()

    def init_weights(self):
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        # x = self.relu(x)
        return x


class MLP(nn.Module):
    """
    Linear Embedding
    """

    def __init__(self, input_dim=2048, embed_dim=768, use_relu=True, norm_cfg=None):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)
        self.zeroLayer = ZeroInitializedLayer(input_dim, embed_dim, norm_cfg)
        self.use_relu = use_relu
        self.relu = nn.ReLU()

        nn.init.normal_(self.proj.weight, std=.02)
        nn.init.constant_(self.proj.bias, 0)

    def forward(self, inputs):  # inputs [B C H W]
        res = self.zeroLayer(inputs).flatten(2)
        x = inputs.flatten(2).transpose(1, 2)
        x = self.proj(x).permute(0, 2, 1)
        out = torch.add(res, x)
        if self.use_relu:
            out = self.relu(out)
        return out


@HEADS.register_module()
class SegFormerHeadWithRes(BaseDecodeHead):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """

    def __init__(self, feature_strides, **kwargs):
        super(SegFormerHeadWithRes, self).__init__(input_transform='multiple_select', **kwargs)
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        decoder_params = kwargs['decoder_params']
        embedding_dim = decoder_params['embed_dim']
        self.use_relu = decoder_params['use_relu']

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim, use_relu=self.use_relu,
                             norm_cfg=self.norm_cfg)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim, use_relu=self.use_relu,
                             norm_cfg=self.norm_cfg)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim, use_relu=self.use_relu,
                             norm_cfg=self.norm_cfg)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim, use_relu=self.use_relu,
                             norm_cfg=self.norm_cfg)

        self.linear_fuse = ConvModule(
            in_channels=embedding_dim * 4,
            out_channels=embedding_dim,
            kernel_size=1,
            norm_cfg=self.norm_cfg
        )

        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)

    def forward(self, inputs):
        x = self._transform_inputs(inputs)  # len=4, 1/4,1/8,1/16,1/32
        c1, c2, c3, c4 = x
        n, _, h, w = c4.shape

        _c4 = self.linear_c4(c4).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = resize(_c4, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c3 = self.linear_c3(c3).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = resize(_c3, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c2 = self.linear_c2(c2).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = resize(_c2, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c1 = self.linear_c1(c1).reshape(n, -1, c1.shape[2], c1.shape[3])

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))

        x = self.dropout(_c)
        x = self.linear_pred(x)

        return x
