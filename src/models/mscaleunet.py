import torch
import torch.nn as nn
import torch.nn.functional as F

from models.unet import UNet


def scale_as(x, y):
    '''
    scale x to the same size as y
    '''
    y_size = y.size(2), y.size(3)

    x_scaled = F.interpolate(
        x, size=y_size, mode='bilinear', align_corners=False)

    return x_scaled


class MscaleUNet(UNet):
    def __init__(self, input_channels):
        super(MscaleUNet, self).__init__(input_channels=input_channels)
        self.scale_attn = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, inputs, scales=[0.5, 1.0, 2.0], inference=False):
        """
        Hierarchical attention, primarily used for getting best inference
        results.
        We use attention at multiple scales, giving priority to the lower
        resolutions. For example, if we have 4 scales {0.5, 1.0, 1.5, 2.0},
        then evaluation is done as follows:
              p_joint = attn_1.5 * p_1.5 + (1 - attn_1.5) * down(p_2.0)
              p_joint = attn_1.0 * p_1.0 + (1 - attn_1.0) * down(p_joint)
              p_joint = up(attn_0.5 * p_0.5) * (1 - up(attn_0.5)) * p_joint
        The target scale is always 1.0, and 1.0 is expected to be part of the
        list of scales. When predictions are done at greater than 1.0 scale,
        the predictions are downsampled before combining with the next lower
        scale.
        Inputs:
          scales - a list of scales to evaluate
          inputs - input image tensor
        Output:
          return prediction
        """
        x_1x = inputs

        if not inference:
            scales = [0.5, 1.0]

        assert 1.0 in scales, 'expected 0.5 to be the target scale'
        # Lower resolution provides attention for higher rez predictions,
        # so we evaluate in order: high to low
        scales = sorted(scales, reverse=True)

        pred = None

        for s in scales:
            x = F.interpolate(x_1x, scale_factor=s, mode='bilinear', 
                                align_corners=False, recompute_scale_factor=True)
            outs = self._fwd(x, inference)
            cls_out = outs[1]
            attn_out = outs[0]

            if pred is None:
                pred = cls_out
            elif s >= 1.0:
                # downscale previous
                pred = scale_as(pred, cls_out)
                pred = attn_out * cls_out + (1 - attn_out) * pred
            else:
                # s < 1.0: upscale current
                cls_out = attn_out * cls_out

                cls_out = scale_as(cls_out, pred)
                attn_out = scale_as(attn_out, pred)

                pred = cls_out + (1 - attn_out) * pred

        return pred

    def _fwd(self, x, inference=False):
        x0 = self.fe(x)
        x1 = self.d1(x0)
        x2 = self.d2(x1)
        x3 = self.d3(x2)
        x4 = self.d4(x3)
        x5 = self.u1(x4, x3)
        x6 = self.u2(x5, x2)
        x7 = self.u3(x6, x1)
        x8 = self.u4(x7, x0)
        # scale attention map
        logits_attn = self.scale_attn(x8)
        # prediction
        logits = self.pred(x8)

        return logits_attn, logits