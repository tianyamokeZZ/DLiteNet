import torch
import torch.nn as nn
import torch.nn.functional as F
import time
# from models.DLNet.model_utils import BasicBlock, Bottleneck, segmenthead, AFF, ASPP, CARAFE, segmentheadCARAFE, iAFF, \
#     segmenthead_drop, Muti_AFF, segmenthead_c, DUC, SPASPP, MFACB, MSAF_small
from models.DLNet.model_utils_onnx import BasicBlock, Bottleneck, Muti_AFF, segmenthead_c, MFACB, MSAF_small
from models.DLNet.shufflenetv2 import ShufNet_stageIII
import logging
import os

# from thop import profile

BatchNorm2d = nn.BatchNorm2d
bn_mom = 0.1
algc = False


class DLiteNet(nn.Module):

    def __init__(self, num_classes=1, planes=32, augment=True):
        super(DLiteNet, self).__init__()
        self.augment = augment
        # I Branch
        self.rgb_backbone = ShufNet_stageIII(input_channels=3)  # [b,96,32,32]
        self.sar_backbone = ShufNet_stageIII(input_channels=1)  # [b,96,32,32]
        self.rgb_sar_fusion = MSAF_small(channels=planes * 2)
        # Detail branch
        self.detail_stage1 = nn.Sequential(
            BasicBlock(inplanes=planes * 2, planes=planes * 2),
            nn.Conv2d(planes * 2, planes * 4,kernel_size=1, bias=False),
            nn.BatchNorm2d(planes * 4, momentum=bn_mom),
            BasicBlock(inplanes=planes * 4, planes=planes * 4)
        )
        self.detail_stage2 = nn.Sequential(
            BasicBlock(inplanes=planes * 4, planes=planes * 4),
            BasicBlock(inplanes=planes * 4, planes=planes * 4)
        )
        self.detail_stage3 = nn.Sequential(
            BasicBlock(inplanes=planes * 4, planes=planes * 4)
        )
        # Contextual branch
        self.context_stage1 = nn.Sequential(
            MFACB(planes * 2, planes * 2, planes * 4, dilation=[2, 2, 2]),
            MFACB(planes * 4, planes * 4, planes * 4, dilation=[2, 2, 2]),
            MFACB(planes * 4, planes * 4, planes * 4, dilation=[3, 3, 3]),
        )
        self.context_stage2 = nn.Sequential(
            MFACB(planes * 4, planes * 4, planes * 8, dilation=[3, 3, 3]),
            MFACB(planes * 8, planes * 8, planes * 8, dilation=[5, 5, 5]),
        )
        self.context_stage2_to_4C = nn.Sequential(
            nn.Conv2d(planes * 8, planes * 4, kernel_size=1, bias=False),
            BatchNorm2d(planes * 4, momentum=bn_mom),
        )
        self.context_stage3 = nn.Sequential(
            Bottleneck(planes * 8, planes * 4, 1, dilation=5),  # bottleneck 输出的是planes的两倍
            nn.Conv2d(planes * 8, planes * 4, kernel_size=1, bias=False),
            BatchNorm2d(planes * 4, momentum=bn_mom),
        )
        # Context Detail Fusion
        self.aff1 = Muti_AFF(channels=planes * 4)
        self.aff2 = Muti_AFF(channels=planes * 4)
        self.aff3 = Muti_AFF(channels=planes * 4)
        # seg head
        self.seghead = segmenthead_c(planes * 4, planes * 4, num_classes)
        # initial weight
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, rgb, sar):

        width_output = rgb.shape[-1]
        height_output = rgb.shape[-2]
        rgb = self.rgb_backbone(rgb)
        sar = self.sar_backbone(sar)
        fuse = self.rgb_sar_fusion(rgb, sar)
        # info_information
        info_branch1 = self.context_stage1(fuse)
        info_branch2 = self.context_stage2(info_branch1)
        info_branch3 = self.context_stage3(info_branch2)
        # detail information
        detail_branch = self.detail_stage1(rgb)
        detail_branch = self.aff1(detail_branch, info_branch1)
        detail_branch = self.detail_stage2(detail_branch)
        detail_branch = self.aff2(detail_branch, self.context_stage2_to_4C(info_branch2))
        detail_branch = self.detail_stage3(detail_branch)
        # info_detail_fuse
        x = self.aff3(info_branch3, detail_branch)
        x = self.seghead(x)
        x = F.interpolate(x, size=[height_output, width_output], mode='bilinear', align_corners=False)
        return x


if __name__ == "__main__":
    model = DLiteNet().cuda()
    # 模型训练模式
    model.train()
    # 模拟输入图像
    rgb = torch.randn(4, 3, 512, 512).cuda()
    sar = torch.randn(4, 1, 512, 512).cuda()
    output = model(rgb, sar)
    print("output:", output.shape)
#
# from thop import profile
# def count_parameters(model):
#     return sum(p.numel() for p in model.parameters() if p.requires_grad)
# if __name__ == '__main__':
#     model = DLiteNet().to("cuda")
#     model.train()
#     input = torch.randn(1, 3, 900, 900).to("cuda")
#     input1 = torch.randn(1, 1, 900, 900).to("cuda")
#     flops, params = profile(model, (input, input1, ))
#     print('FLOPs = ' + str(flops /1000**3) + 'G')
#     print('Params = ' + str(params / 1000 ** 2) + 'M')
#     # stat(model, input_size=(3, 512, 512))
#     Params = count_parameters(model)
#     print("模型总参数量", Params)
#
#     import time
#     device = torch.device('cuda')
#     model.eval()
#     model.to(device)
#     iterations = None
#     input = torch.randn(1, 3, 900, 900).cuda()
#     input1 = torch.randn(1, 1, 900, 900).cuda()
#     with torch.no_grad():
#         for _ in range(10):
#             model(input, input1)
#
#         if iterations is None:
#             elapsed_time = 0
#             iterations = 100
#             while elapsed_time < 1:
#                 torch.cuda.synchronize()
#                 torch.cuda.synchronize()
#                 t_start = time.time()
#                 for _ in range(iterations):
#                     model(input, input1)
#                 torch.cuda.synchronize()
#                 torch.cuda.synchronize()
#                 elapsed_time = time.time() - t_start
#                 iterations *= 2
#             FPS = iterations / elapsed_time
#             iterations = int(FPS * 6)
#
#         print('=========Speed Testing=========')
#         torch.cuda.synchronize()
#         torch.cuda.synchronize()
#         t_start = time.time()
#         for _ in range(iterations):
#             model(input, input1)
#         torch.cuda.synchronize()
#         torch.cuda.synchronize()
#         elapsed_time = time.time() - t_start
#         latency = elapsed_time / iterations * 1000
#     torch.cuda.empty_cache()
#     FPS = 1000 / latency
#     print(FPS)



