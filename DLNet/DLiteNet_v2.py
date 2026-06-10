import torch
import torch.nn as nn
import torch.nn.functional as F
from models.DLNet.model_utils_onnx import BasicBlock, Bottleneck, Muti_AFF, segmenthead_c, MFACB, MSAF_small
from models.DLNet.shufflenetv2 import ShufNet_stageIII

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
        # ditail head
        self.detailhead = segmenthead_c(planes * 4, planes * 4, num_classes)
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
        self.rgb = self.rgb_backbone(rgb)
        self.sar = self.sar_backbone(sar)
        self.fuse = self.rgb_sar_fusion(self.rgb, self.sar)
        # info_information
        self.info_branch1 = self.context_stage1(self.fuse)
        self.info_branch2 = self.context_stage2(self.info_branch1)
        self.info_branch3 = self.context_stage3(self.info_branch2)
        # detail information
        self.detail_branch1 = self.detail_stage1(self.rgb)
        self.detail_branch2 = self.aff1(self.detail_branch1, self.info_branch1)
        self.detail_branch3 = self.detail_stage2(self.detail_branch2)
        self.detail_branch4 = self.aff2(self.detail_branch3, self.context_stage2_to_4C(self.info_branch2))
        self.detail_branch5 = self.detail_stage3(self.detail_branch4)
        # info_detail_fuse
        self.x0 = self.aff3(self.info_branch3, self.detail_branch5)
        # output
        self.x = self.seghead(self.x0)
        self.x = F.interpolate(self.x, size=[height_output, width_output], mode='bilinear', align_corners=False)
        self.y = self.detailhead(self.detail_branch5)
        self.y = F.interpolate(self.y, size=[height_output, width_output], mode='bilinear', align_corners=False)
        return self.x, self.y


if __name__ == "__main__":
    model = DLiteNet().cuda()
    model.train()

    # 模拟输入图像
    rgb = torch.randn(4, 3, 512, 512).cuda()
    sar = torch.randn(4, 1, 512, 512).cuda()

    output = model(rgb, sar)

    # 打印输出
    print("Segmentation Output:", output[0].shape)  # seg head 输出
    print("Detail Output:", output[1].shape)  # detail head 输出

    # 你可以访问中间特征图
    print("RGB Backbone Output Shape:", model.rgb.shape)
    print("SAR Backbone Output Shape:", model.sar.shape)
    print("Fusion Output Shape:", model.fuse.shape)
    print("Info Branch 1 Output Shape:", model.info_branch1.shape)
    print("Info Branch 2 Output Shape:", model.info_branch2.shape)
    print("Info Branch 3 Output Shape:", model.info_branch3.shape)
    print("Detail Branch 1 Output Shape:", model.detail_branch1.shape)
    print("Detail Branch 2 Output Shape:", model.detail_branch2.shape)
    print("Detail Branch 3 Output Shape:", model.detail_branch3.shape)
    print("Detail Branch 4 Output Shape:", model.detail_branch4.shape)
    print("Detail Branch 5 Output Shape:", model.detail_branch5.shape)
    print("Info-Detail Fusion Output Shape (x0):", model.x0.shape)
