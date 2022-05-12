import torch
from ptflops import get_model_complexity_info
from model.build_BiSeNet import BiSeNet
from model.build_discriminator import *
from thop import profile

num_classes = 19

with torch.cuda.device(0):

    # define model
    model1 = BiSeNet(num_classes, "resnet101")
    model2 = Discriminator(num_classes)
    model3 = DepthwiseSeparableDiscriminator(num_classes)

    # using ptflops
    macs_ptflops, _ = get_model_complexity_info(model3, (num_classes, 1024, 512), as_strings=False,
                                             print_per_layer_stat=False, verbose=False)

    # using thop
    macs_thop, _ = profile(model3, inputs=(torch.randn(1, num_classes, 1024, 512),), verbose=False)

# FLOPs ~= 2 * MACS
print(f'ptflops - FLOPs: {2 * macs_ptflops}')
print(f'thop - FLOPs: {2 * macs_thop}')
