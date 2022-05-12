import torch
from ptflops import get_model_complexity_info
from model.build_BiSeNet import BiSeNet
from model.build_discriminator import *
from thop import profile

with torch.cuda.device(0):

    # define model
    model1 = BiSeNet(19, "resnet101")
    model2 = Discriminator(19)
    model3 = DepthwiseSeparableDiscriminator(19)

    # using ptflops
    macs, params = get_model_complexity_info(model3, (19, 1024, 512), as_strings=False,
                                             print_per_layer_stat=False, verbose=False)

    # using thop
    macs2, params2 = profile(model3, inputs=(torch.randn(1, 19, 1024, 512),), verbose=False)

print(f'ptflops - FLOPs: {2*macs}')
print(f'thop - FLOPs: {2*macs2}')
