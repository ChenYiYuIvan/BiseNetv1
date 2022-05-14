import torch
from ptflops import get_model_complexity_info
from model.build_discriminator import *

num_classes = 19

with torch.cuda.device(0):

    # define models
    model = Discriminator(num_classes)
    model_depthwise = DepthwiseSeparableDiscriminator(num_classes)

    # calculate complexities
    macs, _ = get_model_complexity_info(model, (num_classes, 1024, 512), as_strings=False,
                                        print_per_layer_stat=False, verbose=False)
    macs_depthwise, _ = get_model_complexity_info(model_depthwise, (num_classes, 1024, 512), as_strings=False,
                                                  print_per_layer_stat=False, verbose=False)


# FLOPs ~= 2 * MACS
print(f'Discriminator - FLOPs: {2 * macs}')
print(f'Depthwise separable discriminator - FLOPs: {2 * macs_depthwise}')
