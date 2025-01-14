import torch
import numpy as np


def extract_ampl_phase(fft_im):
    # fft_im: size should be bx3xhxwx2
    fft_amp = fft_im[:, :, :, :, 0]**2 + fft_im[:, :, :, :, 1]**2
    fft_amp = torch.sqrt(fft_amp)
    fft_pha = torch.atan2(fft_im[:, :, :, :, 1], fft_im[:, :, :, :, 0])
    return fft_amp, fft_pha


def low_freq_mutate(amp_src, amp_trg, beta=0.1):
    _, _, h, w = amp_src.size()
    b = (np.floor(np.amin((h, w))*beta)).astype(int)     # get b
    amp_src[:, :, 0:b, 0:b] = amp_trg[:, :, 0:b, 0:b]      # top left
    amp_src[:, :, 0:b, w-b:w] = amp_trg[:, :, 0:b, w-b:w]    # top right
    amp_src[:, :, h-b:h, 0:b] = amp_trg[:, :, h-b:h, 0:b]    # bottom left
    amp_src[:, :, h-b:h, w-b:w] = amp_trg[:, :, h-b:h, w-b:w]  # bottom right
    return amp_src


def FDA_source_to_target(src_img, trg_img, beta=0.1):
    # exchange magnitude
    # input: src_img, trg_img

    # get fft of both source and target
    fft_src = torch.view_as_real(torch.fft.rfft2(src_img.clone()))
    fft_trg = torch.view_as_real(torch.fft.rfft2(trg_img.clone()))

    # extract amplitude and phase of both ffts
    amp_src, pha_src = extract_ampl_phase(fft_src.clone())
    amp_trg, pha_trg = extract_ampl_phase(fft_trg.clone())

    # replace the low frequency amplitude part of source with that from target
    amp_src_ = low_freq_mutate(amp_src.clone(), amp_trg.clone(), beta=beta)

    # recompose fft of source
    fft_src_ = torch.zeros(fft_src.size(), dtype=torch.float)
    fft_src_[:, :, :, :, 0] = torch.cos(pha_src.clone()) * amp_src_.clone()
    fft_src_[:, :, :, :, 1] = torch.sin(pha_src.clone()) * amp_src_.clone()

    # get the recomposed image: source content, target style
    _, _, imgH, imgW = src_img.size()
    src_in_trg = torch.fft.irfft2(
        torch.view_as_complex(fft_src_), s=[imgH, imgW])

    return src_in_trg
