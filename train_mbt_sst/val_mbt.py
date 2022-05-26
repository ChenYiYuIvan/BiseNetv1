import argparse
import os
import torch
from torch import nn
from dataset.Cityscapes import Cityscapes
from torch.utils.data import DataLoader
from model.build_BiSeNet import BiSeNet
import numpy as np
from utils import reverse_one_hot, compute_global_accuracy, fast_hist, per_class_iu
from val import val


def val_mbt(config, model1, model2, model3, dataloader):
    print('start val!')
    # label_info = get_label_info(csv_path)

    softmax = nn.Softmax(dim=1)
    with torch.no_grad():
        model1.eval()
        model2.eval()
        model3.eval()

        precision_record = []
        hist = np.zeros((config.num_classes, config.num_classes))
        for i, (data, label) in enumerate(dataloader):
            label = label.type(torch.LongTensor)
            data = data.cuda()
            label = label.long().cuda()

            ## get RGB predict image
            predict1 = model1(data)
            predict1 = softmax(predict1)

            predict2 = model2(data)
            predict2 = softmax(predict2)

            predict3 = model3(data)
            predict3 = softmax(predict3)

            predict = (predict1 + predict2 + predict3) / 3
            predict = torch.argmax(predict, dim=1)
            predict = np.array(predict.cpu())

            # get RGB label image
            label = label.squeeze()
            if config.loss == 'dice':
                label = reverse_one_hot(label)
            label = np.array(label.cpu())

            # compute per pixel accuracy
            precision = compute_global_accuracy(predict, label)
            hist += fast_hist(label.flatten(),
                              predict.flatten(), config.num_classes)

            # there is no need to transform the one-hot array to visual RGB array
            # predict = colour_code_segmentation(np.array(predict), label_info)
            # label = colour_code_segmentation(np.array(label), label_info)
            precision_record.append(precision)

        precision = np.mean(precision_record)
        miou_list = per_class_iu(hist)
        miou = np.mean(miou_list)
        print('precision per pixel for test: %.3f' % precision)
        print('mIoU for validation: %.3f' % miou)
        print(f'mIoU per class: {miou_list}')

    return precision, miou, miou_list


def main(params):

    # basic parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=19, help='num of object classes (with void)')
    parser.add_argument('--num_workers', type=int, default=4, help='num of workers')
    parser.add_argument('--batch_size', type=int, default=1, help='Number of images in each batch')
    parser.add_argument('--loss', type=str, default='crossentropy', help='loss function, dice or crossentropy')
    parser.add_argument('--save_model_path', type=str, default=None, help='path where models are saved')
    parser.add_argument('--model1_name', type=str, default=None, help='name of the model 1')
    parser.add_argument('--model2_name', type=str, default=None, help='name of the model 2')
    parser.add_argument('--model3_name', type=str, default=None, help='name of the model 3')
    args = parser.parse_args(params)
    
    # load 3 models
    model1 = BiSeNet(args.num_classes, "resnet101")
    model1_path = os.path.join(args.save_model_path, args.model1_name)
    model1.load_state_dict(torch.load(model1_path))

    model2 = BiSeNet(args.num_classes, "resnet101")
    model2_path = os.path.join(args.save_model_path, args.model2_name)
    model2.load_state_dict(torch.load(model2_path))

    model3 = BiSeNet(args.num_classes, "resnet101")
    model3_path = os.path.join(args.save_model_path, args.model3_name)
    model3.load_state_dict(torch.load(model3_path))

    if torch.cuda.is_available():
        model1 = torch.nn.DataParallel(model1).cuda()
        model2 = torch.nn.DataParallel(model2).cuda()
        model3 = torch.nn.DataParallel(model3).cuda()

    # define test dataset
    dataset_val = Cityscapes('../Cityscapes', 'val', [512, 1024], False)
    dataloader_val = DataLoader(dataset_val,
                                batch_size=args.batch_size,
                                shuffle=False,
                                num_workers=args.num_workers,
                                pin_memory=True)

    # evaluate models separately
    # val(args, model1, dataloader_val)
    # val(args, model2, dataloader_val)
    # val(args, model3, dataloader_val)

    # evaluate models using multi-band transfer
    val_mbt(args, model1, model2, model3, dataloader_val)


if __name__ == '__main__':
    params = [
        '--save_model_path', './checkpoints_fda',
        '--model1_name', 'best_bisenet_trained_fda_beta0.01_adv.pth',
        '--model2_name', 'best_bisenet_trained_fda_beta0.05_adv.pth',
        '--model3_name', 'best_bisenet_trained_fda_beta0.09_adv.pth',
    ]
    main(params)
