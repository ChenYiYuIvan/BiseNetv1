import argparse
import json
import os
import torch
from torch import nn
import wandb
from dataset.Cityscapes import Cityscapes
from torch.utils.data import DataLoader
from model.build_BiSeNet import BiSeNet
import numpy as np
from utils import reverse_one_hot, compute_global_accuracy, fast_hist, per_class_iu


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
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs to train for')
    parser.add_argument('--validation_step', type=int, default=10, help='How often to perform validation (epochs)')
    parser.add_argument('--num_classes', type=int, default=19, help='num of object classes (with void)')
    parser.add_argument('--num_workers', type=int, default=4, help='num of workers')
    parser.add_argument('--batch_size', type=int, default=1, help='Number of images in each batch')
    parser.add_argument('--loss', type=str, default='crossentropy', help='loss function, dice or crossentropy')
    parser.add_argument('--artifact1_path', type=str, default=None, help='path of artifact 1')
    parser.add_argument('--artifact2_path', type=str, default=None, help='path of artifact 2')
    parser.add_argument('--artifact3_path', type=str, default=None, help='path of artifact 3')
    args = parser.parse_args(params)
    
    # define models
    model1 = BiSeNet(args.num_classes, "resnet101")
    model2 = BiSeNet(args.num_classes, "resnet101")
    model3 = BiSeNet(args.num_classes, "resnet101")
    
    if torch.cuda.is_available():
        model1 = torch.nn.DataParallel(model1).cuda()
        model2 = torch.nn.DataParallel(model2).cuda()
        model3 = torch.nn.DataParallel(model3).cuda()

    # creating table to store metrics for wandb
    metrics_rows = []
    with open('./dataset/info.json', 'r') as f:
        data = json.load(f)
        labels = data['label'][:-1]
        labels = [f'IoU_{label}' for label in labels]
        columns = ['epoch', 'accuracy', 'mIoU']
        columns.extend(labels)

    wandb.login()
    with wandb.init(project="bisenet", entity="mldlproj1gr2", config=vars(args)) as run:
        config = wandb.config

        # define test dataset
        dataset_val = Cityscapes('../Cityscapes', 'val', [512, 1024], False)
        dataloader_val = DataLoader(dataset_val,
                                    batch_size=args.batch_size,
                                    shuffle=False,
                                    num_workers=args.num_workers,
                                    pin_memory=True)

        artifact1 = run.use_artifact(config.artifact1_path, type='model')
        artifact2 = run.use_artifact(config.artifact2_path, type='model')
        artifact3 = run.use_artifact(config.artifact3_path, type='model')

        step = 0
        max_miou = 0
        for epoch in range(config.num_epochs):
            step += 125 # num_images / batch_size = 500 / 4
            if epoch % config.validation_step == config.validation_step - 1:

                # load trained models
                model_path1 = artifact1.get_path(f'bisenet_trained_fda_{epoch}_beta0_01_adv.pth').download()
                model1.load_state_dict(torch.load(model_path1))

                model_path2 = artifact2.get_path(f'bisenet_trained_fda_{epoch}_beta0_05_adv.pth').download()
                model2.load_state_dict(torch.load(model_path2))

                model_path3 = artifact3.get_path(f'bisenet_trained_fda_{epoch}_beta0_09_adv.pth').download()
                model3.load_state_dict(torch.load(model_path3))

                # evaluate with mbt
                precision, miou, miou_list = val_mbt(config, model1, model2, model3, dataloader_val)

                if miou > max_miou:
                    max_miou = miou
                    run.summary['max_mIoU'] = max_miou

                metrics_rows.append([epoch, precision, miou, *miou_list])
                run.log({"accuracy": precision, "mIoU": miou}, step=step)
                run.log(dict(zip(labels, miou_list)), step=step)

                # remove models from local directory to save space
                os.remove(model_path1)
                os.remove(model_path2)
                os.remove(model_path3)
                
        metrics_table = wandb.Table(columns=columns, data=metrics_rows)
        run.log({'metrics_table': metrics_table})


if __name__ == '__main__':
    params = [
        '--artifact1_path', 'mldlproj1gr2/step2/trained_bisenet_fda_sst:v0',
        '--artifact2_path', 'mldlproj1gr2/step2/trained_bisenet_fda_sst:v1',
        '--artifact3_path', 'mldlproj1gr2/step2/trained_bisenet_fda_sst:v2',
    ]
    main(params)
