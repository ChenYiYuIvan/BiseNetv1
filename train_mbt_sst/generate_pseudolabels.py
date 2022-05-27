import argparse
import os
import torch
from torch import nn
import numpy as np
from PIL import Image
from dataset.CityscapesPseudo import CityscapesPseudo
from torch.utils.data import DataLoader
from model.build_BiSeNet import BiSeNet


def generate_pseudolabels(model1, model2, model3, dataloader):

    softmax = nn.Softmax(dim=1)
    prediction_probs = []
    prediction_labels = []
    prediction_paths = []

    with torch.no_grad():
        model1.eval()
        model2.eval()
        model3.eval()

        for i, (data, _, _, pseudo_path) in enumerate(dataloader):
            data = data.cuda()

            # get RGB predict image
            predict1 = model1(data)
            predict1 = softmax(predict1)

            predict2 = model2(data)
            predict2 = softmax(predict2)

            predict3 = model3(data)
            predict3 = softmax(predict3)

            predict = (predict1 + predict2 + predict3) / 3
            predict = predict.squeeze()
            pred_prob, pred_label = torch.max(predict, dim=0)

            prediction_probs.append(pred_prob.cpu().numpy())
            prediction_labels.append(pred_label.cpu().numpy())
            prediction_paths.append(pseudo_path[0])

    # regularize predicted labels
    num_classes = 19
    thresholds = []
    for i in range(num_classes):
        x = prediction_probs[prediction_labels == i]
        if len(x) == 0:
            thresholds.append(0)
            continue
        x = np.sort(x)
        thresholds.append(x[np.int(np.round(len(x) * 0.66))])
    thresholds = np.array(thresholds)
    thresholds[thresholds > 0.9] = 0.9

    # create folder to store pseudolabels if it doesn't exist
    if not os.path.isdir('../Cityscapes/pseudo'):
        os.mkdir('../Cityscapes/pseudo')

    # generate pseudolabels and save them to disk
    for index in range(len(dataloader)):
        prob = prediction_probs[index]
        label = prediction_labels[index]
        path = prediction_paths[index]
        for i in range(num_classes):
            label[(prob < thresholds[i]) * (label == i)] = 255

        output = np.asanyarray(label, dtype=np.uint8)
        output = Image.fromarray(output)
        output.save(path)
        

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
    dataset = CityscapesPseudo('../Cityscapes', 'generate', [512, 1024], False)
    dataloader = DataLoader(dataset,
                            batch_size=args.batch_size,
                            shuffle=False,
                            num_workers=args.num_workers,
                            pin_memory=True)

    generate_pseudolabels(model1, model2, model3, dataloader)


if __name__ == '__main__':
    params = [
        '--save_model_path', './checkpoints_fda',
        '--model1_name', 'best_bisenet_trained_fda_beta0_01_adv.pth',
        '--model2_name', 'best_bisenet_trained_fda_beta0_05_adv.pth',
        '--model3_name', 'best_bisenet_trained_fda_beta0_09_adv.pth',
    ]
    main(params)
    