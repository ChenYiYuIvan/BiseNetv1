from dataset.Cityscapes import Cityscapes
from torch.utils.data import DataLoader
import torch
from model.build_BiSeNet import BiSeNet
from utils import denormalize_image, format_image_print, format_label_print
import matplotlib.pyplot as plt
import wandb
from torch import nn


def mbt(model1, model2, model3, data):

    softmax = nn.Softmax(dim=1)
    # get RGB predict image
    predict1 = model1(data)
    predict1 = softmax(predict1)

    predict2 = model2(data)
    predict2 = softmax(predict2)

    predict3 = model3(data)
    predict3 = softmax(predict3)

    predict = (predict1 + predict2 + predict3) / 3
    predict = torch.argmax(predict, dim=1)

    return predict


def main():
    # download models from wandb and load them

    run = wandb.init(project="testing", entity="mldlproj1gr2")

    # adversarial learning
    artifact0 = run.use_artifact(
        'mldlproj1gr2/bisenet/trained_bisenet_discr:v5', type='model')
    model_path0 = artifact0.get_path(
        'bisenet_adversarial_94.pth').download()
    model0 = BiSeNet(19, "resnet101")
    model0.load_state_dict(torch.load(model_path0))
    model0.eval()

    # FDA+adv, beta = 0.01
    artifact1 = run.use_artifact(
        'mldlproj1gr2/bisenet/trained_bisenet_discr:v6', type='model')
    model_path1 = artifact1.get_path(
        'bisenet_trained_fda_99_beta0_01_adv.pth').download()
    model1 = BiSeNet(19, "resnet101")
    model1.load_state_dict(torch.load(model_path1))
    model1.eval()

    # FDA+adv, beta = 0.05
    artifact2 = run.use_artifact(
        'mldlproj1gr2/bisenet/trained_bisenet_fda:v7', type='model')
    model_path2 = artifact2.get_path(
        'bisenet_trained_fda_89_beta0_05_adv.pth').download()
    model2 = BiSeNet(19, "resnet101")
    model2.load_state_dict(torch.load(model_path2))
    model2.eval()

    # FDA+adv, beta = 0.05
    artifact3 = run.use_artifact(
        'mldlproj1gr2/bisenet/trained_bisenet_fda:v8', type='model')
    model_path3 = artifact3.get_path(
        'bisenet_trained_fda_84_beta0_09_adv.pth').download()
    model3 = BiSeNet(19, "resnet101")
    model3.load_state_dict(torch.load(model_path3))
    model3.eval()

    # FDA+adv+SST, beta = 0.01
    artifact4 = run.use_artifact(
        'mldlproj1gr2/bisenet/trained_bisenet_fda_sst:v0', type='model')
    model_path4 = artifact4.get_path(
        'bisenet_trained_fda_94_beta0_01_adv.pth').download()
    model4 = BiSeNet(19, "resnet101")
    model4.load_state_dict(torch.load(model_path4))
    model4.eval()

    # FDA+adv+SST, beta = 0.05
    artifact5 = run.use_artifact(
        'mldlproj1gr2/bisenet/trained_bisenet_fda_sst:v1', type='model')
    model_path5 = artifact5.get_path(
        'bisenet_trained_fda_94_beta0_05_adv.pth').download()
    model5 = BiSeNet(19, "resnet101")
    model5.load_state_dict(torch.load(model_path5))
    model5.eval()

    # FDA+adv+SST, beta = 0.09
    artifact6 = run.use_artifact(
        'mldlproj1gr2/bisenet/trained_bisenet_fda_sst:v2', type='model')
    model_path6 = artifact6.get_path(
        'bisenet_trained_fda_89_beta0_09_adv.pth').download()  
    model6 = BiSeNet(19, "resnet101")
    model6.load_state_dict(torch.load(model_path6))
    model6.eval()


    # define validation dataset
    batch_size = 4
    dataset_train = Cityscapes('../Cityscapes', 'val', [512, 1024], False)
    dataloader_train = DataLoader(
        dataset_train, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        fig, axarr = plt.subplots(batch_size, 5)

        # calculate prediction for a batch
        img_batch, lbl_batch = next(iter(dataloader_train))
        img_batch_denorm = denormalize_image(img_batch, dataset_train.mean, dataset_train.std)
        pred_adv_batch = torch.argmax(model0(img_batch), dim=1)
        pred_mbt_batch = mbt(model1, model2, model3, img_batch)
        pred_sst_mbt_batch = mbt(model4, model5, model6, img_batch)

        # show results
        for idx in range(batch_size):
            img = img_batch_denorm[idx]
            gt = lbl_batch[idx]
            pred_adv = pred_adv_batch[idx]
            pred_mbt = pred_mbt_batch[idx]
            pred_sst_mbt = pred_sst_mbt_batch[idx]

            axarr[idx, 0].imshow(format_image_print(img))
            axarr[idx, 0].axis('off')
            axarr[idx, 1].imshow(format_label_print(gt, dataset_train.palette))
            axarr[idx, 1].axis('off')
            axarr[idx, 2].imshow(format_label_print(pred_adv, dataset_train.palette))
            axarr[idx, 2].axis('off')
            axarr[idx, 3].imshow(format_label_print(pred_mbt, dataset_train.palette))
            axarr[idx, 3].axis('off')
            axarr[idx, 4].imshow(format_label_print(pred_sst_mbt, dataset_train.palette))
            axarr[idx, 4].axis('off')

        axarr[0, 0].set_title('image', fontsize = 20)
        axarr[0, 1].set_title('ground truth', fontsize = 20)
        axarr[0, 2].set_title('adversarial', fontsize = 20)
        axarr[0, 3].set_title('FDA + adv + MBT', fontsize = 20)
        axarr[0, 4].set_title('FDA + adv + MBT + SST', fontsize = 20)
        
        fig.tight_layout()
        plt.show()


if __name__ == '__main__':
    main()