from dataset.Cityscapes import Cityscapes
from torch.utils.data import DataLoader
import torch
from model.build_BiSeNet import BiSeNet
from utils import denormalize_image, format_image_print, format_label_print, get_legend_handles
import matplotlib.pyplot as plt
import wandb

# download file from wandb
run = wandb.init(project="testing", entity="mldlproj1gr2")
artifact = run.use_artifact('mldlproj1gr2/step2/trained_bisenet:v1', type='model')
model_path = artifact.get_path('bisenet_trained_99').download()

# or use trained model from local files
# model_path = 'checkpoints_adversarial/best_bisenet_adversarial_noaug.pth'

# define validation dataset
batch_size = 4
dataset_train = Cityscapes('../Cityscapes', 'val', [512, 1024], True)
dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=False)

# define and load trained model
model = BiSeNet(19, "resnet101")
model.load_state_dict(torch.load(model_path))
model.eval()

with torch.no_grad():
    fig, axarr = plt.subplots(batch_size, 3, figsize=(10, 10))

    # calculate prediction for a batch
    img_batch, lbl_batch = next(iter(dataloader_train))
    pred_batch = model(img_batch)
    pred_lbl_batch = torch.argmax(pred_batch, dim=1)

    patches = get_legend_handles(dataset_train.labels, dataset_train.palette)

    # show results
    for idx in range(batch_size):
        img = img_batch[idx]
        img_denorm = denormalize_image(img, dataset_train.mean, dataset_train.std)
        lbl = lbl_batch[idx]
        pred = pred_lbl_batch[idx]

        axarr[idx, 0].imshow(format_image_print(img_denorm))
        axarr[idx, 1].imshow(format_label_print(lbl, dataset_train.palette))
        axarr[idx, 2].imshow(format_label_print(pred, dataset_train.palette))

        plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()
