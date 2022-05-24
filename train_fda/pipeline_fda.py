import json
import wandb
from torch.utils.data import DataLoader
import os
from dataset.GTA5 import GTA5
from model.build_BiSeNet import BiSeNet
import torch
from tqdm import tqdm
import numpy as np
from train_fda.fda_utils import FDA_source_to_target
from utils import denormalize_image, poly_lr_scheduler
from loss import DiceLoss
import torch.cuda.amp as amp
from dataset.Cityscapes import Cityscapes
from val import val
from torchvision import transforms as T


def make(config):

    # load datasets and dataloaders
    dataset_src = GTA5(config.data_source, 'train', [
                       config.crop_height, config.crop_width], config.data_augmentation)
    dataset_tgt = Cityscapes(config.data_target, 'train', [
                             config.crop_height, config.crop_width], config.data_augmentation)
    dataset_val = Cityscapes(config.data_target, 'val', [
                             config.crop_height, config.crop_width], config.data_augmentation)

    dataloader_src = DataLoader(dataset_src,
                                batch_size=config.batch_size,
                                shuffle=True,
                                num_workers=config.num_workers,
                                pin_memory=True)
    dataloader_tgt = DataLoader(dataset_tgt,
                                batch_size=config.batch_size,
                                shuffle=True,
                                num_workers=config.num_workers,
                                pin_memory=True)
    dataloader_val = DataLoader(dataset_val,
                                batch_size=1,
                                shuffle=False,
                                num_workers=config.num_workers,
                                pin_memory=True)

    # build model
    os.environ['CUDA_VISIBLE_DEVICES'] = config.cuda
    model = BiSeNet(config.num_classes, config.context_path)
    if torch.cuda.is_available() and config.use_gpu:
        model = torch.nn.DataParallel(model).cuda()

    # build optimizer
    if config.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(
            model.parameters(), config.learning_rate)
    elif config.optimizer == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(), config.learning_rate, momentum=0.9, weight_decay=1e-4)
    elif config.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), config.learning_rate)
    else:  # rmsprop
        print('not supported optimizer \n')
        return None

    # load pretrained model if exists
    if config.pretrained_model_path is not None:
        print('load model from %s ...' % config.pretrained_model_path)
        model.module.load_state_dict(torch.load(config.pretrained_model_path))
        print('Done!')

    # load loss function
    if config.loss == 'dice':
        criterion = DiceLoss()
    elif config.loss == 'crossentropy':
        criterion = torch.nn.CrossEntropyLoss(ignore_index=255)
    else:
        print('not supported loss function \n')
        return None

    return model, criterion, optimizer, dataloader_src, dataloader_tgt, dataloader_val


def train(config, model, loss_func, optimizer, dataloader_src, dataloader_tgt, dataloader_val, wandb_inst):
    wandb_inst.watch(model, loss_func, log_freq=config.batch_size)
    artifact = wandb.Artifact(name='trained_bisenet_fda',
                              type='model', metadata=dict(config))

    # creating table to store metrics for wandb
    metrics_rows = []
    with open('./dataset/info.json', 'r') as f:
        data = json.load(f)
        labels = data['label'][:-1]
        labels = [f'IoU_{label}' for label in labels]
        columns = ['epoch', 'accuracy', 'mIoU']
        columns.extend(labels)

        data_mean = data['mean']
        data_std = data['std']

    # creating directory to store models if it doesn't exist
    if not os.path.isdir(config.save_model_path):
        os.mkdir(config.save_model_path)

    scaler = amp.GradScaler()

    max_miou = 0
    step = 0
    for epoch in range(config.num_epochs):
        lr = poly_lr_scheduler(
            optimizer, config.learning_rate, iter=epoch, max_iter=config.num_epochs)
        model.train()

        tq = tqdm(total=len(dataloader_src) * config.batch_size)
        tq.set_description('epoch %d, lr %f' % (epoch, lr))

        loss_record = []
        for (data_src, label), (data_tgt, _) in zip(dataloader_src, dataloader_tgt):
            # denormalize image batches
            data_src = denormalize_image(data_src, data_mean, data_std)
            data_tgt = denormalize_image(data_tgt, data_mean, data_std)

            # apply FDA
            data_src2tgt = FDA_source_to_target(data_src, data_tgt, beta=0.01)

            # normalize image batches
            normalize = T.Normalize(data_mean, data_std)
            data = normalize(data_src2tgt)

            data = data.cuda()
            label = label.long().cuda()
            optimizer.zero_grad()

            with amp.autocast():
                output, output_sup1, output_sup2 = model(data)
                loss1 = loss_func(output, label)
                loss2 = loss_func(output_sup1, label)
                loss3 = loss_func(output_sup2, label)
                loss = loss1 + loss2 + loss3

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            tq.update(config.batch_size)
            tq.set_postfix(loss='%.6f' % loss)
            step += 1

            wandb_inst.log({"epoch": epoch, "loss": loss}, step=step)

            loss_record.append(loss.item())
        tq.close()
        loss_train_mean = np.mean(loss_record)

        wandb_inst.log({"loss_epoch_train": loss_train_mean}, step=step)

        print('loss for train : %f' % loss_train_mean)
        if epoch % config.checkpoint_step == config.checkpoint_step - 1:

            model_path_name = os.path.join(
                config.save_model_path, f'{config.model_name}.pth')
            torch.save(model.module.state_dict(), model_path_name)
            artifact.add_file(
                model_path_name, name=f'{config.model_name}_{epoch}.pth')

        if epoch % config.validation_step == config.validation_step - 1:
            precision, miou, miou_list = val(config, model, dataloader_val)
            if miou > max_miou:
                max_miou = miou

                model_path_name = os.path.join(
                    config.save_model_path, f'best_{config.model_name}.pth')
                torch.save(model.module.state_dict(), model_path_name)
                artifact.add_file(
                    model_path_name, name=f'best_{config.model_name}_{epoch}.pth')

                wandb_inst.summary['max_mIoU'] = max_miou

            metrics_rows.append([epoch, precision, miou, *miou_list])
            wandb_inst.log({"accuracy": precision, "mIoU": miou}, step=step)
            wandb_inst.log(dict(zip(labels, miou_list)), step=step)

    metrics_table = wandb.Table(columns=columns, data=metrics_rows)

    wandb_inst.log({'metrics_table': metrics_table})
    wandb_inst.log_artifact(artifact)
