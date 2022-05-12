import json
import os
import torch
import wandb
from torch.cuda import amp
from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm
import numpy as np
from dataset.Cityscapes import Cityscapes
from dataset.GTA5 import GTA5
from model.build_BiSeNet import BiSeNet
from model.build_discriminator import Discriminator, DepthwiseSeparableDiscriminator
from utils import poly_lr_scheduler
from val import val


def make(config):
    # load datasets and dataloaders
    dataset_source = GTA5(config.data_source, 'train', [config.crop_height, config.crop_width],
                          config.data_augmentation)
    dataloader_source = DataLoader(dataset_source,
                                   batch_size=config.batch_size,
                                   shuffle=True,
                                   num_workers=config.num_workers,
                                   pin_memory=True)

    dataset_target = Cityscapes(config.data_target, 'train', [config.crop_height, config.crop_width],
                                config.data_augmentation)
    dataloader_target = DataLoader(dataset_target,
                                   batch_size=config.batch_size,
                                   shuffle=True,
                                   num_workers=config.num_workers,
                                   pin_memory=True)

    dataset_val = Cityscapes(config.data_target, 'val', [config.crop_height, config.crop_width],
                             config.data_augmentation)
    dataloader_val = DataLoader(dataset_val,
                                batch_size=1,
                                shuffle=False,
                                num_workers=config.num_workers,
                                pin_memory=True)

    # build models
    os.environ['CUDA_VISIBLE_DEVICES'] = config.cuda

    model_gen = BiSeNet(config.num_classes, config.context_path)

    if config.depthwise_separable:
        model_discr = DepthwiseSeparableDiscriminator(in_ch=config.num_classes)
    else:
        model_discr = Discriminator(in_channels=config.num_classes)

    if torch.cuda.is_available() and config.use_gpu:
        model_gen = torch.nn.DataParallel(model_gen).cuda()
        model_discr = torch.nn.DataParallel(model_discr).cuda()

    # build optimizer
    optim_gen = torch.optim.SGD(model_gen.parameters(), config.learning_rate_gen, momentum=0.9, weight_decay=1e-4)
    optim_discr = torch.optim.Adam(model_discr.parameters(), config.learning_rate_discr, betas=(0.9, 0.99))

    # load loss function
    loss_gen = torch.nn.CrossEntropyLoss(ignore_index=255)
    loss_discr = torch.nn.BCEWithLogitsLoss()

    return model_gen, model_discr, loss_gen, loss_discr, optim_gen, optim_discr, \
        dataloader_source, dataloader_target, dataloader_val


def train(config, model_gen, model_discr, loss_gen, loss_discr, optim_gen, optim_discr,
          dataloader_source, dataloader_target, dataloader_val, wandb_inst):
    wandb_inst.watch(model_gen, loss_gen, log_freq=config.batch_size)
    artifact = wandb.Artifact(name='trained_bisenet_discr', type='model', metadata=dict(config))

    # creating table to store metrics for wandb
    metrics_rows = []
    with open('./dataset/info.json', 'r') as f:
        data = json.load(f)
        labels = data['label'][:-1]
        labels = [f'IoU_{label}' for label in labels]
        columns = ['epoch', 'accuracy', 'mIoU']
        columns.extend(labels)

    # creating directory to store models if it doesn't exist
    if not os.path.isdir(config.save_model_path):
        os.mkdir(config.save_model_path)

    scaler = amp.GradScaler()

    softmax = nn.Softmax(dim=1)

    # labels for adversarial training
    src_label = 0
    tgt_label = 1

    max_miou = 0
    step = 0
    for epoch in range(config.num_epochs):
        lr_gen = poly_lr_scheduler(optim_gen, config.learning_rate_gen, iter=epoch, max_iter=config.num_epochs)
        lr_discr = poly_lr_scheduler(optim_discr, config.learning_rate_discr, iter=epoch, max_iter=config.num_epochs)

        model_gen.train()
        model_discr.train()

        tq = tqdm(total=len(dataloader_source) * config.batch_size)
        tq.set_description('epoch %d, lr_gen %f, lr_discr %f' % (epoch, lr_gen, lr_discr))

        loss_seg_record = []
        loss_adv_record = []
        loss_gen_record = []  # loss_gen = lambda_seg * loss_seg + lambda_adv * loss_adv
        loss_discr_record = []
        for (data_src, label_src), (data_tgt, _) in zip(dataloader_source, dataloader_target):
            # move data to gpu
            data_src = data_src.cuda()
            label_src = label_src.long().cuda()
            data_tgt = data_tgt.cuda()

            optim_gen.zero_grad()
            optim_discr.zero_grad()

            # TRAIN GENERATOR

            # don't accumulate grads in D
            for param in model_discr.parameters():
                param.requires_grad = False

            # train with source
            with amp.autocast():
                out_seg_src, output_sup1, output_sup2 = model_gen(data_src)
                loss1 = loss_gen(out_seg_src, label_src)
                loss2 = loss_gen(output_sup1, label_src)
                loss3 = loss_gen(output_sup2, label_src)
                loss_seg = loss1 + loss2 + loss3
                loss_seg = config.lambda_seg * loss_seg

            scaler.scale(loss_seg).backward()

            # train with target
            with amp.autocast():
                out_seg_tgt, _, _ = model_gen(data_tgt)
                out_discr = model_discr(softmax(out_seg_tgt))
                loss_adv = loss_discr(out_discr, torch.full(out_discr.size(), src_label, dtype=torch.float32,
                                                            device=torch.device('cuda')))
                loss_adv = config.lambda_adv * loss_adv

            scaler.scale(loss_adv).backward()
            scaler.step(optim_gen)

            loss_gen_val = loss_seg + loss_adv
            wandb_inst.log({"loss_seg": loss_seg, "loss_adv": loss_adv}, step=step)
            wandb_inst.log({"epoch": epoch, "loss_gen_val": loss_gen_val}, step=step)
            loss_seg_record.append(loss_seg.item())
            loss_adv_record.append(loss_adv.item())
            loss_gen_record.append(loss_gen_val.item())

            # TRAIN DISCRIMINATOR

            # bring back requires_grad
            for param in model_discr.parameters():
                param.requires_grad = True

            # train with source
            with amp.autocast():
                out_seg_src = out_seg_src.detach()
                out_discr_src = model_discr(softmax(out_seg_src))
                loss_d_src = loss_discr(out_discr_src, torch.full(out_discr.size(), src_label, dtype=torch.float32,
                                                                  device=torch.device('cuda'))) / 2

            scaler.scale(loss_d_src).backward()

            # train with target
            with amp.autocast():
                out_seg_tgt = out_seg_tgt.detach()
                out_discr_tgt = model_discr(softmax(out_seg_tgt))
                loss_d_tgt = loss_discr(out_discr_tgt, torch.full(out_discr.size(), tgt_label, dtype=torch.float32,
                                                                  device=torch.device('cuda'))) / 2

            scaler.scale(loss_d_tgt).backward()
            scaler.step(optim_discr)

            loss_discr_val = loss_d_src.item() + loss_d_tgt.item()
            wandb_inst.log({"loss_discr_val": loss_discr_val}, step=step)
            loss_discr_record.append(loss_discr_val)

            tq.update(config.batch_size)
            tq.set_postfix(loss_gen_val='%.6f' % loss_gen_val, loss_discr_val='%.6f' % loss_discr_val)
            step += 1

            scaler.update()

        tq.close()
        loss_gen_train_mean = np.mean(loss_gen_record)
        loss_discr_train_mean = np.mean(loss_discr_record)

        wandb_inst.log({"loss_gen_train": loss_gen_train_mean, "loss_discr_train": loss_discr_train_mean}, step=step)

        print('loss for generator train : %f' % loss_gen_train_mean)
        print('loss for discriminator train : %f' % loss_discr_train_mean)

        if epoch % config.checkpoint_step == config.checkpoint_step - 1:
            model_path_name = os.path.join(config.save_model_path, f'{config.model_name}.pth')
            torch.save(model_gen.module.state_dict(), model_path_name)
            artifact.add_file(model_path_name, name=f'{config.model_name}_{epoch}.pth')

        if epoch % config.validation_step == config.validation_step - 1:
            precision, miou, miou_list = val(config, model_gen, dataloader_val)
            if miou > max_miou:
                max_miou = miou

                model_path_name = os.path.join(config.save_model_path, f'best_{config.model_name}.pth')
                torch.save(model_gen.module.state_dict(), model_path_name)
                artifact.add_file(model_path_name, name=f'best_{config.model_name}_{epoch}.pth')

                wandb_inst.summary['max_mIoU'] = max_miou

            metrics_rows.append([epoch, precision, miou, *miou_list])
            wandb_inst.log({"accuracy": precision, "mIoU": miou}, step=step)
            wandb_inst.log(dict(zip(labels, miou_list)), step=step)

    metrics_table = wandb.Table(columns=columns, data=metrics_rows)

    wandb_inst.log({'metrics_table': metrics_table})
    wandb_inst.log_artifact(artifact)
