import json
import wandb
from torch.utils.data import DataLoader
import os
from model.build_BiSeNet import BiSeNet
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from utils import poly_lr_scheduler
from utils import reverse_one_hot, compute_global_accuracy, fast_hist, per_class_iu
from loss import DiceLoss
import torch.cuda.amp as amp
from dataset.Cityscapes import Cityscapes


def make(config):

    # load datasets and dataloaders
    dataset_train = Cityscapes(config.data, 'train', [config.crop_height, config.crop_width], config.data_augmentation)
    dataset_val = Cityscapes(config.data, 'val', [config.crop_height, config.crop_width], config.data_augmentation)

    dataloader_train = DataLoader(dataset_train,
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
        optimizer = torch.optim.RMSprop(model.parameters(), config.learning_rate)
    elif config.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), config.learning_rate, momentum=0.9, weight_decay=1e-4)
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

    return model, criterion, optimizer, dataloader_train, dataloader_val


def train(config, model, criterion, optimizer, dataloader_train, dataloader_val, wandb_inst):
    wandb_inst.watch(model, criterion, log_freq=config.batch_size)
    artifact = wandb.Artifact(name='trained_bisenet', type='model', metadata=dict(config))
    # writer = SummaryWriter(comment=''.format(config.optimizer, config.context_path))

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
    loss_func = criterion

    max_miou = 0
    step = 0
    for epoch in range(config.num_epochs):
        lr = poly_lr_scheduler(optimizer, config.learning_rate, iter=epoch, max_iter=config.num_epochs)
        model.train()
        tq = tqdm(total=len(dataloader_train) * config.batch_size)
        tq.set_description('epoch %d, lr %f' % (epoch, lr))
        loss_record = []
        for i, (data, label) in enumerate(dataloader_train):
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
            # writer.add_scalar('loss_step', loss, step)

            loss_record.append(loss.item())
        tq.close()
        loss_train_mean = np.mean(loss_record)

        wandb_inst.log({"loss_epoch_train": loss_train_mean}, step=step)
        # writer.add_scalar('epoch/loss_epoch_train', float(loss_train_mean), epoch)

        print('loss for train : %f' % loss_train_mean)
        if epoch % config.checkpoint_step == config.checkpoint_step - 1:

            model_path_name = os.path.join(config.save_model_path, f'{config.model_name}.pth')
            torch.save(model.module.state_dict(), model_path_name)
            artifact.add_file(model_path_name, name=f'{config.model_name}_{epoch}')

        if epoch % config.validation_step == config.validation_step - 1:
            precision, miou, miou_list = val(config, model, dataloader_val)
            if miou > max_miou:
                max_miou = miou

                model_path_name = os.path.join(config.save_model_path, f'best_{config.model_name}.pth')
                torch.save(model.module.state_dict(), model_path_name)
                artifact.add_file(model_path_name, name=f'best_{config.model_name}_{epoch}')

                wandb_inst.summary['max_mIoU'] = max_miou

            metrics_rows.append([epoch, precision, miou, *miou_list])
            wandb_inst.log({"accuracy": precision, "mIoU": miou}, step=step)
            wandb_inst.log(dict(zip(labels, miou_list)), step=step)
            # writer.add_scalar('epoch/precision_val', precision, epoch)
            # writer.add_scalar('epoch/miou val', miou, epoch)

    metrics_table = wandb.Table(columns=columns, data=metrics_rows)

    wandb_inst.log({'metrics_table': metrics_table})
    wandb_inst.log_artifact(artifact)


def val(config, model, dataloader):
    print('start val!')
    # label_info = get_label_info(csv_path)
    with torch.no_grad():
        model.eval()
        precision_record = []
        hist = np.zeros((config.num_classes, config.num_classes))
        for i, (data, label) in enumerate(dataloader):
            label = label.type(torch.LongTensor)
            data = data.cuda()
            label = label.long().cuda()

            # get RGB predict image
            predict = model(data)
            predict = torch.argmax(predict, dim=1)
            predict = np.array(predict.cpu())

            # get RGB label image
            label = label.squeeze()
            if config.loss == 'dice':
                label = reverse_one_hot(label)
            label = np.array(label.cpu())

            # compute per pixel accuracy
            precision = compute_global_accuracy(predict, label)
            hist += fast_hist(label.flatten(), predict.flatten(), config.num_classes)

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
