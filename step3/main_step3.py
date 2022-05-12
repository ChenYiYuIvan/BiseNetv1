import argparse
from pipeline_step3 import make, train
from val import val
import wandb


def main(params):

    # basic parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=300, help='Number of epochs to train for')
    # parser.add_argument('--epoch_start_i', type=int, default=0, help='Start counting epochs from this number')
    parser.add_argument('--checkpoint_step', type=int, default=10, help='How often to save checkpoints (epochs)')
    parser.add_argument('--validation_step', type=int, default=10, help='How often to perform validation (epochs)')
    parser.add_argument('--dataset_source', type=str, default="Cityscapes", help='Source dataset you are using.')
    parser.add_argument('--dataset_target', type=str, default="GTA5", help='Target dataset you are using.')
    parser.add_argument('--crop_height', type=int, default=512, help='Height of cropped/resized input image to network')
    parser.add_argument('--crop_width', type=int, default=1024, help='Width of cropped/resized input image to network')
    parser.add_argument('--batch_size', type=int, default=2, help='Number of images in each batch')
    parser.add_argument('--context_path', type=str, default="resnet101",
                        help='The context path model you are using, resnet18, resnet101.')
    parser.add_argument('--learning_rate_gen', type=float, default=0.01, help='learning rate for training generator')
    parser.add_argument('--learning_rate_discr', type=float, default=0.01, help='learning rate for training discrim')
    parser.add_argument('--data_source', type=str, default='', help='path of source data')
    parser.add_argument('--data_target', type=str, default='', help='path of target data')
    # parser.add_argument('--lambda_seg', type=float, default=0.1, help='lambda for segmentation loss')
    parser.add_argument('--lambda_adv', type=float, default=0.001, help='lambda for adversarial loss')
    parser.add_argument('--num_workers', type=int, default=4, help='num of workers')
    parser.add_argument('--num_classes', type=int, default=32, help='num of object classes (with void)')
    parser.add_argument('--cuda', type=str, default='0', help='GPU ids used for training')
    parser.add_argument('--use_gpu', type=bool, default=True, help='whether to user gpu for training')
    # parser.add_argument('--pretrained_model_path', type=str, default=None, help='path to pretrained model')
    parser.add_argument('--save_model_path', type=str, default=None, help='path to save model')
    # parser.add_argument('--optimizer', type=str, default='rmsprop', help='optimizer, support rmsprop, sgd, adam')
    parser.add_argument('--loss', type=str, default='crossentropy', help='loss function, dice or crossentropy')
    parser.add_argument('--data_augmentation', dest='data_augmentation', default=False, action='store_true',
                        help='True to include data augmentation during training, False otherwise')
    parser.add_argument('--model_name', type=str, default=None, help='name of the model')

    args = parser.parse_args(params)

    # wandb pipeline
    wandb.login()
    with wandb.init(project="step3", entity="mldlproj1gr2", config=vars(args)) as run:
        config = wandb.config

        # make model, dataloader and optimizer
        model_gen, model_discr, loss_gen, loss_discr, optim_gen, optim_discr, dataloader_source, dataloader_target, \
            dataloader_val = make(config)
        # train
        train(config, model_gen, model_discr, loss_gen, loss_discr, optim_gen, optim_discr,
              dataloader_source, dataloader_target, dataloader_val, run)
        # final test
        val(config, model_gen, dataloader_val)


if __name__ == '__main__':
    params = [
        '--num_epochs', '100',
        '--learning_rate_gen', '2.5e-2',
        '--learning_rate_discr', '1e-4',
        '--data_source', '../GTA5',  # set ../Cityscapes or ../GTA5
        '--data_target', '../Cityscapes',
        '--num_workers', '4',
        '--num_classes', '19',
        '--cuda', '0',
        '--batch_size', '4',
        '--save_model_path', './checkpoints_adversarial',
        '--context_path', 'resnet101',  # set resnet18 or resnet101, only support resnet18 and resnet101
        '--crop_height', '512',
        '--crop_width', '1024',
        '--checkpoint_step', '5',
        '--validation_step', '5',
        '--data_augmentation',
        '--model_name', 'bisenet_trained',
    ]
    main(params)
