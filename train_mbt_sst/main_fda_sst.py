import argparse
from pipeline_fda_sst import make, train
from pipeline_fda_sst_adversarial import make_adversarial, train_adversarial
from val import val
import wandb


def main(params):

    # basic parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs to train for')
    parser.add_argument('--epoch_start_i', type=int, default=0, help='Start counting epochs from this number')
    parser.add_argument('--checkpoint_step', type=int, default=10, help='How often to save checkpoints (epochs)')
    parser.add_argument('--validation_step', type=int, default=10, help='How often to perform validation (epochs)')
    parser.add_argument('--dataset_source', type=str, default="GTA5", help='Source dataset you are using.')
    parser.add_argument('--dataset_target', type=str, default="Cityscapes", help='Target dataset you are using.')
    parser.add_argument('--crop_height', type=int, default=512, help='Height of cropped/resized input image to network')
    parser.add_argument('--crop_width', type=int, default=1024, help='Width of cropped/resized input image to network')
    parser.add_argument('--batch_size', type=int, default=4, help='Number of images in each batch')
    parser.add_argument('--context_path', type=str, default="resnet101",
                        help='The context path model you are using, resnet18, resnet101.')
    parser.add_argument('--learning_rate_gen', type=float, default=2.5e-2, help='learning rate for training generator')
    parser.add_argument('--learning_rate_discr', type=float, default=1e-4, help='learning rate for training discrim')
    parser.add_argument('--data_source', type=str, default='../GTA5', help='path of source data')
    parser.add_argument('--data_target', type=str, default='../Cityscapes', help='path of target data')
    parser.add_argument('--lambda_ent', type=float, default=0.005, help='lambda for high entropy loss')
    parser.add_argument('--lambda_adv', type=float, default=0.001, help='lambda for adversarial loss')
    parser.add_argument('--num_workers', type=int, default=4, help='num of workers')
    parser.add_argument('--num_classes', type=int, default=19, help='num of object classes (with void)')
    parser.add_argument('--cuda', type=str, default='0', help='GPU ids used for training')
    parser.add_argument('--use_gpu', type=bool, default=True, help='whether to user gpu for training')
    parser.add_argument('--pretrained_model_path', type=str, default=None, help='path to pretrained model')
    parser.add_argument('--save_model_path', type=str, default=None, help='path to save model')
    parser.add_argument('--optimizer', type=str, default='sgd', help='optimizer, support rmsprop, sgd, adam')
    parser.add_argument('--loss', type=str, default='crossentropy', help='loss function, dice or crossentropy')
    parser.add_argument('--data_augmentation', dest='data_augmentation', default=False, action='store_true',
                        help='Include to use data augmentation during training')
    parser.add_argument('--depthwise_separable', dest='depthwise_separable', default=False, action='store_true',
                        help='Include to use depthwise_separable convolution in discriminator')
    parser.add_argument('--model_name', type=str, default=None, help='name of the model')
    parser.add_argument('--adversarial', dest='adversarial', default=False, action='store_true',
                        help='True to apply adversarial learning during training, False otherwise')
    parser.add_argument('--beta', type=float, default=0.01, help='beta used for fda')

    args = parser.parse_args(params)

    # wandb pipeline
    wandb.login()
    with wandb.init(project="bisenet", entity="mldlproj1gr2", config=vars(args)) as run:
        config = wandb.config

        if not config.adversarial:  # fda without adversarial learning
            model, criterion, optimizer, dataloader_src, dataloader_tgt, dataloader_val = make(config)
            train(config, model, criterion, optimizer, dataloader_src, dataloader_tgt, dataloader_val, run)
            
        else:  # fda with adversarial learning
            model, model_discr, loss_gen, loss_discr, optim_gen, optim_discr, dataloader_source, dataloader_target, \
                dataloader_val = make_adversarial(config)
            train_adversarial(config, model, model_discr, loss_gen, loss_discr, optim_gen, optim_discr,
                dataloader_source, dataloader_target, dataloader_val, run)

        # final test
        val(config, model, dataloader_val)


if __name__ == '__main__':
    params = [
        '--num_epochs', '100',
        '--checkpoint_step', '5',
        '--validation_step', '5',
        '--save_model_path', './checkpoints_fda',
        '--model_name', 'bisenet_trained_fda',
        '--data_augmentation',
        # '--depthwise_separable',
        '--adversarial',
        '--beta', '0.01',
        '--lambda_ent', '0.005',
    ]
    main(params)
