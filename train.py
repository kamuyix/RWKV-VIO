import argparse
import os
import warnings

import torch
import logging
from path import Path
from vio_utils import custom_transform
from dataset_process.KITTI_dataset import KITTI
from model import DeepVIO
from collections import defaultdict
from vio_utils.kitti_eval import KITTI_tester
import numpy as np
import math
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.optim.lr_scheduler import CosineAnnealingLR
# 设置参数
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data_dir', type=str, default='/mnt/data2/Mutil_data/Visual-Selective-VIO-main/data', help='path to the dataset')
parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
parser.add_argument('--save_dir', type=str, default='./results', help='path to save the result')

parser.add_argument('--train_seq', type=list, default=['00', '01', '02', '04', '06', '08', '09'], help='sequences for training')
parser.add_argument('--val_seq', type=list, default=['05', '07', '10'], help='sequences for validation')
parser.add_argument('--seed', type=int, default=0, help='random seed')

parser.add_argument('--img_w', type=int, default=512, help='image width')
parser.add_argument('--img_h', type=int, default=256, help='image height')
parser.add_argument('--v_f_len', type=int, default=512, help='visual feature length')
parser.add_argument('--i_f_len', type=int, default=256, help='imu feature length')
parser.add_argument('--fuse_method', type=str, default='cat', help='fusion method [cat, soft, hard]')
parser.add_argument('--imu_dropout', type=float, default=0, help='dropout for the IMU encoder')
parser.add_argument('--head_size_a', type=int, default=64, help='')

parser.add_argument('--rwkv_out_size', type=int, default=512)
parser.add_argument('--rnn_dropout_out', type=float, default=0.2, help='dropout for the LSTM output layer')
parser.add_argument('--rnn_dropout_between', type=float, default=0.2, help='dropout within LSTM')
parser.add_argument('--weight_decay', type=float, default=5e-6, help='weight decay for the optimizer')
parser.add_argument('--batch_size', type=int, default=16, help='batch size')
parser.add_argument('--seq_len', type=int, default=16, help='sequence length for LSTM')
parser.add_argument('--workers', type=int, default=32, help='number of workers')
parser.add_argument('--epochs_warmup', type=int, default=40, help='number of epochs for warmup')
parser.add_argument('--epochs_fine', type=int, default=20, help='number of epochs for finetuning')
parser.add_argument('--lr_warmup', type=float, default=5e-4, help='learning rate for warming up stage')
parser.add_argument('--lr_fine', type=float, default=1e-6, help='learning rate for finetuning stage')

parser.add_argument('--n_layer', type=int, default=4, help='num of rwkv')
parser.add_argument('--n_embd', type=int, default=768, help='v_f + i_f')
parser.add_argument('--dim_att', type=int, default=768, help='')
parser.add_argument('--dim_ffn', type=int, default=2048, help='')
parser.add_argument("--head_size_divisor", default=12, type=int)
parser.add_argument("--dropout", default=0.05, type=float)
parser.add_argument("--grad_cp", default=0, type=int)

parser.add_argument('--experiment_name', type=str, default='experiment', help='experiment name')
parser.add_argument('--optimizer', type=str, default='Adam', help='type of optimizer [Adam, SGD]')

parser.add_argument('--pretrain_flownet',type=str, default='./pretrain_models/flownets_bn_EPE2.459.pth.tar', help='wehther to use the pre-trained flownet')
parser.add_argument('--pretrain', type=str, default=None, help='path to the pretrained model')
parser.add_argument('--hflip', default=False, action='store_true', help='whether to use horizontal flip data augmentation')
parser.add_argument('--color', default=False, action='store_true', help='whether to use color data augmentation')

parser.add_argument('--print_frequency', type=int, default=10, help='frequency of printing loss')
parser.add_argument('--weighted', default=False, action='store_true', help='whether to use weighted loss')

args = parser.parse_args()

# 设置随机种子
torch.manual_seed(args.seed)
np.random.seed(args.seed)

def train(model, optimizer, train_loader, logger, ep, weighted=False):
    mse_losses = []
    data_len = len(train_loader)

    for i, (imgs, imus, gts, rot, weight) in enumerate(train_loader):
        imgs = imgs.to('cuda', dtype=torch.bfloat16)
        imus = imus.to('cuda', dtype=torch.bfloat16)
        gts = gts.to('cuda', dtype=torch.bfloat16)
        weight = weight.to('cuda', dtype=torch.bfloat16)

        optimizer.zero_grad()

        poses = model(imgs, imus)

        if not weighted:
            angle_loss = torch.nn.functional.mse_loss(poses[:,:,:3], gts[:, :, :3])
            translation_loss = torch.nn.functional.mse_loss(poses[:,:,3:], gts[:, :, 3:])
        else:
            weight = weight / weight.sum()
            angle_loss = (weight.unsqueeze(-1).unsqueeze(-1) * (poses[:,:,:3] - gts[:, :, :3]) ** 2).mean()
            translation_loss = (weight.unsqueeze(-1).unsqueeze(-1) * (poses[:,:,3:] - gts[:, :, 3:]) ** 2).mean()

        pose_loss = 100 * angle_loss + translation_loss

        pose_loss.backward()
        optimizer.step()

        if i % args.print_frequency == 0:
            message = f"Epoch: {ep}, iters: {i}/{data_len}, pose loss: {pose_loss.item():.6f}"
            print(message)
            logger.info(message)

        mse_losses.append(pose_loss.item())

    return mse_losses

def plot_losses(all_iters, pose_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(all_iters, pose_losses, label='Pose Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Losses')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('training_losses.png')

def main():
    # 创建目录
    experiment_dir = Path('./results')
    experiment_dir.mkdir_p()
    file_dir = experiment_dir.joinpath('{}/'.format(args.experiment_name))
    file_dir.mkdir_p()
    checkpoints_dir = file_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir_p()
    log_dir = file_dir.joinpath('logs/')
    log_dir.mkdir_p()

    # 创建日志
    logger = logging.getLogger(args.experiment_name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(str(log_dir) + '/train_%s.txt' % args.experiment_name)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info('----------------------------------------TRAINING----------------------------------')
    logger.info('PARAMETER ...')
    logger.info(args)

    warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")

    transform_train = [custom_transform.ToTensor(),
                       custom_transform.Resize((args.img_h, args.img_w))]
    if args.hflip:
        transform_train += [custom_transform.RandomHorizontalFlip()]
    if args.color:
        transform_train += [custom_transform.RandomColorAug()]
    transform_train = custom_transform.Compose(transform_train)

    train_dataset = KITTI(args.data_dir,
                        sequence_length=args.seq_len,
                        train_seqs=args.train_seq,
                        transform=transform_train
                        )
    logger.info('train_dataset: ' + str(train_dataset))

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True
    )

    # GPU 选择
    str_ids = args.gpu_ids.split(',')
    gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            gpu_ids.append(id)
    if len(gpu_ids) > 0:
        torch.cuda.set_device(gpu_ids[0])

    # 初始化测试器
    tester = KITTI_tester(args)

    # 模型初始化
    model = DeepVIO(args)

    # 是否加载预训练模型
    if args.pretrain is not None:
        model.load_state_dict(torch.load(args.pretrain))
        print('Loaded pre-trained model %s' % args.pretrain)
        logger.info('Loaded pre-trained model %s' % args.pretrain)
    else:
        print('Training from scratch')
        logger.info('Training from scratch')

    # 是否使用预训练的 FlowNet
    if args.pretrain_flownet and args.pretrain is None:
        pretrained_w = torch.load(args.pretrain_flownet, map_location='cpu')
        model_dict = model.Feature_net.state_dict()
        update_dict = {k: v for k, v in pretrained_w['state_dict'].items() if k in model_dict}
        model_dict.update(update_dict)
        model.Feature_net.load_state_dict(model_dict)

    # 模型加载到 GPU
    model.cuda(gpu_ids[0])

    # 初始化优化器
    optimizer = optim.Adam(model.parameters(), lr=args.lr_warmup, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay)
    best = float('inf')
    all_iters = []
    pose_losses = []
    iter_count = 0
    total_epochs = args.epochs_warmup + args.epochs_fine
    for ep in range(total_epochs):
        model.train()
        # 动态调整学习率
        if ep < args.epochs_warmup:
            lr = args.lr_warmup
        else:
            lr = args.lr_fine
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        message = f'Epoch: {ep}, lr: {lr:.8f}'
        print(message)
        logger.info(message)

        mse_losses = train(model, optimizer, train_loader, logger, ep)

        for mse_loss in mse_losses:
            iter_count += 1
            all_iters.append(iter_count)
            pose_losses.append(mse_loss)

        # 每10个 epoch 保存一次模型
        if (ep + 1) % 10 == 0:
            torch.save(model.state_dict(), f'{checkpoints_dir}/{ep+1:003}.pth')
            message = f'Checkpoint saved for epoch {ep+1}'
            print(message)
            logger.info(message)

        # 评估模型
        if ep >= args.epochs_warmup:
            print('Evaluating the model')
            logger.info('Evaluating the model')
            with torch.no_grad():
                model.eval()
                errors = tester.eval(model, num_gpu=len(gpu_ids))

            t_rel = np.mean([errors[i]['t_rel'] for i in range(len(errors))])
            r_rel = np.mean([errors[i]['r_rel'] for i in range(len(errors))])
            t_rmse = np.mean([errors[i]['t_rmse'] for i in range(len(errors))])
            r_rmse = np.mean([errors[i]['r_rmse'] for i in range(len(errors))])

            if t_rel < best:
                best = t_rel
                torch.save(model.state_dict(), f'{checkpoints_dir}/best_{best:.2f}.pth')

            message = f'Epoch {ep+1} evaluation finished, t_rel: {t_rel:.4f}, r_rel: {r_rel:.4f}, t_rmse: {t_rmse:.4f}, r_rmse: {r_rmse:.4f}, best t_rel: {best:.4f}'
            logger.info(message)
            print(message)

    # 训练结束后保存损失图
    plot_losses(all_iters, pose_losses)

    # 保存最终模型
    torch.save(model.state_dict(), f'{checkpoints_dir}/final_model.pth')
    message = f'Training finished, best t_rel: {best:.4f}'
    logger.info(message)
    print(message)

if __name__ == "__main__":
    main()
