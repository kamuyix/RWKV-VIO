import argparse
import os.path
import torch
import torch.optim as optim
import numpy as np
import logging
from path import Path
from model2 import DeepVIO
from dataset_process.KITTI_dataset import KITTI
from collections import defaultdict
from vio_utils.kitti_eval import KITTI_tester
import torch.optim.lr_scheduler as lr_scheduler
import matplotlib.pyplot as plt
from vio_utils import custom_transform
import warnings
import wandb
from thop import profile
from torch import nn

# 设置参数
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data_dir', type=str, default='/mnt/data2/Mutil_data/Visual-Selective-VIO-main/data',
                    help='path to the dataset')
parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
parser.add_argument('--save_dir', type=str, default='./results', help='path to save the result')
parser.add_argument('--train_seq', type=list, default=['00', '01', '02', '04', '06', '08','05', '07'],
                    help='sequences for training')
parser.add_argument('--val_seq', type=list, default=['09','10'], help='sequences for validation')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--img_w', type=int, default=512, help='image width')
parser.add_argument('--img_h', type=int, default=256, help='image height')
parser.add_argument('--v_f_len', type=int, default=512, help='visual feature length')
parser.add_argument('--i_f_len', type=int, default=256, help='imu feature length')
parser.add_argument('--fuse_method', type=str, default='cat', help='fusion method [cat, soft, hard]')
parser.add_argument('--imu_dropout', type=float, default=0, help='dropout for the IMU encoder')
parser.add_argument('--rwkv_dropout_out', type=float, default=0.02)
parser.add_argument('--n_layer', type=int, default=1, help='num of rwkv layers')
parser.add_argument('--n_heads', type=int, default=8, help='num of attention heads')
parser.add_argument('--n_embd', type=int, default=512, help='embedding dimension')
parser.add_argument('--pretrain', type=str, default=None, help='path to the pretrained model')
parser.add_argument('--optimizer', type=str, default='Adam', help='type of optimizer [Adam, SGD]')
parser.add_argument('--batch_size', type=int, default=16, help='batch size')
parser.add_argument('--seq_len', type=int, default=11, help='sequence length for RWKV')
parser.add_argument('--workers', type=int, default=32, help='number of workers')
parser.add_argument('--epochs_warmup', type=int, default=20, help='number of epochs for warmup')
parser.add_argument('--epochs_fine', type=int, default=60, help='number of epochs for finetuning')
parser.add_argument('--lr_warmup', type=float, default=5e-4, help='learning rate for warming up stage')
parser.add_argument('--lr_fine', type=float, default=1e-4, help='learning rate for finetuning stage')
parser.add_argument('--experiment_name', type=str, default='experiment_12_21', help='experiment name')
parser.add_argument('--hflip', default=True, action='store_true',help='whether to use horizontal flip data augmentation')
parser.add_argument('--pretrain_flownet', type=str, default='/mnt/data2/Mutil_data/Visual-Selective-VIO-main/pretrain_models/flownets_bn_EPE2.459.pth.tar', help='wehther to use the pre-trained flownet')
parser.add_argument('--color', default=True, action='store_true', help='whether to use color data augmentation')
parser.add_argument('--print_frequency', type=int, default=10, help='frequency of printing loss')
parser.add_argument('--weighted', default=False, action='store_true', help='whether to use weighted loss')
parser.add_argument('--weight_decay', type=float, default=5e-6)

args = parser.parse_args()

# 设置随机种子
torch.manual_seed(args.seed)
np.random.seed(args.seed)

def update_status(ep, args):
    if ep < args.epochs_warmup:  # Warmup stage
        lr = args.lr_warmup
    elif ep >= args.epochs_warmup and ep < args.epochs_warmup + args.epochs_fine:  # fine training stage
        lr = args.lr_fine
    # wandb.log({"learning_rate": lr, "epoch": ep})
    return lr


def train(model, optimizer, train_loader, logger, ep, weighted=False):
    mse_losses = []
    data_len = len(train_loader)
    for i, (imgs, imus, gts, rot, weight) in enumerate(train_loader):
        imgs = imgs.to('cuda', dtype=torch.float)
        imus = imus.to('cuda', dtype=torch.float)
        gts = gts.to('cuda', dtype=torch.float)
        weight = weight.to('cuda', dtype=torch.float)
        optimizer.zero_grad()
        poses = model(imgs, imus)  # Assuming model returns poses and consistency_score

        if not weighted:
            angle_loss = torch.nn.functional.mse_loss(poses[:, :, :3], gts[:, :, :3])
            translation_loss = torch.nn.functional.mse_loss(poses[:, :, 3:], gts[:, :, 3:])
        else:
            weight = weight / weight.sum()
            angle_loss = (weight.unsqueeze(-1).unsqueeze(-1) * (poses[:, :, :3] - gts[:, :, :3]) ** 2).mean()
            translation_loss = (weight.unsqueeze(-1).unsqueeze(-1) * (poses[:, :, 3:] - gts[:, :, 3:]) ** 2).mean()
        pose_loss = 100 * angle_loss + translation_loss
        loss = pose_loss.to(torch.float)
        loss.backward()
        optimizer.step()
        # print(f"consistency_penalty: {consistency_penalty}")
        # print(f"pose loss:{pose_loss}, loss: {loss}")
        if i % args.print_frequency*8 == 0:
            message = f"Epoch: {ep}, iters: {i}/{data_len}, pose loss: {pose_loss.item():.6f}, loss: {loss.item():.6f}"
            print(message)
            logger.info(message)
            # wandb.log({"pose_loss": pose_loss.item()})
        mse_losses.append(pose_loss.item())
    return np.mean(mse_losses)


def plot_losses(all_epochs, train_losses, save_dir):
    plt.figure(figsize=(10, 5))
    plt.plot(all_epochs, train_losses, label='Pose Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Losses')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_losses.png'))
    plt.close()


def main():
    # wandb.login()
    # wandb.init(
    #     project="RWKV-VIO",
    #     name=args.experiment_name,
    #     config=args
    # )
    experiment_dir = Path('./results')
    experiment_dir.mkdir_p()
    file_dir = experiment_dir.joinpath('{}/'.format(args.experiment_name))
    file_dir.mkdir_p()
    checkpoints_dir = file_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir_p()
    log_dir = file_dir.joinpath('logs/')
    log_dir.mkdir_p()

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

    str_ids = args.gpu_ids.split(',')
    gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            gpu_ids.append(id)
    if len(gpu_ids) > 0:
        torch.cuda.set_device(gpu_ids[0])

    tester = KITTI_tester(args)
    model = DeepVIO(args)
    total_params = sum(p.numel() for p in model.parameters())
    for name, p in model.named_parameters():
        print(f"net name: {name} -- params num: {p.numel()/1000000} M")
    visual_encoder_params = sum(p.numel() for p in model.Feature_net.conv1.parameters())
    visual_encoder_params += sum(p.numel() for p in model.Feature_net.conv2.parameters())
    visual_encoder_params += sum(p.numel() for p in model.Feature_net.conv3.parameters())
    visual_encoder_params += sum(p.numel() for p in model.Feature_net.conv3_1.parameters())
    visual_encoder_params += sum(p.numel() for p in model.Feature_net.conv4.parameters())
    visual_encoder_params += sum(p.numel() for p in model.Feature_net.conv4_1.parameters())
    visual_encoder_params += sum(p.numel() for p in model.Feature_net.conv5.parameters())
    visual_encoder_params += sum(p.numel() for p in model.Feature_net.conv5_1.parameters())
    visual_encoder_params += sum(p.numel() for p in model.Feature_net.conv6.parameters())
    visual_encoder_params += sum(p.numel() for p in model.Feature_net.visual_head.parameters())
    remaining_params = total_params - visual_encoder_params
    logger.info(f'Total number of parameters: {total_params/1000000:,}')
    logger.info(f'remaining_params: {remaining_params/1000000:,}')
    logger.info('Model layers counts:')
    for name, param in model.named_parameters():
        if param.requires_grad:
            logger.info(f'{name}: {param.numel()/1000000} parameters')
    rwkv_params = sum(p.numel() for p in model.Pose_net.rwkv.parameters())
    regressor_params = sum(p.numel() for p in model.Pose_net.regressor.parameters())
    inertial_encoder_params = sum(p.numel() for p in model.Feature_net.inertial_encoder.parameters())
    remaining_params = total_params - visual_encoder_params
    print('total = ', total_params/1000000, 'M')
    print("Number of parameters in the visual encoder:", visual_encoder_params/1000000, 'M')
    print("Number of parameters in the inertial_encoder_params:", inertial_encoder_params/1000000, 'M')
    print("Number of parameters in the rwkv:", rwkv_params/1000000, 'M')
    print("Number of parameters in the regressor:", regressor_params/1000000, 'M')
    print("Number of parameters in the rest of the model:", remaining_params/1000000, 'M')


    if args.pretrain is not None:
        model.load_state_dict(torch.load(args.pretrain))
        print('load model %s' % args.pretrain)
        logger.info('load model %s' % args.pretrain)
    else:
        print('Training from scratch')
        logger.info('Training from scratch')


    # Use the pre-trained flownet or not
    # if args.pretrain_flownet and args.pretrain is None:
    #     pretrained_w = torch.load(args.pretrain_flownet, map_location='cpu')
    #     model_dict = model.Feature_net.state_dict()
    #     update_dict = {k: v for k, v in pretrained_w['state_dict'].items() if k in model_dict}
    #     model_dict.update(update_dict)
    #     model.Feature_net.load_state_dict(model_dict)
    #     print(f"load model from {args.pretrain_flownet}")

    model.cuda(gpu_ids[0])
    # wandb.watch(model, log="all")
    best = 10000
    optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay)
    all_epochs = []
    train_losses = []
    total_epochs = args.epochs_warmup + args.epochs_fine

    for ep in range(total_epochs):
        lr = update_status(ep, args)
        optimizer.param_groups[0]['lr'] = lr

        message = f'Epoch: {ep}, lr: {lr:.8f}'
        print(message)
        logger.info(message)

        model.train()
        avg_pose_loss = train(model, optimizer, train_loader, logger, ep)
        all_epochs.append(ep)
        train_losses.append(avg_pose_loss)

        # 如果 warmup 阶段已经结束，才进行验证集评估
        if ep >= args.epochs_warmup:
            model.eval()  # 切换到评估模式
            # 执行原来的 tester.eval() 计算 errors
            print('Evaluating the model')
            logger.info('Evaluating the model')
            with torch.no_grad():
                errors = tester.eval(model, num_gpu=len(gpu_ids))

            t_rel = np.mean([errors[i]['t_rel'] for i in range(len(errors))])
            r_rel = np.mean([errors[i]['r_rel'] for i in range(len(errors))])
            t_rmse = np.mean([errors[i]['t_rmse'] for i in range(len(errors))])
            r_rmse = np.mean([errors[i]['r_rmse'] for i in range(len(errors))])
            if t_rel < best:
                best = t_rel
                torch.save(model.state_dict(), f'{checkpoints_dir}/best_{best:.2f}.pth')
                # wandb.save('best.pth')
            message = f'Epoch {ep + 1} evaluation finished, t_rel: {t_rel:.4f}, r_rel: {r_rel:.4f}, t_rmse: {t_rmse:.4f}, r_rmse: {r_rmse:.4f}, best t_rel: {best:.4f}'
            logger.info(message)
            print(message)

        # 保存模型检查点
        if (ep + 1) % 10 == 0:
            torch.save(model.state_dict(), f'{checkpoints_dir}/{ep + 1:003}.pth')
            message = f'Checkpoint saved for epoch {ep + 1}, pose loss: {avg_pose_loss:.6f}, model saved'
            print(message)
            logger.info(message)

    torch.save(model.state_dict(), f'{checkpoints_dir}/final_model.pth')
    message = f'Training finished, best t_rel: {best:.4f}'
    logger.info(message)
    print(message)
    plot_losses(all_epochs, train_losses, file_dir)


if __name__ == "__main__":
    main()
