import argparse
import os
import torch
from path import Path
from vio_utils import custom_transform
from dataset_process.KITTI_dataset import KITTI
from model import DeepVIO
from vio_utils.kitti_eval import KITTI_tester
import numpy as np

# 设置参数
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data_dir', type=str, default='/mnt/data2/Mutil_data/Visual-Selective-VIO-main/data', help='path to the dataset')
parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
parser.add_argument('--save_dir', type=str, default='./results', help='path to save the result')
parser.add_argument('--seq_len', type=int, default=16, help='sequence length for LSTM')
parser.add_argument("--grad_cp", default=0, type=int)  # gradient checkpt: saves VRAM, but slower

parser.add_argument('--train_seq', type=list, default=['00', '01', '02', '04', '06', '08', '09'], help='sequences for training')
parser.add_argument('--val_seq', type=list, default=['04','05', '07', '10'], help='sequences for validation')
parser.add_argument('--seed', type=int, default=0, help='random seed')

parser.add_argument('--img_w', type=int, default=512, help='image width')
parser.add_argument('--img_h', type=int, default=256, help='image height')
parser.add_argument('--v_f_len', type=int, default=512, help='visual feature length')
parser.add_argument('--i_f_len', type=int, default=256, help='imu feature length')
parser.add_argument('--fuse_method', type=str, default='cat', help='fusion method [cat, soft, hard]')
parser.add_argument('--imu_dropout', type=float, default=0, help='dropout for the IMU encoder')
parser.add_argument('--head_size_a', type=int, default=64, help='')
parser.add_argument("--head_size_divisor", default=12, type=int)
parser.add_argument("--dropout", default=0.1, type=float)

parser.add_argument('--rwkv_out_size', type=int, default=1024, help='size of the LSTM latent')
parser.add_argument('--rwkv_dropout_out', type=float, default=0.01, help='dropout for the rwkv output layer')
parser.add_argument('--rnn_dropout_between', type=float, default=0.2, help='dropout within LSTM')
parser.add_argument('--n_layer', type=int, default=4, help='num of rwkv')
parser.add_argument('--n_embd', type=int, default=768, help='v_f + i_f')
parser.add_argument('--dim_att', type=int, default=768, help='')
parser.add_argument('--dim_ffn', type=int, default=2048, help='')
parser.add_argument('--workers', type=int, default=32, help='number of workers')
parser.add_argument('--experiment_name', type=str, default='test', help='experiment name')
parser.add_argument('--model', type=str, default='/mnt/data2/Mutil_data/VIO/results/experiment/checkpoints/best_19.94.pth', help='path to the trained model')

args = parser.parse_args()

# 设置随机种子
torch.manual_seed(args.seed)
np.random.seed(args.seed)

def main():
    # 创建保存目录
    experiment_dir = Path('./results')
    experiment_dir.mkdir_p()
    file_dir = experiment_dir.joinpath('{}/'.format(args.experiment_name))
    file_dir.mkdir_p()
    result_dir = file_dir.joinpath('files/')
    result_dir.mkdir_p()

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

    # 加载已训练模型
    model.load_state_dict(torch.load(args.model))
    print('Loaded model %s' % args.model)

    # 模型加载到 GPU
    model.cuda(gpu_ids[0])
    model = torch.nn.DataParallel(model, device_ids=gpu_ids)
    model.eval()

    # 开始评估
    errors = tester.eval(model, num_gpu=len(gpu_ids))
    tester.save_text(result_dir)

    for i, seq in enumerate(args.val_seq):
        message = f"Seq: {seq}, t_rel: {tester.errors[i]['t_rel']:.4f}, r_rel: {tester.errors[i]['r_rel']:.4f}, "
        message += f"t_rmse: {tester.errors[i]['t_rmse']:.4f}, r_rmse: {tester.errors[i]['r_rmse']:.4f}"
        print(message)

        tester.animate(i, result_dir)
        tester.generate_plots(result_dir)


if __name__ == "__main__":
    main()
