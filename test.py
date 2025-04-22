import argparse
import os
import torch
from path import Path
from vio_utils import custom_transform
from dataset_process.KITTI_dataset import KITTI
from model2 import DeepVIO
# from model import DeepVIO as DeepVIO2
from thop import profile, clever_format
from torch import nn
import time
from vio_utils.kitti_eval import KITTI_tester
import numpy as np

# 设置参数
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data_dir', type=str, default='/mnt/data2/Mutil_data/Visual-Selective-VIO-main/data', help='path to the dataset')
parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
parser.add_argument('--save_dir', type=str, default='./results', help='path to save the result')
parser.add_argument('--seq_len', type=int, default=11, help='sequence length for LSTM')
parser.add_argument("--grad_cp", default=0, type=int)  # gradient checkpt: saves VRAM, but slower

parser.add_argument('--train_seq', type=list, default=['00', '01', '02', '04', '06', '08', '09'], help='sequences for training')
parser.add_argument('--val_seq', type=list, default=['00', '01', '02', '04', '06', '08', '09'], help='sequences for validation')
parser.add_argument('--seed', type=int, default=0, help='random seed')

parser.add_argument('--img_w', type=int, default=512, help='image width')
parser.add_argument('--img_h', type=int, default=256, help='image height')
parser.add_argument('--v_f_len', type=int, default=512, help='visual feature length')
parser.add_argument('--i_f_len', type=int, default=256, help='imu feature length')
parser.add_argument('--fuse_method', type=str, default='cat', help='fusion method [cat, soft, hard]')
parser.add_argument('--imu_dropout', type=float, default=0, help='dropout for the IMU encoder')
parser.add_argument('--head_size_a', type=int, default=96, help='')
parser.add_argument("--head_size_divisor", default=8, type=int)
parser.add_argument("--dropout", default=0.1, type=float)

parser.add_argument('--rwkv_out_size', type=int, default=512, help='size of the LSTM latent')
parser.add_argument('--rwkv_dropout_out', type=float, default=0.06, help='dropout for the rwkv output layer')
parser.add_argument('--rnn_dropout_between', type=float, default=0.02, help='dropout within LSTM')
parser.add_argument('--rnn_hidden_size', type=int, default=768, help='size of the LSTM latent')
parser.add_argument('--rnn_dropout_out', type=float, default=0, help='dropout for the LSTM output layer')
parser.add_argument('--n_heads', type=int, default=8, help='num of attention heads')
parser.add_argument('--block_size', type=int, default=16, help='block size for RWKV')

parser.add_argument('--n_layer', type=int, default=1, help='num of rwkv')
parser.add_argument('--n_embd', type=int, default=512, help='v_f + i_f')
parser.add_argument('--dim_att', type=int, default=768, help='')
parser.add_argument('--dim_ffn', type=int, default=2048, help='')
parser.add_argument('--workers', type=int, default=32, help='number of workers')
parser.add_argument('--experiment_name', type=str, default='experiment_12_02', help='experiment name')
parser.add_argument('--RNN_RWKV', type=str, default='RWKV', help='choose RNN or RWKV')

parser.add_argument('--model', type=str, default='/mnt/data2/Mutil_data/VIO/results/experiment_12_02/checkpoints/best_2.30.pth', help='path to the trained model')

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
    if args.RNN_RWKV == 'RWKV':
        model = DeepVIO(args)
    elif args.RNN_RWKV == 'RNN':
        model = DeepVIO2(args)
    # 加载已训练模型
    model.load_state_dict(torch.load(args.model))
    print('Loaded model %s' % args.model)

    # 模型加载到 GPU
    model.cuda(gpu_ids[0])
    # model = torch.nn.DataParallel(model, device_ids=gpu_ids)
    model.eval()
    img_input = torch.randn(1, 11, 3, args.img_w, args.img_h).cuda()  # 假设视觉输入为 (3, img_h, img_w)
    imu_input = torch.randn(1, 101, 6).cuda()  # 假设 IMU 输入为 (seq_len, i_f_len)
    # -------------------------------
    # 统计整个模型的 MACs
    # -------------------------------
    inputs = (img_input, imu_input)  # 将图像和 IMU 数据作为输入
    macs, params = profile(model, inputs=inputs, verbose=False)
    # 格式化输出
    # macs, params = clever_format([macs, params], "%.3f")
    print(f"Total MACs: {macs/1e9} G")
    print(f"Total Params: {params/1e6} M")
    print("**************")

    from test2 import Encoder
    encoder = Encoder().cuda()
    img_input2 = torch.randn(1, 11, 3, 256, 512).cuda()
    # 计算 MACs 和参数量
    macs2, params2 = profile(encoder, inputs=(img_input2,), verbose=False)
    # 格式化输出
    # macs, params = clever_format([macs, params], "%.3f")
    print(f"Encoder MACs: {macs2 / 1e9} G")
    print(f"Encoder Params: {params2 / 1e6} M")
    print("**************")
    print(f"exclude visual MACs: {(macs-macs2)/1e6} M")
    print(f"exclude visual Params: {(params-params2)/1e6} M")
    with torch.no_grad():
        # GPU 时间同步
        torch.cuda.synchronize()
        start_time = time.time()

        # 执行前向传播
        for _ in range(100):  # 执行多次推理以减少波动
            _ = model(*inputs)

        # GPU 时间同步
        torch.cuda.synchronize()
        end_time = time.time()

    # 计算平均推理时间
    total_time = end_time - start_time
    average_time = total_time / 100 # 每次推理的平均时间
    print(f"Total time for 100 iterations: {total_time:.6f} seconds")
    print(f"Average inference time per iteration: {average_time:.6f} seconds")
    fps = 1 / average_time
    print(f"Fps: {fps:.2f} ")



    errors = tester.eval(model, num_gpu=len(gpu_ids))
    tester.save_text(result_dir)

    for i, seq in enumerate(args.val_seq):
        message = f"Seq: {seq}, t_rel: {tester.errors[i]['t_rel']:.4f}, r_rel: {tester.errors[i]['r_rel']:.4f}, "
        message += f"t_rmse: {tester.errors[i]['t_rmse']:.4f}, r_rmse: {tester.errors[i]['r_rmse']:.4f}"
        print(message)

        # tester.animate(i, result_dir)
        tester.generate_plots(result_dir)


if __name__ == "__main__":
    main()
