import os
import glob
import numpy as np
import scipy.io as sio
import torch
from PIL import Image
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from vio_utils.utils import *
from tqdm import tqdm


class data_partition():
    def __init__(self, opt, folder):
        super(data_partition, self).__init__()
        self.opt = opt
        self.data_dir = opt.data_dir
        self.seq_len = self.opt.seq_len
        self.folder = folder
        self.load_data()

    def load_data(self):
        image_dir = self.data_dir + '/sequences/'
        imu_dir = self.data_dir + '/imus/'
        pose_dir = self.data_dir + '/poses/'

        self.img_paths = glob.glob('{}{}/image_2/*.png'.format(image_dir, self.folder))
        self.imus = sio.loadmat('{}{}.mat'.format(imu_dir, self.folder))['imu_data_interp']
        self.poses, self.poses_rel = read_pose_from_text('{}{}.txt'.format(pose_dir, self.folder))
        self.img_paths.sort()

        self.img_paths_list, self.poses_list, self.imus_list = [], [], []
        start = 0
        n_frames = len(self.img_paths)
        while start + self.seq_len < n_frames:
            self.img_paths_list.append(self.img_paths[start:start + self.seq_len])
            self.poses_list.append(self.poses_rel[start:start + self.seq_len - 1])
            self.imus_list.append(self.imus[start * 10:(start + self.seq_len - 1) * 10 + 1])
            start += self.seq_len - 1
        self.img_paths_list.append(self.img_paths[start:])
        self.poses_list.append(self.poses_rel[start:])
        self.imus_list.append(self.imus[start * 10:])

    def __len__(self):
        return len(self.img_paths_list)

    def __getitem__(self, i):
        image_path_sequence = self.img_paths_list[i]
        image_sequence = []
        for img_path in image_path_sequence:
            img_as_img = Image.open(img_path)
            img_as_img = TF.resize(img_as_img, size=(self.opt.img_h, self.opt.img_w))
            img_as_tensor = TF.to_tensor(img_as_img) - 0.5
            img_as_tensor = img_as_tensor.unsqueeze(0)
            image_sequence.append(img_as_tensor)
        image_sequence = torch.cat(image_sequence, 0)
        imu_sequence = torch.FloatTensor(self.imus_list[i])
        gt_sequence = self.poses_list[i][:, :6]
        return image_sequence, imu_sequence, gt_sequence


class KITTI_tester():
    def __init__(self, args):
        super(KITTI_tester, self).__init__()

        self.dataloader = []
        for seq in args.val_seq:
            self.dataloader.append(data_partition(args, seq))

        self.args = args

    def test_one_path(self, net, df, num_gpu=1):
        pose_list = []
        for i, (image_seq, imu_seq, gt_seq) in tqdm(enumerate(df), total=len(df), smoothing=0.9):
            x_in = image_seq.unsqueeze(0).repeat(num_gpu, 1, 1, 1, 1).cuda()
            i_in = imu_seq.unsqueeze(0).repeat(num_gpu, 1, 1).cuda()
            with torch.no_grad():
                pose = net(x_in, i_in)
            pose_list.append(pose[0, :, :].detach().cpu().to(torch.float32).numpy())
        pose_est = np.vstack(pose_list)
        return pose_est

    def eval(self, net, num_gpu=1):
        self.errors = []
        self.est = []
        for i, seq in enumerate(self.args.val_seq):
            print(f'testing sequence {seq}')
            pose_est = self.test_one_path(net, self.dataloader[i], num_gpu=num_gpu)
            pose_est_global, pose_gt_global, t_rel, r_rel, t_rmse, r_rmse, speed = kitti_eval(pose_est, self.dataloader[
                i].poses_rel)

            self.est.append({'pose_est_global': pose_est_global, 'pose_gt_global': pose_gt_global, 'speed': speed})
            self.errors.append({'t_rel': t_rel, 'r_rel': r_rel, 't_rmse': t_rmse, 'r_rmse': r_rmse})

        return self.errors

    def generate_plots(self, save_dir):
        for i, seq in enumerate(self.args.val_seq):
            plotPath_2D(seq,
                        self.est[i]['pose_gt_global'],
                        self.est[i]['pose_est_global'],
                        save_dir)

    def save_text(self, save_dir):
        for i, seq in enumerate(self.args.val_seq):
            path = save_dir / '{}_pred.txt'.format(seq)
            saveSequence(self.est[i]['pose_est_global'], path)
            print('Seq {} saved'.format(seq))

    def animate(self, seq, save_dir):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), gridspec_kw={'height_ratios': [1, 1.5]})
        fig.subplots_adjust(hspace=0.1, top=0.95)

        pose_gt_global = self.est[seq]['pose_gt_global']
        pose_est_global = self.est[seq]['pose_est_global']

        x_gt = [pose[0, 3] for pose in pose_gt_global]
        z_gt = [pose[2, 3] for pose in pose_gt_global]
        x_pred = [pose[0, 3] for pose in pose_est_global]
        z_pred = [pose[2, 3] for pose in pose_est_global]
        x_min = min(x_gt + x_pred) - 10
        x_max = max(x_gt + x_pred) + 10
        z_min = min(z_gt + z_pred) - 10
        z_max = max(z_gt + z_pred) + 10

        def update(num):
            ax1.clear()
            img_path = self.dataloader[seq].img_paths[num]
            img = Image.open(img_path)
            ax1.imshow(img)
            ax1.set_title(f"{self.args.val_seq[seq]} Sequence")
            ax1.axis('off')

            ax2.clear()
            x_gt = [pose[0, 3] for pose in pose_gt_global[:num + 1]]
            z_gt = [pose[2, 3] for pose in pose_gt_global[:num + 1]]
            x_pred = [pose[0, 3] for pose in pose_est_global[:num + 1]]
            z_pred = [pose[2, 3] for pose in pose_est_global[:num + 1]]

            ax2.plot(x_gt, z_gt, 'r-', label="Ground Truth")
            ax2.plot(x_pred, z_pred, 'b-', label="Prediction")
            ax2.legend()
            ax2.set_xlabel('x (m)')
            ax2.set_ylabel('z (m)')
            ax2.set_title('2D Trajectory')
            ax2.set_xlim([x_min, x_max])
            ax2.set_ylim([z_min, z_max])

        ani = animation.FuncAnimation(fig, update, frames=len(pose_gt_global), interval=100 // 3, repeat=False)
        ani.save(os.path.join(save_dir, f'{self.args.val_seq[seq]}.mp4'), writer='ffmpeg', fps=30)
        plt.close()

        # Save the trajectory plot as an image
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(x_gt, z_gt, 'r-', label="Ground Truth")
        ax.plot(x_pred, z_pred, 'b-', label="Prediction")
        ax.legend()
        ax.set_xlabel('x (m)')
        ax.set_ylabel('z (m)')
        ax.set_title(f'{self.args.val_seq[seq]} 2D Trajectory')
        ax.set_xlim([x_min, x_max])
        ax.set_ylim([z_min, z_max])
        plt.savefig(os.path.join(save_dir, f'{self.args.val_seq[seq]}_path_2d.png'), bbox_inches='tight',
                    pad_inches=0.1)
        plt.close()


def kitti_eval(pose_est, pose_gt):
    t_rmse, r_rmse = rmse_err_cal(pose_est, pose_gt)
    pose_est_mat = path_accu(pose_est)
    pose_gt_mat = path_accu(pose_gt)
    err_list, t_rel, r_rel, speed = kitti_err_cal(pose_est_mat, pose_gt_mat)

    t_rel = t_rel * 100
    r_rel = r_rel / np.pi * 180 * 100
    r_rmse = r_rmse / np.pi * 180

    return pose_est_mat, pose_gt_mat, t_rel, r_rel, t_rmse, r_rmse, speed


def kitti_err_cal(pose_est_mat, pose_gt_mat):
    lengths = [100, 200, 300, 400, 500, 600, 700, 800]
    num_lengths = len(lengths)

    err = []
    dist, speed = trajectoryDistances(pose_gt_mat)
    step_size = 10  # 10Hz

    for first_frame in range(0, len(pose_gt_mat), step_size):

        for i in range(num_lengths):
            len_ = lengths[i]
            last_frame = lastFrameFromSegmentLength(dist, first_frame, len_)
            if last_frame == -1 or last_frame >= len(pose_est_mat) or first_frame >= len(pose_est_mat):
                continue

            pose_delta_gt = np.dot(np.linalg.inv(pose_gt_mat[first_frame]), pose_gt_mat[last_frame])
            pose_delta_result = np.dot(np.linalg.inv(pose_est_mat[first_frame]), pose_est_mat[last_frame])

            r_err = rotationError(pose_delta_result, pose_delta_gt)
            t_err = translationError(pose_delta_result, pose_delta_gt)

            err.append([first_frame, r_err / len_, t_err / len_, len_])

    t_rel, r_rel = computeOverallErr(err)
    return err, t_rel, r_rel, np.asarray(speed)


def plotPath_2D(seq, poses_gt_mat, poses_est_mat, plot_path_dir):
    fontsize_ = 10
    plot_keys = ["Ground Truth", "Ours"]
    start_point = [0, 0]
    style_pred = 'b-'
    style_gt = 'r-'
    style_O = 'ko'

    x_gt = np.asarray([pose[0, 3] for pose in poses_gt_mat])
    y_gt = np.asarray([pose[1, 3] for pose in poses_gt_mat])
    z_gt = np.asarray([pose[2, 3] for pose in poses_gt_mat])

    x_pred = np.asarray([pose[0, 3] for pose in poses_est_mat])
    y_pred = np.asarray([pose[1, 3] for pose in poses_est_mat])
    z_pred = np.asarray([pose[2, 3] for pose in poses_est_mat])

    fig = plt.figure(figsize=(6, 6), dpi=100)
    ax = plt.gca()
    plt.plot(x_gt, z_gt, style_gt, label=plot_keys[0])
    plt.plot(x_pred, z_pred, style_pred, label=plot_keys[1])
    plt.plot(start_point[0], start_point[1], style_O, label='Start Point')
    plt.legend(loc="upper right", prop={'size': fontsize_})
    plt.xlabel('x (m)', fontsize=fontsize_)
    plt.ylabel('z (m)', fontsize=fontsize_)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xmean = np.mean(xlim)
    ymean = np.mean(ylim)
    plot_radius = max([abs(lim - mean_) for lims, mean_ in ((xlim, xmean), (ylim, ymean)) for lim in lims])
    ax.set_xlim([xmean - plot_radius, xmean + plot_radius])
    ax.set_ylim([ymean - plot_radius, ymean + plot_radius])

    plt.title('2D path')
    png_title = "{}_path_2d".format(seq)
    plt.savefig(plot_path_dir + "/" + png_title + ".png", bbox_inches='tight', pad_inches=0.1)