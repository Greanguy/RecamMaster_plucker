import argparse
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import json
import glob
import os

class CameraPoseVisualizer:
    def __init__(self, xlim, ylim, zlim):
        self.fig = plt.figure(figsize=(18, 7))
        self.ax = self.fig.add_subplot(projection='3d')
        self.plotly_data = None
        self.ax.set_aspect("auto")
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        self.ax.set_zlim(zlim)
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.ax.set_zlabel('z')
        print('initialize camera pose visualizer')

    def extrinsic2pyramid(self, extrinsic, color_map='red', hw_ratio=9/16, base_xval=1, zval=3):
        vertex_std = np.array([[0, 0, 0, 1],
                               [base_xval, -base_xval * hw_ratio, zval, 1],
                               [base_xval, base_xval * hw_ratio, zval, 1],
                               [-base_xval, base_xval * hw_ratio, zval, 1],
                               [-base_xval, -base_xval * hw_ratio, zval, 1]])
        vertex_transformed = vertex_std @ extrinsic.T
        meshes = [[vertex_transformed[0, :-1], vertex_transformed[1][:-1], vertex_transformed[2, :-1]],
                            [vertex_transformed[0, :-1], vertex_transformed[2, :-1], vertex_transformed[3, :-1]],
                            [vertex_transformed[0, :-1], vertex_transformed[3, :-1], vertex_transformed[4, :-1]],
                            [vertex_transformed[0, :-1], vertex_transformed[4, :-1], vertex_transformed[1, :-1]],
                            [vertex_transformed[1, :-1], vertex_transformed[2, :-1], vertex_transformed[3, :-1], vertex_transformed[4, :-1]]]

        color = color_map if isinstance(color_map, str) else plt.cm.rainbow(color_map)

        self.ax.add_collection3d(
            Poly3DCollection(meshes, facecolors=color, linewidths=0.3, edgecolors=color, alpha=0.35))

    def customize_legend(self, list_label):
        list_handle = []
        for idx, label in enumerate(list_label):
            color = plt.cm.viridis(idx / len(list_label))
            patch = Patch(color=color, label=label)
            list_handle.append(patch)
        plt.legend(loc='right', bbox_to_anchor=(1.8, 0.5), handles=list_handle)

    def colorbar(self, max_frame_length):
        cmap = mpl.cm.rainbow
        norm = mpl.colors.Normalize(vmin=0, vmax=max_frame_length)
        self.fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=self.ax, orientation='vertical', label='Frame Number')

    def show(self):
        plt.title('Extrinsic Parameters')
        plt.savefig('extrinsic_parameters_blender.jpg', format='jpg', dpi=300)
        plt.show()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pose_file_path', default='./example_test_data/cameras/camera_extrinsics.json', type=str, help='the path of the pose file')
    parser.add_argument('--hw_ratio', default=9/16, type=float, help='the height over width of the film plane')
    parser.add_argument('--total_frame', type=int, default=81)
    parser.add_argument('--stride', type=int, default=4)
    parser.add_argument('--cam_idx', type=str, default="05")
    parser.add_argument('--base_xval', type=float, default=0.08)
    parser.add_argument('--zval', type=float, default=0.15)
    parser.add_argument('--x_min', type=float, default=-2)
    parser.add_argument('--x_max', type=float, default=2)
    parser.add_argument('--y_min', type=float, default=-2)
    parser.add_argument('--y_max', type=float, default=2)
    parser.add_argument('--z_min', type=float, default=-1.)
    parser.add_argument('--z_max', type=float, default=1)
    return parser.parse_args()

def get_c2w(w2cs, transform_matrix, relative_c2w=True):
    if relative_c2w:
        target_cam_c2w = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        abs2rel = target_cam_c2w @ w2cs[0]
        ret_poses = [target_cam_c2w, ] + [abs2rel @ np.linalg.inv(w2c) for w2c in w2cs[1:]]
    else:
        ret_poses = [np.linalg.inv(w2c) for w2c in w2cs]
    ret_poses = [transform_matrix @ x for x in ret_poses]
    return np.array(ret_poses, dtype=np.float32)

def parse_matrix(matrix_str):
    rows = matrix_str.strip().split('] [')
    matrix = []
    for row in rows:
        row = row.replace('[', '').replace(']', '')
        if len((list(map(float, row.split())))) == 3:
            matrix.append((list(map(float, row.split()))) +[0.])
        else:
            matrix.append(list(map(float, row.split())))
    return np.array(matrix)


# 替换原来的数据加载部分
def load_camera_data(args):
    # 获取所有npy文件并按名称排序
    npy_path = '/data1/blender_dataset/cameras_cam09/'
    npy_files = sorted(glob.glob(os.path.join(npy_path, '*.npy'))) 
    npy_files = [f for f in npy_files if os.path.basename(f) != 'camera_intrinsics.npy']
    npy_files = npy_files[::args.stride]
    
    cameras = []
    for i in range(0, len(npy_files)):
        if i < len(npy_files):
            data = np.load(npy_files[i])
            cameras.append(data)
    
    return cameras

if __name__ == '__main__':
    args = get_args()

    came = load_camera_data(args)
    came = np.stack(came)
    cameras = []
    for cam in came:
        if cam.shape[0] == 3:
            cam = np.vstack((cam, np.array([[0, 0, 0, 1]])))
            cameras.append(cam)
    cameras = np.stack(cameras)

    w2cs = []
    for cam in cameras:
        if cam.shape[0] == 3:
            cam = np.vstack((cam, np.array([[0, 0, 0, 1]])))
        cam = cam[:, [1, 2, 0, 3]]
        cam[:3, 3] *= -1.
        w2cs.append(cam)
    transform_matrix = np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
    c2ws = get_c2w(w2cs, transform_matrix, True)
    scale = max(max(abs(c2w[:3, 3])) for c2w in c2ws)
    if scale > 1e-3:  # otherwise, pan or tilt
        for c2w in c2ws:
            c2w[:3, 3] /= scale

    visualizer = CameraPoseVisualizer([args.x_min, args.x_max], [args.y_min, args.y_max], [args.z_min, args.z_max])
    for frame_idx, c2w in enumerate(c2ws):
        visualizer.extrinsic2pyramid(c2w, frame_idx / len(cameras), hw_ratio=args.hw_ratio, base_xval=args.base_xval,
                                     zval=(args.zval))
    visualizer.colorbar(len(cameras))
    visualizer.show()
