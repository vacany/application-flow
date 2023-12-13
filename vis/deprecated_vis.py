import os.path
import numpy as np
import torch
import glob
import matplotlib.pyplot as plt
import sys
import socket
import time

from data.PATHS import VIS_PATH

# try:
def imshow(fig=None, path=None):
    try:
        from IPython.display import display, Image

        if fig is not None:
            URL = VIS_PATH + '/tmp.png'
            # if hasattr(fig, 'shape'):
            #     plt.imshow(fig)
            #     plt.savefig(URL)
            #     plt.close()
            # else:
            fig.savefig(URL)

            display(Image(URL, width=2000))
            os.remove(URL)
        if path is not None:
            display(Image(path))
    except:
        print('You do not have IPython installed!')
        pass




class FileWatcher():

    def __init__(self, file_path):
        self.file_path = file_path
        self.last_modified = os.stat(file_path).st_mtime

    def check_modification(self):
        stamp = os.stat(self.file_path).st_mtime
        if stamp > self.last_modified:
            self.last_modified = stamp
            return True



if socket.gethostname().startswith("Pat"):

    sys.path.append('/home/patrik/.local/lib/python3.8/site-packages')
    import pptk

    def visualize_points3D(points, labels=None, point_size=0.02, **kwargs):
        if not socket.gethostname().startswith("Pat"):
            return

        if type(points) is not np.ndarray:
            points = points.detach().cpu().numpy()

        if type(labels) is not np.ndarray and labels is not None:
            labels = labels.detach().cpu().numpy()


        if labels is None:
            v = pptk.viewer(points[:,:3])
        else:
            v = pptk.viewer(points[:, :3], labels)

        v.set(point_size=point_size)
        v.set(**kwargs)

        return v

    def visualize_pcd(file, point_size=0.01):
        import open3d
        points = np.asarray(open3d.io.read_point_cloud(file).points)
        v=pptk.viewer(points[:,:3])
        v.set(point_size=point_size)

    def visualize_voxel(voxel, cell_size=(0.2, 0.2, 0.2)):
        x,y,z = np.nonzero(voxel)
        label = voxel[x,y,z]
        pcl = np.stack([x / cell_size[0], y / cell_size[1], z / cell_size[2]]).T
        visualize_points3D(pcl, label)

    def visualize_poses(poses):
        xyz = poses[:,:3,-1]
        fig, axes = plt.subplots(2, 1)
        axes[0].plot(xyz[:,0], xyz[:,1])
        res = np.abs(poses[:-1, :3, -1] - poses[1:, :3, -1]).sum(1)
        axes[1].plot(res)
        plt.show()

    def visualize_multiple_pcls(*args, **kwargs):
        p = []
        l = []

        for n, points in enumerate(args):
            if type(points) == torch.Tensor:
                p.append(points[:,:3].detach().cpu().numpy())
            else:
                p.append(points[:,:3])
            l.append(n * np.ones((points.shape[0])))

        p = np.concatenate(p)
        l = np.concatenate(l)
        v=visualize_points3D(p, l)
        v.set(**kwargs)


    def visualize_plane_with_points(points, n_vector, d):

        xx, yy = np.meshgrid(np.linspace(points[:,0].min(), points[:,0].max(), 100),
                             np.linspace(points[:,1].min(), points[:,1].max(), 100))

        z = (- n_vector[0] * xx - n_vector[1] * yy - d) * 1. / n_vector[2]
        x = np.concatenate(xx)
        y = np.concatenate(yy)
        z = np.concatenate(z)

        plane_pts = np.stack((x, y, z, np.zeros(z.shape[0]))).T

        d_dash = - n_vector.T @ points[:,:3].T

        bin_points = np.concatenate((points, (d - d_dash)[:, None]), axis=1)

        vis_pts = np.concatenate((bin_points, plane_pts))

        visualize_points3D(vis_pts, vis_pts[:,3])


    def visualize_flow3d(pts1, pts2, frame_flow, vis='pptk'):
        # flow from multiple pcl vis
        # valid_flow = frame_flow[:, 3] == 1
        # vis_flow = frame_flow[valid_flow]
        # threshold for dynamic is flow larger than 0.05 m

        if len(pts1.shape) == 3:
            pts1 = pts1[0]

        if len(pts2.shape) == 3:
            pts2 = pts2[0]

        if len(frame_flow.shape) == 3:
            frame_flow = frame_flow[0]

        if type(pts1) is not np.ndarray:
            pts1 = pts1.detach().cpu().numpy()

        if type(pts2) is not np.ndarray:
            pts2 = pts2.detach().cpu().numpy()

        if type(frame_flow) is not np.ndarray:
            frame_flow = frame_flow.detach().cpu().numpy()

        dist_mask = np.sqrt((frame_flow[:,:3] ** 2).sum(1)) > 0.05

        vis_pts = pts1[dist_mask,:3]
        vis_flow = frame_flow[dist_mask]


        all_rays = []

        for x in range(1, int(20)):
            ray_points = vis_pts + (vis_flow[:, :3]) * (x / int(20))
            all_rays.append(ray_points)

        all_rays = np.concatenate(all_rays)

        visualize_multiple_pcls(*[pts1, all_rays, pts2], point_size=0.02, show_grid=False, lookat=[0,0,0], bg_color_top=[0,0,0,1])

        def visualize_flow_frame(pc1, pc2, est_flow1, pose=None, pose2=None, normals1=None, normals2=None):

            figure = mlab.figure(1, bgcolor=(1, 1, 1), size=(640, 480))
            vis_pc1 = pc1.detach().cpu().numpy()
            vis_pc2 = pc2.detach().cpu().numpy()

            vis_est_rigid_flow = est_flow1.detach().cpu().numpy()

            mlab.points3d(0, 0, 0, color=(0, 0, 1), scale_factor=0.3, mode='axes')
            if pose2 is not None:
                mlab.points3d(pose2.detach()[0, 0, -1], pose2.detach()[0, 1, -1], pose2.detach()[0, 2, -1],
                              color=(1, 0, 0), scale_factor=0.3, mode='axes')
            mlab.points3d(vis_pc1[0, :, 0], vis_pc1[0, :, 1], vis_pc1[0, :, 2], color=(0, 0, 1), scale_factor=0.1)
            mlab.points3d(vis_pc2[0, :, 0], vis_pc2[0, :, 1], vis_pc2[0, :, 2], color=(1, 0, 0), scale_factor=0.1)
            mlab.quiver3d(vis_pc1[0, :, 0], vis_pc1[0, :, 1], vis_pc1[0, :, 2], vis_est_rigid_flow[0, :, 0],
                          vis_est_rigid_flow[0, :, 1], vis_est_rigid_flow[0, :, 2], color=(0, 1, 0), scale_factor=1)

            if normals1 is not None:
                vis_normals1 = normals1.detach().cpu().numpy()
                mlab.quiver3d(vis_pc1[0, :, 0], vis_pc1[0, :, 1], vis_pc1[0, :, 2], vis_normals1[0, :, 0],
                              vis_normals1[0, :, 1], vis_normals1[0, :, 2], color=(0, 0, 0.4), scale_factor=0.4)

            if normals2 is not None:
                vis_normals2 = normals2.detach().cpu().numpy()
                mlab.quiver3d(vis_pc2[0, :, 0], vis_pc2[0, :, 1], vis_pc2[0, :, 2], vis_normals2[0, :, 0],
                              vis_normals2[0, :, 1], vis_normals2[0, :, 2], color=(0.4, 0, 0), scale_factor=0.4)
            mlab.show()

    def visualizer_transform(p_i, p_j, trans_mat):
        '''

        :param p_i: source
        :param p_j: target
        :param trans_mat: p_i ---> p_j transform matrix
        :return:
        '''

        p_i = np.insert(p_i, obj=3, values=1, axis=1)
        vis_p_i = p_i @ trans_mat.T
        visualize_multiple_pcls(*[p_i, vis_p_i, p_j])


else:
    def visualize_points3D(points, labels=None, **kwargs):
        if type(points) is not np.ndarray:
            points = points.detach().cpu().numpy()

        if type(labels) is not np.ndarray and labels is not None:
            labels = labels.detach().cpu().numpy()

        folder_path = VIS_PATH + '/tmp_vis/'
        file = f'{time.time()}_cur.npz'

        command = 'visualize_points3D'
        # breakpoint()
        np.savez(folder_path + '/' + file, points=points, labels=labels, command=command, **kwargs)



    def visualize_flow3d(pts1, pts2, frame_flow):

        if type(pts1) is not np.ndarray:
            pts1 = pts1.detach().cpu().numpy()

        if type(pts2) is not np.ndarray:
            pts2 = pts2.detach().cpu().numpy()

        if type(frame_flow) is not np.ndarray:
            frame_flow = frame_flow.detach().cpu().numpy()

        folder_path = VIS_PATH + '/tmp_vis/'
        file = f'{time.time()}_cur.npz'

        command = 'visualize_flow3d'
        # breakpoint()
        np.savez(folder_path + '/' + file, pts1=pts1, pts2=pts2, frame_flow=frame_flow, command=command)
        # visualize_multiple_pcls(*[pts1, all_rays, pts2], point_size=0.02)


    def visualize_plane_with_points(points, n_vector, d):
        pass
    def visualize_multiple_pcls(*args, **kwargs):

        folder_path = VIS_PATH + '/tmp_vis/'
        file = f'{time.time()}_cur.npz'

        command = 'visualize_multiple_pcls'

        # pcs = [a.detach().cpu().numpy() for a in args if type(a) is not np.ndarray]
        pcs = [a for a in args]

        np.savez(folder_path + '/' + file, args=pcs, command=command, **kwargs)


    def show_image(image, title=None):
        plt.imshow(image)
        plt.savefig('my_plot.png')

def visualize_KNN_connections(pc, KNN_matrix, fill_pts=10):
    from ops.rays import raycast_NN

    r_, ind_ = raycast_NN(pc, KNN_matrix, fill_pts=fill_pts)

    visualize_points3D(r_, ind_[:,1], lookat=[0,0,0])


# one KNN visual in depth image
def visualize_one_KNN_in_depth(KNN_image_indices, depth2, chosen_NN, K, margin=0.05, output_path=None):

    origin_pt = KNN_image_indices[chosen_NN, 0]
    knn_pts = KNN_image_indices[chosen_NN, 1:]

    # connect to origin
    connections = []

    for k in range(0, K - 1):
        px = torch.linspace(origin_pt[0], knn_pts[k, 0], 300, device=origin_pt.device)
        py = torch.linspace(origin_pt[1], knn_pts[k, 1], 300, device=origin_pt.device)

        # linear depth
        lin_d = depth2[origin_pt[0], origin_pt[1]] + (
                    depth2[knn_pts[k, 0], knn_pts[k, 1]] - depth2[origin_pt[0], origin_pt[1]]) * torch.linspace(0, 1,
                                                                                                                300,
                                                                                                                device=origin_pt.device)
        orig_d = depth2[px.to(torch.long), py.to(torch.long)]
        logic_d = lin_d <= orig_d
        px = px.to(torch.long)
        py = py.to(torch.long)
        connections.append(torch.stack([px, py, lin_d, orig_d, logic_d], dim=1))

    fig, ax = plt.subplots(3, 1, figsize=(10, 10), dpi=400)
    # vis_im = torch.zeros(depth2.shape, device=device)
    vis_im = depth2.clone()
    vis_im1 = depth2.clone()
    vis_im2 = depth2.clone()

    for con in connections:
        vis_im[con[:, 0].long(), con[:, 1].long()] = con[:, 2]
        vis_im1[con[:, 0].long(), con[:, 1].long()] = con[:, 3]
        vis_im2[con[:, 0].long(), con[:, 1].long()] = con[:, 4] * 300

        vis_im[origin_pt[0], origin_pt[1]] = 300
        vis_im[knn_pts[:, 0], knn_pts[:, 1]] = 175

        vis_im1[origin_pt[0], origin_pt[1]] = 300
        vis_im1[knn_pts[:, 0], knn_pts[:, 1]] = 175

        vis_im2[origin_pt[0], origin_pt[1]] = 300
        vis_im2[knn_pts[:, 0], knn_pts[:, 1]] = 175



    ax[0].imshow(np.flip(vis_im.detach().cpu().numpy(), axis=0), interpolation='none', aspect='auto')
    ax[0].set_aspect('equal')
    ax[1].imshow(np.flip(vis_im1.detach().cpu().numpy(), axis=0), interpolation='none', aspect='auto', cmap='jet')
    ax[1].set_aspect('equal')
    # for idx in knn_pts.detach().cpu().numpy():
    #     j, i = idx
    #     label = depth2[idx[0], idx[1]]
    #     # ax[1].text(i,j,label.item(),ha='center', va='center', fontsize=2)
    #     # print(i,j,label)

    ax[2].imshow(np.flip(vis_im2.detach().cpu().numpy(), axis=0), interpolation='none', aspect='auto')
    ax[2].set_aspect('equal')

    fig.savefig(output_path)


# matplotlib
def visualize_connected_points(pts1, pts2, title=None, savefig=None):
    plt.plot(pts1[:, 0], pts1[:, 1], 'ob')
    plt.plot(pts2[:, 0], pts2[:, 1], 'or')

    for i in range(len(pts1)):
        p = pts1[i]
        r = pts2[i]

        diff = r - p
        yaw_from_meds = np.arctan2(diff[1], diff[0])
        yaw_degree = 180 * yaw_from_meds / np.pi

        plt.plot(p[0], p[1], '.b')
        plt.plot(r[0], r[1], '.r')

        plt.annotate(f"{yaw_degree:.2f} deg", p[:2] + (0.01, 0))

        connection = np.array([(p[0], p[1]), (r[0], r[1])])
        plt.plot(connection[:, 0], connection[:, 1], 'g--')

    if title is not None:
        plt.title(title)
    plt.axis('equal')
    plt.show()

if __name__ == "__main__":
    # tmp_folder = os.path.expanduser("~") + '/rci/data/tmp_vis/'
    tmp_folder = os.path.expanduser("~") + '/cmp/visuals/tmp_vis/'

    # command = 'visualize_points3D'
    # print(getattr(sys.modules[__name__], command))

    # Clean tmp folder before starting
    files = glob.glob(tmp_folder + '/*.npz') + glob.glob(tmp_folder + '/*.png')

    for file in files:
        os.remove(file)

    print('waiting for files in folder ', tmp_folder, ' ...')


    while True:
        time.sleep(0.8)

        files = glob.glob(tmp_folder + '/*.npz') + glob.glob(tmp_folder + '/*.png')

        _ = os.stat(tmp_folder).st_mtime    # to refresh the cache

        if len(files) > 0:

            for file in files:
                print('Loading file: ', file)


                if file.endswith('.npz'):
                    data = np.load(file, allow_pickle=True)
                    command = data['command'].item()

                    kwargs = {name: data[name] for name in data.files if name not in ['command']}

                    if 'multiple' in command:
                        pcs = [data['args'][i] for i in range(len(data['args']))]
                        # breakpoint()
                        visualize_multiple_pcls(*pcs)
                    else:
                        getattr(sys.modules[__name__], command)(**kwargs)

                elif file.endswith('.png'):
                    from PIL import Image
                    img = Image.open(file)
                    img.show()



                else:
                    raise ValueError('Unknown file type: ', file)

                os.remove(file)



