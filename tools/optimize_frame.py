import os
import sys
import torch
# os.system('CUDA_LAUNCH_BLOCKING=1')

import pandas as pd
from pytorch3d.ops.knn import knn_points
from tqdm import tqdm
from sklearn.cluster import DBSCAN


from data.dataloader import NSF_dataset, SFDataset4D
from loss.flow import GeneralLoss, SC2_KNN
from models.NP import NeuralPriorNetwork, PoseNeuralPrior
from ops.transform import find_weighted_rigid_alignment
from vis.deprecated_vis import *

cfg = {     'lr': 0.001,
            'K': 8,
            'beta': 0.95,
            'dynamic_threshold' : 0.15,
            'max_radius': 2,
            'max_iters': 100,
            'eps' : 0.8,
            'min_samples' : 5,
            'max_radius' : 70,
        }

if __name__ == '__main__':

    if len(sys.argv) > 1:
        file = sys.argv[1]
    else:
        file = '/home.dokt/vacekpa2/data/valeo_filip/1000.npz'
    

    device = torch.device('cuda:0')

    first = 0
    max_radius = cfg['max_radius']
    
    # Load data
    data = np.load(file, allow_pickle=True)

    pc1 = data['pc1']
    pc2 = data['pc2']

    pc1_mask = np.linalg.norm(pc1, axis=-1) < max_radius
    pc1 = pc1[pc1_mask]
    pc2 = pc2[np.linalg.norm(pc2, axis=-1) < max_radius]
    pc1 = pc1[pc1[:, 2] > 0.3]  # remove ground
    pc2 = pc2[pc2[:, 2] > 0.3]  # remove ground

    pc1 = torch.from_numpy(pc1[None,:]).to(device)
    pc2 = torch.from_numpy(pc2[None,:]).to(device)

    # GT pose, not used
    # pose0 = np.load(file, allow_pickle=True)['pose']
    # pose1 = data['pose']    


    model = PoseNeuralPrior(pc1, pc2, init_transform=0, use_transform=1).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'])

    LossModule = GeneralLoss(pc1=pc1, pc2=pc2, dist_mode='knn_points', K=cfg['K'], max_radius=cfg['max_radius'],
                            smooth_weight=0, forward_weight=0, sm_normals_K=0, pc2_smooth=True)

    st = time.time()

    SC2_Loss = SC2_KNN(pc1=pc1, K=cfg['K'], use_normals=False)

    for flow_e in tqdm(range(cfg['max_iters'])):
        
        pc1 = pc1.contiguous()

        pred_flow = model(pc1)

        loss = LossModule(pc1, pred_flow, pc2)
        
        loss += cfg['beta'] * SC2_Loss(pred_flow)


        loss.mean().backward()

        optimizer.step()
        optimizer.zero_grad()
        
    eval_time = time.time() - st



    ### Pose estimation
    weights = torch.ones((1,pc1.shape[1])).to(device)
    trans = find_weighted_rigid_alignment(pc1, pc1 + pred_flow, weights)[0]
    kabsch_shifted_pc1 = (pc1[0] @ trans[:3,:3].T + trans[:3,3]).unsqueeze(0)


    ### Dynamic motion after ego-motion compensation
    compensated_flow = kabsch_shifted_pc1 - pc1
    object_flow = pred_flow - compensated_flow
    dynamic_mask = object_flow.norm(dim=-1) > cfg['dynamic_threshold']


    ### Clustering dynamic points to IDS based on geometry AND motion
    clusters = DBSCAN(eps=cfg['eps'], min_samples=cfg['min_samples']).fit_predict((pc1+pred_flow)[0].detach().cpu().numpy()) + 1
    clusters[dynamic_mask[0].cpu() == 0] = 0

    # order instances as integers
    for i, idx in enumerate(np.unique(clusters)):
        if idx == 0: continue
        clusters[clusters == idx] = i    
    
    print(' Eval time: ', f"{eval_time:.3f}", '\n',
          'Clusters: ', clusters.shape, '\n',
          'Dynamic points: ', dynamic_mask[0].shape, '\n',
          'Flow: ' , pred_flow[0].shape, '\n',
          'Pose: ' , trans.shape, '\n',
         )