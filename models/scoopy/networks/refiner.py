import os
import sys
import numpy as np
import time
import torch

# add path
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_dir not in sys.path:
    sys.path.append(project_dir)

from models.scoopy.networks.graph import Graph
from models.scoopy.utils.losses import chamfer_loss, smooth_loss, pcflow_smooth_loss
# from ops.visibility import mask_KNN_by_visibility

class Refiner(torch.nn.Module):
    def __init__(self, shape, device):
        """
        Construct a model for refining scene flow between two point clouds.

        Parameters
        ----------
        shape:
            Shape of the refinement tensor.
        device:
            Device of the refinement tensor.
        """

        super(Refiner, self).__init__()

        self.refinement = torch.nn.Parameter(torch.zeros(shape, dtype=torch.float32, device=device, requires_grad=True))


    def forward(self, flow):
        refined_flow = flow + self.refinement

        return refined_flow

    def refine_flow(self, batch, flow, corr_conf, optimizer, args):
        pc_0, pc_1 = batch["sequence"][0], batch["sequence"][1]
        gt_flow = batch["ground_truth"][1]

        n0 = int(batch["orig_size"][0].cpu().numpy())
        n1 = int(batch["orig_size"][1].cpu().numpy())

        b, nb_points0, c = pc_0.shape
        b, nb_points1, c = pc_1.shape

        pc_0_orig = torch.unsqueeze(torch.reshape(pc_0, (b * nb_points0, 3))[:n0], dim=0)
        pc_1_orig = torch.unsqueeze(torch.reshape(pc_1, (b * nb_points1, 3))[:n1], dim=0)
        gt_flow_orig = torch.unsqueeze(torch.reshape(gt_flow, (b * nb_points0, 3))[:n0], dim=0)

        start_time = time.time()

        graph = Graph.construct_graph_in_chunks(pc_0_orig, 32, 2048)

        # results aggregation
        target_recon_loss_all = np.zeros(args.test_time_num_step + 1, dtype=np.float32)
        smooth_flow_loss_all = np.zeros(args.test_time_num_step + 1, dtype=np.float32)
        epe_all = np.zeros(args.test_time_num_step + 1, dtype=np.float32)

        # load NN mask
        # personilize functions?
        # mask to smooth loss

        cfg = {"x_min" : -50,
           "x_max" : 50,
           "y_min" : -50,
           "y_max" : 50,
           "z_min" : -100,
           "z_max" : 100,
           "cell_size" : (0.05,0.05,0.05),
           "size_of_block" : 0,
           }

        if args.use_visibility_smooth_loss:
            # breakpoint()
            masked_KNN = mask_KNN_by_visibility(pc_0_orig[0].detach().cpu().numpy(), 32, cfg)
            masked_KNN = torch.from_numpy(masked_KNN).cuda()
            # load from harddisk?

        else:
            masked_KNN = None
            from sklearn.neighbors import NearestNeighbors
            orig_KNN = NearestNeighbors(n_neighbors=32, algorithm='kd_tree').fit(pc_0_orig[0].detach().cpu().numpy()).kneighbors(pc_0_orig[0].detach().cpu().numpy(),
                                                                                            return_distance=True)[1]
            orig_KNN = torch.from_numpy(orig_KNN).cuda()



        for step in range(args.test_time_num_step):
            refined_flow = self(flow)
            target_pc_recon = pc_0_orig + refined_flow

            target_recon_loss = chamfer_loss(target_pc_recon, pc_1_orig, corr_conf,
                                             backward_dist_weight=args.backward_dist_weight, mask=None, use_chamfer_cuda=bool(args.use_chamfer_cuda))
            loss = args.target_recon_loss_weight * target_recon_loss

            if args.use_smooth_flow and args.smooth_flow_loss_weight > 0:
                # frame: 162 EPE: 0.0192
                # test_frames: EPE: 0.0296

                if args.use_visibility_smooth_loss:
                    smooth_flow_loss, _ = pcflow_smooth_loss(refined_flow, masked_KNN, loss_norm=1)

                else:
                    smooth_flow_loss, _ = pcflow_smooth_loss(refined_flow, orig_KNN, loss_norm=1)
                    # smooth_flow_loss, _ = smooth_loss(refined_flow, graph, args.nb_neigh_smooth_flow, loss_norm=1, mask=None)

                loss = loss + (args.smooth_flow_loss_weight * smooth_flow_loss)
            else:
                smooth_flow_loss = 0

            # Gradient step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Loss evolution
            loss_curr = loss.item()
            target_recon_loss_curr = target_recon_loss.item()
            smooth_flow_loss_curr = smooth_flow_loss.item() if args.use_smooth_flow and args.smooth_flow_loss_weight > 0 else smooth_flow_loss

            # EPE
            error = refined_flow - gt_flow_orig
            epe_per_point = torch.sqrt(torch.sum(torch.pow(error, 2.0), -1))
            epe = epe_per_point.mean()
            epe_curr = epe.item()

            if args.test_time_verbose:
                print("Refinement step %04d/%04d: loss: %.6f, target_recon_loss: %.6f, smooth_flow_loss: %.6f, epe: %.3f" %
                      ((step + 1), args.test_time_num_step, loss_curr, target_recon_loss_curr, smooth_flow_loss_curr, epe_curr))

            # aggregate results
            target_recon_loss_all[step] = target_recon_loss_curr
            smooth_flow_loss_all[step] = smooth_flow_loss_curr
            epe_all[step] = epe_curr

        refined_flow = self(flow)

        duration = time.time() - start_time

        # EPE last
        error = refined_flow - gt_flow_orig
        epe_per_point = torch.sqrt(torch.sum(torch.pow(error, 2.0), -1))
        epe = epe_per_point.mean()
        epe_curr = epe.item()

        epe_all[-1] = epe_curr

        refine_metrics = {"target_recon_loss_all": target_recon_loss_all, "smooth_flow_loss_all": smooth_flow_loss_all, "epe_all": epe_all}

        return refined_flow, refine_metrics, duration
