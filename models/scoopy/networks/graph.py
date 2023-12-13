import os
import sys
import torch
import numpy as np

# add path
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_dir not in sys.path:
    sys.path.append(project_dir)

from models.scoopy.utils.utils import iterate_in_chunks

class Graph:
    def __init__(self, edges, edge_feats, k_neighbors, size):
        """
        Directed nearest neighbor graph constructed on a point cloud.

        Parameters
        ----------
        edges : torch.Tensor
            Contains list with nearest neighbor indices.
        edge_feats : torch.Tensor
            Contains edge features: relative point coordinates.
        k_neighbors : int
            Number of nearest neighbors.
        size : tuple(int, int)
            Number of points.

        """

        self.edges = edges
        self.size = tuple(size)
        self.edge_feats = edge_feats
        self.k_neighbors = k_neighbors

    @staticmethod
    def construct_graph(pcloud, nb_neighbors):
        """
        Construct a directed nearest neighbor graph on the input point cloud.

        Parameters
        ----------
        pcloud : torch.Tensor
            Input point cloud. Size B x N x 3.
        nb_neighbors : int
            Number of nearest neighbors per point.

        Returns
        -------
        graph : scoop.models.graph.Graph
            Graph build on input point cloud containing the list of nearest 
            neighbors (NN) for each point and all edge features (relative 
            coordinates with NN).
            
        """

        # Size
        nb_points = pcloud.shape[1]
        size_batch = pcloud.shape[0]

        # Distance between points
        distance_matrix = torch.sum(pcloud ** 2, -1, keepdim=True)
        distance_matrix = distance_matrix + distance_matrix.transpose(1, 2)
        distance_matrix = distance_matrix - 2 * torch.bmm(
            pcloud, pcloud.transpose(1, 2)
        )

        # Find nearest neighbors
        neighbors = torch.argsort(distance_matrix, -1)[..., :nb_neighbors]
        effective_nb_neighbors = neighbors.shape[-1]
        neighbors = neighbors.reshape(size_batch, -1)

        # Edge origin
        idx = torch.arange(nb_points, device=distance_matrix.device).long()
        idx = torch.repeat_interleave(idx, effective_nb_neighbors)

        # Edge features
        edge_feats = []
        for ind_batch in range(size_batch):
            edge_feats.append(
                pcloud[ind_batch, neighbors[ind_batch]] - pcloud[ind_batch, idx]
            )
        edge_feats = torch.cat(edge_feats, 0)

        # Handle batch dimension to get indices of nearest neighbors
        for ind_batch in range(1, size_batch):
            neighbors[ind_batch] = neighbors[ind_batch] + ind_batch * nb_points
        neighbors = neighbors.view(-1)

        # Create graph
        graph = Graph(
            neighbors,
            edge_feats,
            effective_nb_neighbors,
            [size_batch * nb_points, size_batch * nb_points],
        )

        return graph

    @staticmethod
    def construct_graph_in_chunks(pcloud, nb_neighbors, chunk_size):
        # Size
        size_batch, nb_points, _ = pcloud.shape
        assert size_batch == 1, "For construction of graph in chucks, the batch size should be 1, got %d." % size_batch


        # Find nearest neighbors
        #distance_matrix = -1 * torch.ones([nb_points, nb_points], dtype=torch.float32, device=pcloud.device)
        neighbors = -1 * torch.ones([nb_points, nb_neighbors], dtype=torch.int64, device=pcloud.device)
        idx = np.arange(nb_points)
        for b in iterate_in_chunks(idx, chunk_size):
            # Distance between points
            points_curr = torch.transpose(pcloud[:, b], 0, 1)
            #distance_matrix_curr = torch.sum(torch.pow(points_curr - pcloud, 2), -1)
            distance_matrix_curr = torch.sum(points_curr ** 2, -1) - 2 * torch.sum(points_curr * pcloud, dim=-1) + torch.sum(pcloud ** 2, -1)
            #distance_matrix[b, :] = distance_matrix_curr

            # Find nearest neighbors
            neighbors_curr = torch.argsort(distance_matrix_curr, -1)[..., :nb_neighbors]
            neighbors[b, :] = neighbors_curr

        assert torch.all(neighbors >= 0), "Problem with nearest neighbors computation. Not all indices filled correctly."
        #assert torch.all(distance_matrix >= 0), "Problem with distance matrix computation. Not all distances filled correctly."

        effective_nb_neighbors = neighbors.shape[-1]
        edge_feats = torch.empty(0, device=pcloud.device)

        neighbors = neighbors.view(-1)

        # Create graph
        graph = Graph(
            neighbors,
            edge_feats,
            effective_nb_neighbors,
            [size_batch * nb_points, size_batch * nb_points],
        )

        return graph
