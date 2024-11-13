import torch
import dgl
import numpy as np
import torch.nn.functional as F

from torch import nn
from model.MAGNN import MAGNNLinkPredictionModel

init = nn.init.xavier_uniform_


class AdaMH(torch.nn.Module):
    def __init__(self, datasets, num_circrna, mask_type, etypes_lists, device, args):
        super(AdaMH, self).__init__()
        self.num_circrna = num_circrna
        self.mask_type = mask_type
        self.device = device
        self.samples = args.samples
        self.hidden_dim = args.hidden_dim

        self.magnn_model = MAGNNLinkPredictionModel(
            [2, 2],
            6,
            etypes_lists,
            datasets.in_dims,
            self.hidden_dim,
            self.hidden_dim,
            args.num_heads,
            args.attn_vec_dim,
            args.rnn_type,
            args.dropout_rate
        ).to(device)

        self.adj = datasets.adj
        self.edge = datasets.edge
        self.features_list = datasets.features_list
        self.type_mask = datasets.type_mask

        self.mlp = MLP(self.hidden_dim * 2, self.hidden_dim * 4, 1).to(device)
        self.graph_processor = GraphDataProcessor(len(datasets.adj[0][0]), args.batch_size)

    def perturbed_encoder(self, embeddings, split_shape):
        noise = torch.rand(embeddings.shape).to(self.device)
        embeddings += (embeddings.sign() * noise.norm(2, dim=1, keepdim=True)) * 0.1
        mean_embeddings = torch.mean(embeddings, dim=0)
        return torch.split(mean_embeddings, split_shape, dim=0)

    def calculate_cl_loss(self, rna_embeddings, disease_embeddings, split_shape):
        combined_embeddings = torch.cat(
            (rna_embeddings.view(rna_embeddings.size(0), -1),
             disease_embeddings.view(disease_embeddings.size(0), -1)),
            dim=1
        )
        user_emb1, user_emb2 = self.perturbed_encoder(combined_embeddings, split_shape)
        item_emb1 = torch.mean(rna_embeddings.view(rna_embeddings.size(0), -1), dim=0)
        item_emb2 = torch.mean(disease_embeddings.view(disease_embeddings.size(0), -1), dim=0)

        user_emb1 = F.normalize(user_emb1, p=2, dim=0)
        user_emb2 = F.normalize(user_emb2, p=2, dim=0)
        item_emb1 = F.normalize(item_emb1, p=2, dim=0)
        item_emb2 = F.normalize(item_emb2, p=2, dim=0)

        pos_score_user = torch.exp(torch.sum(user_emb1 * user_emb2) / 0.2)
        total_score_user = torch.sum(torch.exp(user_emb1.unsqueeze(1) @ user_emb2.unsqueeze(0) / 0.2))

        pos_score_item = torch.exp(torch.sum(item_emb1 * item_emb2) / 0.2)
        total_score_item = torch.sum(torch.exp(item_emb1.unsqueeze(1) @ item_emb2.unsqueeze(0) / 0.2))

        cl_loss = -torch.log(pos_score_user / total_score_user) - torch.log(pos_score_item / total_score_item)
        return 0.5 * cl_loss

    def process_batch(self, batch, mask_key):
        data = self.graph_processor.minibatch(
            self.adj, self.edge, batch, self.device, self.samples, self.mask_type[mask_key], self.num_circrna
        )
        embeddings, _ = self.magnn_model(
            (data[0], self.features_list, self.type_mask, data[1], data[2])
        )
        circrna_emb, disease_emb = embeddings
        circrna_emb = circrna_emb.unsqueeze(1)
        disease_emb = disease_emb.unsqueeze(2)
        score = torch.bmm(circrna_emb, disease_emb)
        x = torch.cat(
            (circrna_emb.view(circrna_emb.size(0), -1),
             disease_emb.view(disease_emb.size(0), -1)),
            dim=1
        )
        x = self.mlp(x)
        return x, score, circrna_emb, disease_emb

    def forward(self, pos_batch, neg_batch):
        x_pos, pos_score, pos_circrna_emb, pos_disease_emb = self.process_batch(pos_batch, "use_mask")
        x_neg, neg_score, neg_circrna_emb, neg_disease_emb = self.process_batch(neg_batch, "no_mask")
        neg_score = -neg_score
        train_loss = -torch.mean(F.logsigmoid(pos_score) + F.logsigmoid(neg_score))

        cl_loss = self.calculate_cl_loss(
            pos_circrna_emb.squeeze(1), pos_disease_emb.squeeze(2), pos_disease_emb.size(1)
        )

        return x_pos, x_neg, train_loss, cl_loss


class GraphDataProcessor:
    def __init__(self, num_nodes, dim):
        super(GraphDataProcessor, self).__init__()
        self.num_nodes = num_nodes
        self.dim = dim
        self.init_node_weights(self.num_nodes)

    def init_node_weights(self, num_nodes):
        self.node_weights = nn.Parameter(init(torch.empty(1, num_nodes)))

    def get_node_weights(self, nodes):
        weights = []
        for node in nodes:
            weights.append(float(self.node_weights[0][node].data))
        return np.array(weights)

    def process_training_data(self, adjacency_list, edge_indices, samples=None, exclude=None, offset=0, mode=0):
        edge_list = []
        node_set = set()
        all_indices = []

        for adj_row_str, indices in zip(adjacency_list, edge_indices):
            row_parsed = list(map(int, adj_row_str.strip().split()))
            src_node = row_parsed[0]
            neighbors = row_parsed[1:]
            node_set.add(src_node)

            if neighbors:
                neighbors = np.array(neighbors)
                indices = np.array(indices)

                if samples is not None:
                    sampled_neighbors, selected_indices = self.sample_neighbors(neighbors, indices, samples)
                else:
                    sampled_neighbors = neighbors
                    selected_indices = indices

                if exclude is not None:
                    mask = self.create_exclude_mask(selected_indices, exclude, offset, mode)
                    sampled_neighbors = sampled_neighbors[mask]
                    selected_indices = selected_indices[mask]
            else:
                sampled_neighbors = np.array([src_node])
                selected_indices = np.full((1, indices.shape[1]), src_node)
                if mode == 1:
                    selected_indices += offset
            all_indices.append(selected_indices)
            for dst_node in sampled_neighbors:
                node_set.add(dst_node)
                edge_list.append((src_node, dst_node))
        sorted_node_list = sorted(node_set)
        node_id_mapping = {node_id: idx for idx, node_id in enumerate(sorted_node_list)}
        mapped_edges = [(node_id_mapping[src], node_id_mapping[dst]) for src, dst in edge_list]
        all_indices = np.vstack(all_indices)

        return mapped_edges, all_indices, len(node_set), node_id_mapping

    def sample_neighbors(self, neighbors, indices, samples):
        unique_neighbors, counts = np.unique(neighbors, return_counts=True)
        weight_values = self.get_node_weights(unique_neighbors)
        sorted_node_indices = np.argsort(weight_values)
        samples = min(samples, len(unique_neighbors))
        selected_node_indices = sorted_node_indices[:samples]
        selected_nodes = unique_neighbors[selected_node_indices]

        node_positions = {node: np.where(neighbors == node)[0] for node in selected_nodes}
        position_indices = np.concatenate([positions for positions in node_positions.values()])
        position_indices.sort()

        sampled_neighbors = neighbors[position_indices]
        sampled_indices = indices[position_indices]

        return sampled_neighbors, sampled_indices

    def create_exclude_mask(self, indices, exclude, offset, mode):
        if mode == 0:
            idx_columns = [0, 1, -1, -2]
        else:
            idx_columns = [1, 0, -2, -1]

        selected_indices_array = indices[:, idx_columns]

        if offset is not None:
            selected_indices_array[:, [1, 3]] -= offset

        mask = []
        for idx in selected_indices_array:
            u1, a1, u2, a2 = idx
            if [u1, a1] in exclude or [u2, a2] in exclude:
                mask.append(False)
            else:
                mask.append(True)

        return np.array(mask)

    def minibatch(self, adjacency_lists_by_mode, edge_indices_by_mode, batch_pairs, device,
                  samples=None, masks_by_mode=None, offset=None):
        graph_lists = [[], []]
        result_indices_list = [[], []]
        batch_mapped_indices_list = [[], []]

        for mode in [0, 1]:
            adjacency_lists = adjacency_lists_by_mode[mode]
            edge_indices_list = edge_indices_by_mode[mode]
            masks_for_mode = masks_by_mode[mode]

            for adjacency_list, edge_indices, use_mask in zip(adjacency_lists, edge_indices_list, masks_for_mode):
                adjlist_rows = [adjacency_list[pair[mode]] for pair in batch_pairs]
                indices_rows = [edge_indices[pair[mode]] for pair in batch_pairs]

                if use_mask:
                    edges, result_indices, num_nodes, node_id_mapping = self.process_training_data(
                        adjlist_rows, indices_rows, samples, batch_pairs, offset, mode)
                else:
                    edges, result_indices, num_nodes, node_id_mapping = self.process_training_data(
                        adjlist_rows, indices_rows, samples, offset=offset, mode=mode)

                g = dgl.DGLGraph(multigraph=True)
                g.add_nodes(num_nodes)

                if edges:
                    sorted_indices = sorted(range(len(edges)), key=lambda i: edges[i])
                    sorted_edges = [edges[i] for i in sorted_indices]
                    src_nodes, dst_nodes = zip(*[(edge[1], edge[0]) for edge in sorted_edges])
                    g.add_edges(src_nodes, dst_nodes)
                    result_indices_tensor = torch.LongTensor([result_indices[i] for i in sorted_indices]).to(device)
                else:
                    result_indices_tensor = torch.LongTensor(result_indices).to(device)

                graph_lists[mode].append(g)
                idx_mapped = np.array([node_id_mapping[pair[mode]] for pair in batch_pairs])
                batch_mapped_indices_list[mode].append(idx_mapped)
                result_indices_list[mode].append(result_indices_tensor)

        return graph_lists, result_indices_list, batch_mapped_indices_list


class MLP(torch.nn.Module):

    def __init__(self, num_i, num_h, num_o):
        super(MLP, self).__init__()

        self.linear1 = torch.nn.Linear(num_i, num_h)
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(num_h, num_h)
        self.relu2 = torch.nn.ReLU()
        self.linear3 = torch.nn.Linear(num_h, num_o)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        x = self.sigmoid(x)
        return x


class IndexGenerator:
    def __init__(self, batch_size, num_data=None, indices=None, shuffle=True):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.iter_counter = 0
        self.data_init(num_data, indices)

    def data_init(self, num_data=None, indices=None):
        if indices is not None:
            self.indices = np.array(indices)
        elif num_data is not None:
            self.indices = np.arange(num_data)
        else:
            raise ValueError("One of num_data or indices must be provided. ")
        self.num_data = len(self.indices)
        if self.shuffle:
            np.random.shuffle(self.indices)
        self.iter_counter = 0

    def next(self):
        if self.num_iterations_left() <= 0:
            self.reset()
        start = self.iter_counter * self.batch_size
        end = start + self.batch_size
        batch_indices = self.indices[start:end]
        self.iter_counter += 1
        return batch_indices

    def num_iterations(self):
        return (self.num_data + self.batch_size - 1) // self.batch_size

    def num_iterations_left(self):
        return self.num_iterations() - self.iter_counter

    def reset(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
        self.iter_counter = 0
