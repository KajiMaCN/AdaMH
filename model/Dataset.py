import torch
from util.tools import load_data


class DataSet:
    def __init__(self):
        self.features_list = []
        self.in_dims = []
        self.init_dataset()

    def init_dataset(self):
        data_dict = load_data()
        self.adj = data_dict.get("adj")
        self.edge = data_dict.get("edge")
        self.adjM = data_dict.get("adjM")
        self.type_mask = data_dict.get("type_mask")
        self.pos_data = data_dict.get("pos_data")
        self.neg_data = data_dict.get("neg_data")

    def create_feature(self, feats_type, num_ntype, device):
        for i in range(num_ntype):
            node_indices = (self.type_mask == i)
            num_nodes = node_indices.sum()
            if feats_type == 0:
                dim = num_nodes
                self.in_dims.append(dim)
                indices = torch.arange(dim, device=device)
                indices = torch.stack([indices, indices], dim=0)
                values = torch.ones(dim, device=device)
                size = torch.Size([dim, dim])
                feature = torch.sparse_coo_tensor(indices, values, size).to(device)
            elif feats_type == 1:
                dim = 10
                self.in_dims.append(dim)
                feature = torch.zeros((num_nodes, dim), device=device)
            else:
                raise ValueError(f"Unsupported feats_type：{feats_type}")
            self.features_list.append(feature)
