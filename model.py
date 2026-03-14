import torch
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, softmax
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_geometric.nn.inits import glorot, zeros
from kan import *
from layer.MultKAN_type import MultKAN
from layer.KANLayer_cus import KANLayer_cus
from layer.bsrbf_kan import BSRBF_KANLayer
from layer.cheby_kan import ChebyKANLayer
from layer.efficient_kan import EfficientKANLinear
from layer.fast_kan import FastKANLayer
from layer.faster_kan import FasterKANLayer
from layer.fourier_kan import NaiveFourierKANLayer
from layer.jacobi_kan import JacobiKANLayer
from layer.laplace_kan import NaiveLaplaceKANLayer
from layer.legendre_kan import RecurrentLegendreLayer
from layer.wavlet_kan import WavletKANLinear
from layer.TransKGNN import KANTransformerEncoderLayer
num_atom_type = 120 #including the extra mask tokens
num_chirality_tag = 3

num_bond_type = 7 #including aromatic and self-loop edge, and extra masked tokens
num_bond_direction = 3 

class GINConv(MessagePassing):
    """
    Extension of GIN aggregation to incorporate edge information by concatenation.

    Args:
        emb_dim (int): dimensionality of embeddings for nodes and edges.
        embed_input (bool): whether to embed input or not. 
        

    See https://arxiv.org/abs/1810.00826
    """
    def __init__(self, emb_dim, aggr="add", kan_mlp = False, kan_mp = False, kan_type = None, grid =None ,k = None, neuron_fun =None ):
        super(GINConv, self).__init__()
        
        #node updateing stage
        self.kan_mlp = kan_mlp
        if kan_mlp == "mlp":
            print("Using MLP in updating")
            self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.ReLU(), torch.nn.Linear(2*emb_dim, emb_dim))
        elif kan_mlp == "kan":
            if kan_type == "ori":
                print(f"Using KAN in updating, grid:{grid}, k:{k}")
                # self.mlp = KAN(width = [emb_dim, 2*emb_dim, emb_dim], grid = grid, k = k).speed()
                # self.mlp =  torch.nn.Sequential(
                #     #normal 
                #     KANLayer_cus(in_dim = emb_dim, out_dim = 2*emb_dim, num = grid, k = k, return_y= True, neuron_fun = neuron_fun), 
                #     KANLayer_cus(in_dim = 2*emb_dim, out_dim = emb_dim, num = grid, k = k, return_y= True, neuron_fun = neuron_fun),
                # )  
                self.mlp = MultKAN(width=[emb_dim,2*emb_dim,emb_dim], grid=grid, k=k)                                               
            else:
                raise ValueError("Wrong kan type")            
        else:
            raise ValueError("Wrong mlp")
          

        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)
        self.aggr = aggr

        #add mp layer
        self.kan_mp = kan_mp
        if kan_mp == "kan":
            print("using KAN message passing")
            self.message_passing_kan = KANLayer_cus(in_dim = emb_dim, out_dim = emb_dim, num = grid, k = k,
                                                    return_y= True, neuron_fun = "mean", use_base= False)

    def forward(self, x, edge_index, edge_attr):
        #add self loops in the edge space
        edge_index, _ = add_self_loops(edge_index, num_nodes = x.size(0))

        #add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:,0] = 4 #bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        #change to one-hot
        #bond_type_one_hot = torch.nn.functional.one_hot(edge_attr[:, 0].long(), num_classes=num_bond_type)
        #bond_direction_one_hot = torch.nn.functional.one_hot(edge_attr[:, 1].long(), num_classes=num_bond_direction)
        #edge_embeddings = torch.cat([bond_type_one_hot, bond_direction_one_hot], dim=1)
        
        edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(edge_attr[:, 1])
        return self.propagate(edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        # Default message is x_j + edge_attr
        msg = x_j + edge_attr
        #add mp_kan
        if self.kan_mp == "kan":
            kan_msg = self.message_passing_kan(msg)
            return msg + kan_msg
        else:
            return msg

    def update(self, aggr_out):
        return self.mlp(aggr_out)


class GCNConv(MessagePassing):

    def __init__(self, in_dim, out_dim, aggr="add", is_kan = False):
        super(GCNConv, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.linear = torch.nn.Linear(in_dim, out_dim)
        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, out_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, out_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

        self.aggr = aggr

    def norm(self, edge_index, num_nodes, dtype):
        ### assuming that self-loops have been already added in edge_index
        edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype, device=edge_index.device)
        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, edge_attr):
        #add self loops in the edge space
        edge_index, _ = add_self_loops(edge_index, num_nodes = x.size(0))

        #add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:, 0] = 4 #bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(edge_attr[:, 1])

        norm = self.norm(edge_index, x.size(0), x.dtype)

        x = self.linear(x)
        
        return self.propagate(edge_index, x=x, edge_attr=edge_embeddings, norm=norm)

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * (x_j + edge_attr)


class GATConv(MessagePassing):
    def __init__(self, emb_dim, heads=2, negative_slope=0.2, aggr="add", is_kan = False):
        super(GATConv, self).__init__()

        self.aggr = aggr

        self.emb_dim = emb_dim
        self.heads = heads
        self.negative_slope = negative_slope

        self.weight_linear = torch.nn.Linear(emb_dim, heads * emb_dim)
        self.att = torch.nn.Parameter(torch.Tensor(1, heads, 2 * emb_dim))

        self.bias = torch.nn.Parameter(torch.Tensor(emb_dim))

        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, heads * emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, heads * emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x, edge_index, edge_attr):

        #add self loops in the edge space
        edge_index, _ = add_self_loops(edge_index, num_nodes = x.size(0))

        #add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:,0] = 4 #bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim = 0)

        edge_embeddings = self.edge_embedding1(edge_attr[:,0]) + self.edge_embedding2(edge_attr[:,1])
        
        x = self.weight_linear(x)
        return self.propagate(edge_index, x=x, edge_attr=edge_embeddings, aggr = self.aggr)

    def message(self, edge_index, x_i, x_j, edge_attr):
        edge_attr = edge_attr.view(-1, self.heads, self.emb_dim)
        x_i = x_i.view(-1, self.heads, self.emb_dim)  # Restore heads dimension
        x_j = x_j.view(-1, self.heads, self.emb_dim)  # Restore heads dimension
        x_j += edge_attr

        alpha = (torch.cat([x_i, x_j], dim=-1)  * self.att).sum(dim=-1)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index[0])

        return (x_j * alpha.view(-1, self.heads, 1)).view(-1, self.heads * self.emb_dim)

    def update(self, aggr_out):
        aggr_out = aggr_out.view(-1, self.heads, self.emb_dim)
        aggr_out = aggr_out.mean(dim=1) + self.bias

        return aggr_out


class GraphSAGEConv(MessagePassing):
    def __init__(self, emb_dim, aggr = "mean", is_kan = False):
        super(GraphSAGEConv, self).__init__()

        self.emb_dim = emb_dim
        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

        self.aggr = aggr

    def forward(self, x, edge_index, edge_attr):
        #add self loops in the edge space
        edge_index, _ = add_self_loops(edge_index, num_nodes = x.size(0))

        #add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:,0] = 4 #bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim = 0)

        edge_embeddings = self.edge_embedding1(edge_attr[:,0]) + self.edge_embedding2(edge_attr[:,1])

        x = self.linear(x)

        return self.propagate(edge_index, x=x, edge_attr=edge_embeddings, aggr = self.aggr)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return F.normalize(aggr_out, p=2, dim=-1)




class GNN_imp_estimator(torch.nn.Module):
    """

    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        JK (str): last, concat, max or sum.
        max_pool_layer (int): the layer from which we use max pool rather than add pool for neighbor aggregation
        drop_ratio (float): dropout rate
        gnn_type: gin, gcn, graphsage, gat

    Output:
        node representations

    """

    def __init__(self, num_layer, emb_dim, JK="last", drop_ratio=0):
        super(GNN_imp_estimator, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.x_embedding1 = torch.nn.Embedding(num_atom_type, emb_dim)
        self.x_embedding2 = torch.nn.Embedding(num_chirality_tag, emb_dim)

        torch.nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.x_embedding2.weight.data)

        ###List of MLPs
        self.gnns = torch.nn.ModuleList()
        self.gnns.append(GCNConv(emb_dim, 128))
        self.gnns.append(GCNConv(128, 64))
        self.gnns.append(GCNConv(64, 32))

        ###List of batchnorms
        self.batch_norms = torch.nn.ModuleList()
        self.batch_norms.append(torch.nn.BatchNorm1d(128))
        self.batch_norms.append(torch.nn.BatchNorm1d(64))
        self.batch_norms.append(torch.nn.BatchNorm1d(32))

        self.linear = torch.nn.Linear(32, 1)

    def forward(self, *argv):
        if len(argv) == 4:
            x, edge_index, edge_attr, batch = argv[0], argv[1], argv[2], argv[3]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        else:
            raise ValueError("unmatched number of arguments.")

        x = self.x_embedding1(x[:, 0]) + self.x_embedding2(x[:, 1])

        h_list = [x]
        for layer in range(len(self.gnns)):
            h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            if layer == len(self.gnns) - 1:
                # remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)
            h_list.append(h)

        node_representation = h_list[-1]
        node_representation = self.linear(node_representation)
        node_representation = softmax(node_representation, batch)

        return node_representation


class GNN(torch.nn.Module):
    """
    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        JK (str): last, concat, max or sum.
        max_pool_layer (int): the layer from which we use max pool rather than add pool for neighbor aggregation
        drop_ratio (float): dropout rate
        gnn_type: gin, gcn, graphsage, gat

    Output:
        node representations
    """
    def __init__(self, num_layer, emb_dim, JK = "last", drop_ratio=0, gnn_type = "gin",
                 kan_mlp = False, kan_mp = False, kan_type = None, grid = None, k = None, neuron_fun =None,
                 use_transformer=False, num_heads=None):
        super(GNN, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.use_transformer = use_transformer
        dim_feedforward = 2*emb_dim

        # if self.num_layer < 2:
        #     raise ValueError("Number of GNN layers must be greater than 1.")

        self.x_embedding1 = torch.nn.Embedding(num_atom_type, emb_dim)
        self.x_embedding2 = torch.nn.Embedding(num_chirality_tag, emb_dim)

        torch.nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.x_embedding2.weight.data)

        ###List of MLPs
        print("using neuron_fun: ", neuron_fun)
        self.gnns = torch.nn.ModuleList()
        for layer in range(num_layer):
            if gnn_type == "gin":
                self.gnns.append(GINConv(emb_dim, aggr="add", kan_mlp = kan_mlp, kan_mp = kan_mp, kan_type= kan_type, grid = grid, k = k, neuron_fun = neuron_fun))
                # if use_transformer == "mlp":
                #     print("using mlp transformer")
                #     self.gnns.append(torch.nn.TransformerEncoderLayer(d_model=emb_dim, nhead=num_heads, dim_feedforward=dim_feedforward))
                # elif use_transformer == "kan":
                #     print("using kan transformer")
                #     self.gnns.append(KANTransformerEncoderLayer(d_model=emb_dim, nhead=num_heads, dim_feedforward=dim_feedforward, grid = grid, k = k))
            
            elif gnn_type == "gcn":
                self.gnns.append(GCNConv(emb_dim, emb_dim, kan_mlp = kan_mlp))

            elif gnn_type == "gat":
                self.gnns.append(GATConv(emb_dim, kan_mlp = kan_mlp))

            elif gnn_type == "graphsage":
                self.gnns.append(GraphSAGEConv(emb_dim, kan_mlp = kan_mlp))



        ###List of batchnorms
        self.batch_norms = torch.nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))


    def forward(self, x, edge_index, edge_attr, batch):
        # Embed node features
        x = self.x_embedding1(x[:, 0]) + self.x_embedding2(x[:, 1])
        h = x
        h_list = [h]

        for layer in range(self.num_layer):
            h = h_list[layer]

            # If the current layer is a Transformer layer
            if (isinstance(self.gnns[layer], nn.TransformerEncoderLayer) or isinstance(self.gnns[layer], KANTransformerEncoderLayer)) :
                # Pad h
                h_padded, mask = to_dense_batch(h, batch)  # h_padded: [batch_size, max_num_nodes, emb_dim]
                batch_size, max_num_nodes, emb_dim = h_padded.shape
                # Reshape for Transformer input: [max_num_nodes, batch_size, emb_dim]
                h_padded = h_padded.permute(1, 0, 2)
                # Apply Transformer layer
                h_padded = self.gnns[layer](h_padded, src_key_padding_mask=~mask)# Invert the mask to indicate padding with True
                # Reshape back and unpad h
                h_padded = h_padded.permute(1, 0, 2)
                h = h_padded[mask].reshape(-1, emb_dim)
            else:
                # GNN layer: process h with edge_index and edge_attr
                h = self.gnns[layer](h, edge_index, edge_attr)

            h = self.batch_norms[layer](h)
            if layer == self.num_layer - 1:
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)

            h_list.append(h)

        # Combine node representations
        if self.JK == "concat":
            node_representation = torch.cat(h_list, dim=1)
        elif self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "max":
            node_representation = torch.stack(h_list, dim=0).max(dim=0)[0]
        elif self.JK == "sum":
            node_representation = torch.stack(h_list, dim=0).sum(dim=0)

        return node_representation


class GNN_graphpred(torch.nn.Module):
    """
    Extension of GIN to incorporate edge information by concatenation.

    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        num_tasks (int): number of tasks in multi-task learning scenario
        drop_ratio (float): dropout rate
        JK (str): last, concat, max or sum.
        graph_pooling (str): sum, mean, max, attention, set2set
        gnn_type: gin, gcn, graphsage, gat
        
    See https://arxiv.org/abs/1810.00826
    JK-net: https://arxiv.org/abs/1806.03536
    """
    def __init__(self, num_layer, emb_dim, num_tasks, JK = "last", drop_ratio = 0, graph_pooling = "mean", gnn_type = "gin",
                 kan_mlp = None, kan_mp = None, kan_type = None, grid = None, k = None, num_heads = None, neuron_fun = None, use_transformer = False):
        super(GNN_graphpred, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks
        self.kan_mlp = kan_mlp
        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.gnn = GNN(num_layer, emb_dim, JK, drop_ratio, gnn_type=gnn_type, 
                       kan_mlp = kan_mlp, kan_mp = kan_mp, kan_type = kan_type, grid = grid, k = k, num_heads = num_heads, neuron_fun= neuron_fun,
                       use_transformer = use_transformer)

        #Different kind of graph pooling
        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        elif graph_pooling == "attention":
            if self.JK == "concat":
                self.pool = GlobalAttention(gate_nn = torch.nn.Linear((self.num_layer + 1) * emb_dim, 1))
            else:
                self.pool = GlobalAttention(gate_nn = torch.nn.Linear(emb_dim, 1))
        elif graph_pooling[:-1] == "set2set":
            set2set_iter = int(graph_pooling[-1])
            if self.JK == "concat":
                self.pool = Set2Set((self.num_layer + 1) * emb_dim, set2set_iter)
            else:
                self.pool = Set2Set(emb_dim, set2set_iter)
        else:
            raise ValueError("Invalid graph pooling type.")

        #For graph-level binary classification
        if graph_pooling[:-1] == "set2set":
            self.mult = 2
        else:
            self.mult = 1
        
        # if kan_mlp == "kan":
        #     if self.JK == "concat":
        #         self.graph_pred_linear = KANLayer(in_dim = self.mult * (self.num_layer + 1) * self.emb_dim,
        #                                            out_dim =  self.num_tasks, num = grid, k = k)
        #     else:
        #         self.graph_pred_linear = KANLayer(in_dim = self.mult * self.emb_dim,
        #                                            out_dim =  self.num_tasks, num = grid, k = k)
        # else:
        if self.JK == "concat":
            self.graph_pred_linear = torch.nn.Linear(self.mult * (self.num_layer + 1) * self.emb_dim, self.num_tasks)
        else:
            self.graph_pred_linear = torch.nn.Linear(self.mult * self.emb_dim, self.num_tasks)

    def from_pretrained(self, model_file, device):
        if not model_file == "":
            self.gnn.load_state_dict(torch.load(model_file, map_location=lambda storage, loc: storage))
            self.gnn.to(device)

    def forward(self, *argv):
        if len(argv) == 4:
            x, edge_index, edge_attr, batch = argv[0], argv[1], argv[2], argv[3]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        else:
            raise ValueError("unmatched number of arguments.")
        
        node_representation = self.gnn(x, edge_index, edge_attr, batch)

        return self.graph_pred_linear(self.pool(node_representation, batch))


if __name__ == "__main__":
    pass

