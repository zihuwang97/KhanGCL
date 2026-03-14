import os.path as osp
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.utils import softmax

import numpy as np
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
import sys
import torch_scatter

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


class Encoder(torch.nn.Module):
    def __init__(self, num_features, dim, num_gc_layers, pooling, kan_mlp=False, 
                 kan_mp=False, kan_type=None, grid=None, k=None, neuron_fun=None, device='cuda:0'):
        super(Encoder, self).__init__()

        self.num_gc_layers = num_gc_layers
        self.pooling = pooling

        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.dim = dim

        if kan_mlp=='mlp':
            for i in range(num_gc_layers):
                if i:
                    nn = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
                else:
                    nn = Sequential(Linear(num_features, dim), ReLU(), Linear(dim, dim))
                conv = GINConv(nn)
                bn = torch.nn.BatchNorm1d(dim)

                self.convs.append(conv)
                self.bns.append(bn)
        elif kan_mlp=='kan':
            for i in range(num_gc_layers):
                if kan_type == "ori":
                    print(f"Using KAN in updating, grid:{grid}, k:{k}")
                    # self.mlp = KAN(width = [emb_dim, 2*emb_dim, emb_dim], grid = grid, k = k).speed()
                    if i:
                        nn =  MultKAN(width=[dim,dim,dim], grid=grid, k=k)
                    else:
                        nn =  MultKAN(width=[num_features,dim,dim], grid=grid, k=k)
                    # nn =  torch.nn.Sequential(
                    #     #normal 
                    #     KANLayer_cus(in_dim=dim, out_dim=dim, num=grid, k=k, return_y= True, neuron_fun=neuron_fun), 
                    #     KANLayer_cus(in_dim=dim, out_dim=dim, num=grid, k=k, return_y= True, neuron_fun=neuron_fun),
                    # )
                elif kan_type == "bsrbf":
                    print(f"Using bsrbf KAN in updating, grid:{grid}, k:{k}")
                    nn =  torch.nn.Sequential(
                        #normal 
                        BSRBF_KANLayer(input_dim = dim if i else num_features, output_dim = dim, grid_size = grid, spline_order = k), 
                        BSRBF_KANLayer(input_dim = dim, output_dim = dim, grid_size = grid, spline_order = k), 
                    )
                elif kan_type == "cheby":
                    print(f"Using cheby KAN in updating, degree:{k}")
                    nn =  torch.nn.Sequential(
                        #normal 
                        ChebyKANLayer(input_dim = dim if i else num_features, output_dim = dim, degree = k),
                        torch.nn.LayerNorm(dim), 
                        ChebyKANLayer(input_dim = dim, output_dim = dim, degree = k), 
                    )
                elif kan_type == 'eff':
                    print(f"Using efficient KAN in updating, grid:{grid}, k:{k}")
                    nn =  torch.nn.Sequential(
                        #normal 
                        EfficientKANLinear(in_dim = dim if i else num_features, out_dim = dim, num = grid, k = k), 
                        EfficientKANLinear(in_dim = dim, out_dim = dim, num = grid, k = k),
                    )
                elif kan_type == 'fast':
                    print(f"Using fast KAN in updating, num_grids:{grid}")
                    nn =  torch.nn.Sequential(
                        #normal 
                        FastKANLayer(input_dim = dim if i else num_features, output_dim = dim, num_grids = grid), 
                        FastKANLayer(input_dim = dim, output_dim = dim, num_grids = grid), 
                    )
                elif kan_type == 'faster':
                    print(f"Using faster KAN in updating, num_grids:{grid}")
                    nn =  torch.nn.Sequential(
                        #normal 
                        FasterKANLayer(input_dim = dim if i else num_features, output_dim = dim, num_grids = grid), 
                        FasterKANLayer(input_dim = dim, output_dim = dim, num_grids = grid), 
                    )                                                      
                else:
                    raise ValueError("Wrong kan type")
                conv = GINConv(nn)
                bn = torch.nn.BatchNorm1d(dim)

                self.convs.append(conv)
                self.bns.append(bn)
        else:
            raise ValueError("Wrong mlp")
        
    def update_kan_grid(self, x, edge_index, batch):
        if x is None:
            x = torch.ones((batch.shape[0], 1)).to(device)

        for i in range(self.num_gc_layers):
            if isinstance(x, torch.Tensor):
                x = (x, x)
            out = self.convs[i].propagate(edge_index, x=x)
            x_r = x[1]
            if x_r is not None:
                out = out + (1 + self.convs[i].eps) * x_r

            # UPDATE GRID HERE
            self.convs[i].nn.update_grid(out)
            x = self.convs[i].nn(out)
            x = F.relu(x)
            x = self.bns[i](x)

    def forward(self, x, edge_index, batch):

        if x is None:
            x = torch.ones((batch.shape[0], 1)).to(device)

        xs = []
        for i in range(self.num_gc_layers):
            x = F.relu(self.convs[i](x, edge_index))
            # x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            xs.append(x)

        if self.pooling == 'last':
            x = global_add_pool(xs[-1], batch)
        else:
            xpool = [global_add_pool(x, batch) for x in xs]
            x = torch.cat(xpool, 1)

        return x, torch.cat(xs, 1)

    def get_embeddings(self, loader):

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ret = []
        y = []
        with torch.no_grad():
            for data in loader:

                data = data[0]
                data.to(device)
                x, edge_index, batch = data.x, data.edge_index, data.batch
                if x is None:
                    x = torch.ones((batch.shape[0], 1)).to(device)
                x, _ = self.forward(x, edge_index, batch)

                ret.append(x.cpu().numpy())
                y.append(data.y.cpu().numpy())
        ret = np.concatenate(ret, 0)
        y = np.concatenate(y, 0)
        return ret, y

    def get_embeddings_v(self, loader):

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ret = []
        y = []
        with torch.no_grad():
            for n, data in enumerate(loader):
                data.to(device)
                x, edge_index, batch = data.x, data.edge_index, data.batch
                if x is None:
                    x = torch.ones((batch.shape[0],1)).to(device)
                x_g, x = self.forward(x, edge_index, batch, None)
                x_g = x_g.cpu().numpy()
                ret = x.cpu().numpy()
                y = data.edge_index.cpu().numpy()
                print(data.y)
                if n == 1:
                   break

        return x_g, ret, y



class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        try:
            num_features = dataset.num_features
        except:
            num_features = 1
        dim = 32

        self.encoder = Encoder(num_features, dim)

        self.fc1 = Linear(dim*5, dim)
        self.fc2 = Linear(dim, dataset.num_classes)

    def forward(self, x, edge_index, batch):
        if x is None:
            x = torch.ones(batch.shape[0]).to(device)

        x, _ = self.encoder(x, edge_index, batch)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)


def train(epoch):
    model.train()

    if epoch == 51:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.5 * param_group['lr']

    loss_all = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data.x, data.edge_index, data.batch)
        loss = F.nll_loss(output, data.y)
        loss.backward()
        loss_all += loss.item() * data.num_graphs
        optimizer.step()

    return loss_all / len(train_dataset)

def test(loader):
    model.eval()

    correct = 0
    for data in loader:
        data = data.to(device)
        output = model(data.x, data.edge_index, data.batch)
        pred = output.max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
    return correct / len(loader.dataset)

if __name__ == '__main__':

    for percentage in [ 1.]:
        for DS in [sys.argv[1]]:
            if 'REDDIT' in DS:
                epochs = 200
            else:
                epochs = 100
            path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', DS)
            accuracies = [[] for i in range(epochs)]
            #kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)
            dataset = TUDataset(path, name=DS) #.shuffle()
            num_graphs = len(dataset)
            print('Number of graphs', len(dataset))
            dataset = dataset[:int(num_graphs * percentage)]
            dataset = dataset.shuffle()

            kf = KFold(n_splits=10, shuffle=True, random_state=None)
            for train_index, test_index in kf.split(dataset):

                # x_train, x_test = x[train_index], x[test_index]
                # y_train, y_test = y[train_index], y[test_index]
                train_dataset = [dataset[int(i)] for i in list(train_index)]
                test_dataset = [dataset[int(i)] for i in list(test_index)]
                print('len(train_dataset)', len(train_dataset))
                print('len(test_dataset)', len(test_dataset))

                train_loader = DataLoader(train_dataset, batch_size=128)
                test_loader = DataLoader(test_dataset, batch_size=128)
                # print('train', len(train_loader))
                # print('test', len(test_loader))

                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model = Net().to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

                for epoch in range(1, epochs+1):
                    train_loss = train(epoch)
                    train_acc = test(train_loader)
                    test_acc = test(test_loader)
                    accuracies[epoch-1].append(test_acc)
                    tqdm.write('Epoch: {:03d}, Train Loss: {:.7f}, '
                          'Train Acc: {:.7f}, Test Acc: {:.7f}'.format(epoch, train_loss,
                                                                       train_acc, test_acc))
            tmp = np.mean(accuracies, axis=1)
            print(percentage, DS, np.argmax(tmp), np.max(tmp), np.std(accuracies[np.argmax(tmp)]))
            input()
