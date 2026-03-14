import os.path as osp

from aug import TUDataset_aug as TUDataset
# from torch_geometric.data import DataLoader
from torch_geometric.loader import DataLoader

from gin_saliency import Encoder
from evaluate_embedding import evaluate_embedding
from model import *
from layer.MultKAN_type import MultKAN

from arguments import arg_parse
import random
from torch import nn
import numpy as np
import copy

from hosvd_loo import leave_one_out_hosvd_error


class EMA():
    "Adapted from https://github.com/GRAND-Lab/MERIT"
    "Multi-Scale Contrastive Siamese Networks for Self-Supervised Graph Representation Learning"
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def update_moving_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

def setup_seed(seed):

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)


class simsiam(nn.Module):
    def __init__(self, hidden_dim, num_gc_layers, pooling, beta=1., ema_decay=0.8):
        super(simsiam, self).__init__()

        self.beta = beta
        self.prior = args.prior

        if pooling == 'last':
            self.embedding_dim = hidden_dim
        else:
            self.embedding_dim = hidden_dim*num_gc_layers
        self.num_gc_layers = num_gc_layers
        self.ema_updater = EMA(ema_decay)
        self.online_encoder = Encoder(dataset_num_features, hidden_dim, num_gc_layers, pooling, kan_mlp=args.kan_mlp, 
                                kan_mp=args.kan_mp, kan_type=args.kan_type1, grid=args.grid, k=args.k, neuron_fun=args.neuron_fun)
        self.target_encoder = copy.deepcopy(self.online_encoder)

        self.predictor = self.load_pred(args, args.kan_pred_type)
        self.hosvd_saliency = None

        self.init_emb()

    def load_pred(self, args, kan_type):
        if kan_type=='mlp':
            proj_head = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim), nn.ReLU(inplace=True),
                                        nn.Linear(self.embedding_dim, self.embedding_dim))
            
        elif kan_type == "ori":
            print(f"Using KAN in updating, grid:{args.grid_pred}, k:{args.k_pred}")
            # self.mlp = KAN(width = [embedding_dim, 2*embedding_dim, embedding_dim], grid = grid, k = k).speed()
            proj_head =  MultKAN(width=[self.embedding_dim,self.embedding_dim], grid=args.grid_pred, k=args.k_pred)                                      
        else:
            raise ValueError("Wrong kan type")
        
        return proj_head

    def init_emb(self):
        # initrange = -1.5 / self.embedding_dim
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def update_ma(self):
        assert self.target_encoder is not None, 'target encoder has not been created yet'
        self.ema_updater.update_moving_average(self.target_encoder, self.online_encoder)

    def sim(self, h1, h2):
        z1 = F.normalize(h1, dim=-1, p=2)
        z2 = F.normalize(h2, dim=-1, p=2)
        return torch.mm(z1, z2.t())

    def loss_cross_view(self, h1, z):
        f = lambda x: torch.exp(x)
        cross_sim = f(self.sim(h1, z))
        # return -torch.log(cross_sim.diag() / cross_sim.sum(dim=-1))
        return -torch.log(cross_sim.diag()) 

    def loss_cross_net(self, h1, h2):
        f = lambda x: torch.exp(x)
        intra_sim = f(self.sim(h1, h1))
        inter_sim = f(self.sim(h1, h2))
        return -torch.log(inter_sim.diag() /
                        (intra_sim.sum(dim=-1) + inter_sim.sum(dim=-1) - intra_sim.diag()))
    
    def loss_neg_bank(self, rep, neg):
        f = lambda x: torch.exp(x)
        inter_sim = f(self.sim(rep, neg))
        # print(torch.max(inter_sim),torch.min(inter_sim))
        # return torch.log(inter_sim.diag().sum())
        return torch.log(inter_sim.diag())
        # return torch.log(inter_sim.sum(dim=-1))

    def update_hosvd_saliency(self):
        coef = []
        for i in range(self.num_gc_layers):
            coef.append(self.online_encoder.convs[i].nn.act_fun[-1].coef)
            # coef.append(self.target_encoder.convs[i].nn.act_fun[-1].coef)
        coef = torch.cat(coef, dim=1).permute(1,0,2) # coef shape (num_gc_layers*out_dim, in_dim, G+k)
        ranks = [20, 10, 5]
        errors = leave_one_out_hosvd_error(coef, ranks)
        self.hosvd_saliency = errors
    
    def get_saliency_scores(self, method):
        '''Method chosen from std, coef, or both.'''
        # get feature saliency
        feature_scores = 0
        if 'std' in method:
            feature_scores_std = []
            for i in range(self.num_gc_layers):
                feature_score = self.online_encoder.convs[i].nn.subnode_actscale
                # feature_score = self.target_encoder.convs[i].nn.subnode_actscale
                feature_score = torch.stack(feature_score, 0).mean(dim=0) if len(feature_score)>1 else feature_score[0]
                feature_scores_std.append(feature_score)
            feature_scores_std = torch.cat(feature_scores_std)
            feature_scores += feature_scores_std
        elif 'coef_var' in method:
            feature_scores_coef_var = []
            for i in range(self.num_gc_layers):
                # take last layer coef
                # coef = self.online_encoder.convs[i].nn.act_fun[-1].coef  # coef shape (in_dim, out_dim, G+k)
                coef = self.target_encoder.convs[i].nn.act_fun[-1].coef  # coef shape (in_dim, out_dim, G+k)
                variance = torch.var(coef, dim=2)
                feature_score = variance.sum(dim=0)
                feature_scores_coef_var.append(feature_score)
            feature_scores_coef_var = torch.cat(feature_scores_coef_var)
            feature_scores += feature_scores_coef_var
        elif 'coef_corr' in method:
            coef = []
            for i in range(self.num_gc_layers):
                coef.append(self.online_encoder.convs[i].nn.act_fun[-1].coef)
                # coef.append(self.target_encoder.convs[i].nn.act_fun[-1].coef)
            coef = torch.cat(coef, dim=1).permute(1,0,2) # coef shape (num_gc_layers*out_dim, in_dim, G+k)
            coef = coef.view(coef.size(0), -1)
            corr = torch.abs(torch.corrcoef(coef)).sum(dim=-1)
            feature_scores_coef_corr = 1/corr
            feature_scores += feature_scores_coef_corr
        elif 'hosvd' in method:
            feature_scores_hosvd = self.hosvd_saliency
            feature_scores += feature_scores_hosvd
        elif 'random' in method:
            feature_scores_rand = torch.normal(mean=torch.zeros(96), std=0.05)
            feature_scores += feature_scores_rand
        return feature_scores

    def get_hard_neg(self,rep,saliency_method):
        # get feature saliency
        feature_scores = self.get_saliency_scores(saliency_method)
        
        # perturb important dimensions
        feature_scores = (feature_scores-torch.min(feature_scores))/ \
            (torch.max(feature_scores)-torch.min(feature_scores))
        # thr = feature_scores.median()           # compute median :contentReference[oaicite:2]{index=2}
        # feature_scores.masked_fill_(feature_scores < thr, 0)
        
        feature_scores = feature_scores.unsqueeze(0).expand(rep.size(0),-1)
        perturbation = torch.normal(mean=0.2*feature_scores, std=0.05).to(device)
        # perturbation = torch.normal(mean=0.15*feature_scores, std=0.02).to(device)
        mask = torch.rand_like(perturbation) < 0.5
        perturbation[mask] = -perturbation[mask]
        
        hard_neg = rep + perturbation
        return hard_neg

    def forward(self, x, edge_index, batch, x_aug, edge_index_aug, batch_aug, saliency_method):
        self.update_ma()
        if x is None:
            x = torch.ones(batch.shape[0]).to(device)
        if x_aug is None:
            x_aug = torch.ones(batch_aug.shape[0]).to(device)
        # print(x.size())
        h_ol, _ = self.online_encoder(x, edge_index, batch)
        h_ol_aug, _ = self.online_encoder(x_aug, edge_index_aug, batch_aug)
        with torch.no_grad():
            h_tg, _ = self.target_encoder(x, edge_index, batch)
            h_tg_aug, _ = self.target_encoder(x_aug, edge_index_aug, batch_aug)
        
        z = self.predictor(h_ol)
        z_aug = self.predictor(h_ol_aug)

        l1 = self.beta * self.loss_cross_net(z, z_aug) + (1-self.beta) * self.loss_cross_view(z, h_tg_aug)
        l2 = self.beta * self.loss_cross_net(z_aug, z) + (1-self.beta) * self.loss_cross_view(z_aug, h_tg)

        loss = (l1 + l2)/2
        loss = loss.mean()

        # hard negative loss
        if saliency_method != 'None':
            # # online z vs. online z
            # hard_neg = self.get_hard_neg(z,saliency_method)
            # hard_neg_aug = self.get_hard_neg(z_aug,saliency_method)
            # loss_neg = self.loss_neg_bank(z, hard_neg)
            # loss_neg_aug = self.loss_neg_bank(z_aug, hard_neg_aug)
            # loss_hardneg = (loss_neg.mean() + loss_neg_aug.mean())/2
            # online z vs. target h
            hard_neg = self.get_hard_neg(h_tg,saliency_method)
            hard_neg_aug = self.get_hard_neg(h_tg_aug,saliency_method)
            loss_neg = self.loss_neg_bank(h_ol, hard_neg.detach())
            loss_neg_aug = self.loss_neg_bank(h_ol_aug, hard_neg_aug.detach())
            loss_hardneg = (loss_neg.mean() + loss_neg_aug.mean())/2
        else:
            loss_hardneg = 0

        loss = loss+loss_hardneg

        return loss


if __name__ == '__main__':

    args = arg_parse()

    setup_seed(args.seed)

    accuracies = {'val': [], 'test': []}
    imp_batch_size = 2048
    epochs = args.epochs
    log_interval = args.log_interval
    batch_size = args.batch_size
    lr = args.lr
    DS = args.DS
    num_workers = args.num_workers
    path = 'YOUR_PATH' + DS

    dataset = TUDataset(path, name=DS, aug=args.aug, rho=args.rho)
    dataset_eval = TUDataset(path, name=DS, aug='none')
    print(len(dataset))
    print(dataset.get_num_feature())
    try:
        dataset_num_features = dataset.get_num_feature()
    except:
        dataset_num_features = 1

    dataloader_eval = DataLoader(dataset_eval, batch_size=batch_size, num_workers=2*num_workers)

    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    model = simsiam(args.hidden_dim, args.num_gc_layers, args.pooling, args.beta, args.ema_decay).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print('================')
    print('dataset: {}'.format(DS))
    print('lr: {}'.format(lr))
    print('num_features: {}'.format(dataset_num_features))
    print('hidden_dim: {}'.format(args.hidden_dim))
    print('num_gc_layers: {}'.format(args.num_gc_layers))
    print('pooling: {}'.format(args.pooling))
    print('================')
    args.saliency_method = set(args.saliency_method)
    
    best_test_acc = 0
    for epoch in range(1, epochs+1):
        loss_all = 0
        dataset.aug = args.aug
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
        model.train()
        torch.set_grad_enabled(True)
        for step, data in enumerate(dataloader):
            data, data_aug = data
            optimizer.zero_grad()
            
            node_num, _ = data.x.size()
            data = data.to(device)
            data_aug = data_aug.to(device)
            
            ## update grids
            if step!=0 and step % args.update_grid_freq==0:
                model.online_encoder.update_kan_grid(data.x, data.edge_index, data.batch)
                model.target_encoder.update_kan_grid(data.x, data.edge_index, data.batch)

            ## calculate and update hosvd dimensions
            if 'hosvd' in args.saliency_method and step%args.hosvd_update_freq==0:
                model.update_hosvd_saliency()
            loss = model(data.x, data.edge_index, data.batch,
                      data_aug.x, data_aug.edge_index, data_aug.batch, args.saliency_method)
            
            loss_all += loss.item() * data.num_graphs
            loss.backward()
            optimizer.step()
            model.update_ma()
        print('Epoch {}, Loss {}'.format(epoch, loss_all / len(dataloader)))

        if epoch % log_interval == 0:
            model.eval()
            emb, y = model.online_encoder.get_embeddings(dataloader_eval)
            # emb, y = model.target_encoder.get_embeddings(dataloader_eval)
            acc_val, acc = evaluate_embedding(emb, y)
            accuracies['val'].append(acc_val)
            accuracies['test'].append(acc)
            if acc>best_test_acc:
                best_test_acc = acc
    
    # save trained model
    print("Best Test Acc: {}".format(best_test_acc))
    if args.model_save_path != 'None':
        model_path = args.model_save_path
        torch.save(model.state_dict(), model_path)
