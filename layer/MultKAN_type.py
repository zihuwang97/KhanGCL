import torch
import torch.nn as nn
import numpy as np
# from .KANLayer import KANLayer
import os
import random
from layer.KANLayer_cus import KANLayer_cus


class MultKAN(nn.Module):
    '''
    KAN class
    
    Attributes:
    -----------
        grid : int
            the number of grid intervals
        k : int
            spline order
        act_fun : a list of KANLayers
        symbolic_fun: a list of Symbolic_KANLayer
        depth : int
            depth of KAN
        width : list
            number of neurons in each layer.
            Without multiplication nodes, [2,5,5,3] means 2D inputs, 3D outputs, with 2 layers of 5 hidden neurons.
            With multiplication nodes, [2,[5,3],[5,1],3] means besides the [2,5,53] KAN, there are 3 (1) mul nodes in layer 1 (2). 
        mult_arity : int, or list of int lists
            multiplication arity for each multiplication node (the number of numbers to be multiplied)
        grid : int
            the number of grid intervals
        k : int
            the order of piecewise polynomial
        base_fun : fun
            residual function b(x). an activation function phi(x) = sb_scale * b(x) + sp_scale * spline(x)
        symbolic_fun : a list of Symbolic_KANLayer
            Symbolic_KANLayers
        symbolic_enabled : bool
            If False, the symbolic front is not computed (to save time). Default: True.
        width_in : list
            The number of input neurons for each layer
        width_out : list
            The number of output neurons for each layer
        base_fun_name : str
            The base function b(x)
        grip_eps : float
            The parameter that interpolates between uniform grid and adaptive grid (based on sample quantile)
        node_bias : a list of 1D torch.float
        node_scale : a list of 1D torch.float
        subnode_bias : a list of 1D torch.float
        subnode_scale : a list of 1D torch.float
        symbolic_enabled : bool
            when symbolic_enabled = False, the symbolic branch (symbolic_fun) will be ignored in computation (set to zero)
        affine_trainable : bool
            indicate whether affine parameters are trainable (node_bias, node_scale, subnode_bias, subnode_scale)
        sp_trainable : bool
            indicate whether the overall magnitude of splines is trainable
        sb_trainable : bool
            indicate whether the overall magnitude of base function is trainable
        save_act : bool
            indicate whether intermediate activations are saved in forward pass
        node_scores : None or list of 1D torch.float
            node attribution score
        edge_scores : None or list of 2D torch.float
            edge attribution score
        subnode_scores : None or list of 1D torch.float
            subnode attribution score
        cache_data : None or 2D torch.float
            cached input data
        acts : None or a list of 2D torch.float
            activations on nodes
        auto_save : bool
            indicate whether to automatically save a checkpoint once the model is modified
        state_id : int
            the state of the model (used to save checkpoint)
        ckpt_path : str
            the folder to store checkpoints
        round : int
            the number of times rewind() has been called
        device : str
    '''
    def __init__(self, width=None, grid=3, k=3, mult_arity = 2, base_fun='silu', neuron_fun='sum', 
                 affine_trainable=False, grid_eps=0.02, grid_range=[-1, 1], sp_trainable=True, 
                 sb_trainable=True, seed=1, save_act=True, auto_save=True, first_init=True, 
                 ckpt_path='./model', state_id=0, round=0, device='cpu', update_grid=False):
        '''
        initalize a KAN model
        
        Args:
        -----
            width : list of int
                Without multiplication nodes: :math:`[n_0, n_1, .., n_{L-1}]` specify the number of neurons in each layer (including inputs/outputs)
                With multiplication nodes: :math:`[[n_0,m_0=0], [n_1,m_1], .., [n_{L-1},m_{L-1}]]` specify the number of addition/multiplication nodes in each layer (including inputs/outputs)
            grid : int
                number of grid intervals. Default: 3.
            k : int
                order of piecewise polynomial. Default: 3.
            mult_arity : int, or list of int lists
                multiplication arity for each multiplication node (the number of numbers to be multiplied)
            noise_scale : float
                initial injected noise to spline.
            base_fun : str
                the residual function b(x). Default: 'silu'
            symbolic_enabled : bool
                compute (True) or skip (False) symbolic computations (for efficiency). By default: True. 
            affine_trainable : bool
                affine parameters are updated or not. Affine parameters include node_scale, node_bias, subnode_scale, subnode_bias
            grid_eps : float
                When grid_eps = 1, the grid is uniform; when grid_eps = 0, the grid is partitioned using percentiles of samples. 0 < grid_eps < 1 interpolates between the two extremes.
            grid_range : list/np.array of shape (2,))
                setting the range of grids. Default: [-1,1]. This argument is not important if fit(update_grid=True) (by default updata_grid=True)
            sp_trainable : bool
                If true, scale_sp is trainable. Default: True.
            sb_trainable : bool
                If true, scale_base is trainable. Default: True.
            device : str
                device
            seed : int
                random seed
            save_act : bool
                indicate whether intermediate activations are saved in forward pass
            sparse_init : bool
                sparse initialization (True) or normal dense initialization. Default: False.
            auto_save : bool
                indicate whether to automatically save a checkpoint once the model is modified
            state_id : int
                the state of the model (used to save checkpoint)
            ckpt_path : str
                the folder to store checkpoints. Default: './model'
            round : int
                the number of times rewind() has been called
            device : str
            
        Returns:
        --------
            self
            
        Example
        -------
        >>> from kan import *
        >>> model = KAN(width=[2,5,1], grid=5, k=3, seed=0)
        checkpoint directory created: ./model
        saving model version 0.0
        '''
        super(MultKAN, self).__init__()

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        ### initializeing the numerical front ###

        self.act_fun = []
        self.depth = len(width) - 1
        
        for i in range(len(width)):
            if type(width[i]) == int:
                width[i] = [width[i],0]
            
        self.width = width
        
        # if mult_arity is just a scalar, we extend it to a list of lists
        # e.g, mult_arity = [[2,3],[4]] means that in the first hidden layer, 2 mult ops have arity 2 and 3, respectively;
        # in the second hidden layer, 1 mult op has arity 4.
        if isinstance(mult_arity, int):
            self.mult_homo = True # when homo is True, parallelization is possible
        else:
            self.mult_homo = False # when home if False, for loop is required. 
        self.mult_arity = mult_arity

        width_in = self.width_in
        width_out = self.width_out
        
        self.base_fun_name = base_fun
        if base_fun == 'silu':
            base_fun = torch.nn.SiLU()
        elif base_fun == 'identity':
            base_fun = torch.nn.Identity()
        elif base_fun == 'zero':
            base_fun = lambda x: x*0.
            
        self.grid_eps = grid_eps
        self.grid_range = grid_range
            
        
        for l in range(self.depth):
            # splines
            sp_batch = KANLayer_cus(in_dim=width_in[l], out_dim=width_out[l+1], num=grid, k=k, 
                                    return_y=False, neuron_fun=neuron_fun, update_grid=update_grid)
            self.act_fun.append(sp_batch)

        self.node_bias = []
        self.node_scale = []
        self.subnode_bias = []
        self.subnode_scale = []
        
        globals()['self.node_bias_0'] = torch.nn.Parameter(torch.zeros(3,1)).requires_grad_(False)
        exec('self.node_bias_0' + " = torch.nn.Parameter(torch.zeros(3,1)).requires_grad_(False)")
        
        for l in range(self.depth):
            exec(f'self.node_bias_{l} = torch.nn.Parameter(torch.zeros(width_in[l+1])).requires_grad_(affine_trainable)')
            exec(f'self.node_scale_{l} = torch.nn.Parameter(torch.ones(width_in[l+1])).requires_grad_(affine_trainable)')
            exec(f'self.subnode_bias_{l} = torch.nn.Parameter(torch.zeros(width_out[l+1])).requires_grad_(affine_trainable)')
            exec(f'self.subnode_scale_{l} = torch.nn.Parameter(torch.ones(width_out[l+1])).requires_grad_(affine_trainable)')
            exec(f'self.node_bias.append(self.node_bias_{l})')
            exec(f'self.node_scale.append(self.node_scale_{l})')
            exec(f'self.subnode_bias.append(self.subnode_bias_{l})')
            exec(f'self.subnode_scale.append(self.subnode_scale_{l})')
            
        
        self.act_fun = nn.ModuleList(self.act_fun)

        self.grid = grid
        self.k = k
        self.base_fun = base_fun

        self.affine_trainable = affine_trainable
        self.sp_trainable = sp_trainable
        self.sb_trainable = sb_trainable
        
        self.save_act = save_act
            
        self.node_scores = None
        self.edge_scores = None
        self.subnode_scores = None
        
        self.cache_data = None
        self.acts = None
        
        self.auto_save = auto_save
        self.state_id = 0
        self.ckpt_path = ckpt_path
        self.round = round
        
        self.device = device
        self.to(device)
        
        if auto_save:
            if first_init:
                if not os.path.exists(ckpt_path):
                    # Create the directory
                    os.makedirs(ckpt_path)
                print(f"checkpoint directory created: {ckpt_path}")
                print('saving model version 0.0')

                history_path = self.ckpt_path+'/history.txt'
                with open(history_path, 'w') as file:
                    file.write(f'### Round {self.round} ###' + '\n')
                    file.write('init => 0.0' + '\n')
                # self.saveckpt(path=self.ckpt_path+'/'+'0.0')
            else:
                self.state_id = state_id
            
        self.input_id = torch.arange(self.width_in[0],)

        self.subnode_actscale = []
        self.edge_actscale = []
        
    def to(self, device):
        '''
        move the model to device
        
        Args:
        -----
            device : str or device

        Returns:
        --------
            self
            
        Example
        -------
        >>> from kan import *
        >>> device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        >>> model = KAN(width=[2,5,1], grid=5, k=3, seed=0)
        >>> model.to(device)
        '''
        super(MultKAN, self).to(device)
        self.device = device
        
        for kanlayer in self.act_fun:
            kanlayer.to(device)
            
        # for symbolic_kanlayer in self.symbolic_fun:
        #     symbolic_kanlayer.to(device)
            
        return self
    
    @property
    def width_in(self):
        '''
        The number of input nodes for each layer
        '''
        width = self.width
        width_in = [width[l][0]+width[l][1] for l in range(len(width))]
        return width_in
        
    @property
    def width_out(self):
        '''
        The number of output subnodes for each layer
        '''
        width = self.width
        if self.mult_homo == True:
            width_out = [width[l][0]+self.mult_arity*width[l][1] for l in range(len(width))]
        else:
            width_out = [width[l][0]+int(np.sum(self.mult_arity[l])) for l in range(len(width))]
        return width_out
    
    @property
    def n_sum(self):
        '''
        The number of addition nodes for each layer
        '''
        width = self.width
        n_sum = [width[l][0] for l in range(1,len(width)-1)]
        return n_sum
    
    def get_act(self, x=None):
        '''
        collect intermidate activations
        '''
        if isinstance(x, dict):
            x = x['train_input']
        if x == None:
            if self.cache_data != None:
                x = self.cache_data
            else:
                raise Exception("missing input data x")
        save_act = self.save_act
        self.save_act = True
        self.forward(x)
        self.save_act = save_act

    def update_grid_from_samples(self, x):
        '''
        update grid from samples
        
        Args:
        -----
            x : 2D torch.tensor
                inputs

        Returns:
        --------
            None
            
        Example
        -------
        >>> from kan import *
        >>> model = KAN(width=[1,1], grid=5, k=3, seed=0)
        >>> print(model.act_fun[0].grid)
        >>> x = torch.linspace(-10,10,steps=101)[:,None]
        >>> model.update_grid_from_samples(x)
        >>> print(model.act_fun[0].grid)
        ''' 
        for l in range(self.depth):
            self.get_act(x)
            self.act_fun[l].update_grid_from_samples(self.acts[l])
            
    def update_grid(self, x):
        '''
        call update_grid_from_samples. This seems unnecessary but we retain it for the sake of classes that might inherit from MultKAN
        '''
        self.update_grid_from_samples(x)

    def forward(self, x, singularity_avoiding=False, y_th=10.):
        '''
        forward pass
        
        Args:
        -----
            x : 2D torch.tensor
                inputs
            singularity_avoiding : bool
                whether to avoid singularity for the symbolic branch
            y_th : float
                the threshold for singularity

        Returns:
        --------
            None
            
        Example1
        --------
        >>> from kan import *
        >>> model = KAN(width=[2,5,1], grid=5, k=3, seed=0)
        >>> x = torch.rand(100,2)
        >>> model(x).shape
        
        Example2
        --------
        >>> from kan import *
        >>> model = KAN(width=[1,1], grid=5, k=3, seed=0)
        >>> x = torch.tensor([[1],[-0.01]])
        >>> model.fix_symbolic(0,0,0,'log',fit_params_bool=False)
        >>> print(model(x))
        >>> print(model(x, singularity_avoiding=True))
        >>> print(model(x, singularity_avoiding=True, y_th=1.))
        '''
        x = x[:,self.input_id.long()]
        assert x.shape[1] == self.width_in[0]
        
        # cache data
        self.cache_data = x
        
        self.acts = []  # shape ([batch, n0], [batch, n1], ..., [batch, n_L])
        # self.acts_premult = []
        # self.spline_preacts = []
        # self.spline_postsplines = []
        # self.spline_postacts = []
        # self.acts_scale = []
        # self.acts_scale_spline = []
        # self.subnode_actscale = []
        # self.edge_actscale = []
        # self.neurons_scale = []

        self.acts.append(x)  # acts shape: (batch, width[l])

        for l in range(self.depth):
            
            x_numerical, preacts, postacts_numerical, postspline = self.act_fun[l](x)

            x = x_numerical
            
            if self.save_act:
                # save subnode_scale
                self.subnode_actscale.append(torch.std(x, dim=0).detach())
                if len(self.subnode_actscale)>20:
                    self.subnode_actscale.pop(0)
            
            # subnode affine transform
            x = self.subnode_scale[l][None,:] * x + self.subnode_bias[l][None,:]
            
            if self.save_act:
                postacts = postacts_numerical

                # self.neurons_scale.append(torch.mean(torch.abs(x), dim=0))
                #grid_reshape = self.act_fun[l].grid.reshape(self.width_out[l + 1], self.width_in[l], -1)
                # input_range = torch.std(preacts, dim=0) + 0.1
                # output_range_spline = torch.std(postacts_numerical, dim=0) # for training, only penalize the spline part
                output_range = torch.std(postacts, dim=0) # for visualization, include the contribution from both spline + symbolic
                # save edge_scale
                # self.edge_actscale.append(output_range)
                
                # self.acts_scale.append((output_range / input_range).detach())
                # self.acts_scale_spline.append(output_range_spline / input_range)
                # self.spline_preacts.append(preacts.detach())
                # self.spline_postacts.append(postacts.detach())
                # self.spline_postsplines.append(postspline.detach())
                # self.acts_premult.append(x.detach())
            
            # multiplication
            dim_sum = self.width[l+1][0]
            dim_mult = self.width[l+1][1]
            
            if self.mult_homo == True:
                for i in range(self.mult_arity-1):
                    if i == 0:
                        x_mult = x[:,dim_sum::self.mult_arity] * x[:,dim_sum+1::self.mult_arity]
                    else:
                        x_mult = x_mult * x[:,dim_sum+i+1::self.mult_arity]
                        
            else:
                for j in range(dim_mult):
                    acml_id = dim_sum + np.sum(self.mult_arity[l+1][:j])
                    for i in range(self.mult_arity[l+1][j]-1):
                        if i == 0:
                            x_mult_j = x[:,[acml_id]] * x[:,[acml_id+1]]
                        else:
                            x_mult_j = x_mult_j * x[:,[acml_id+i+1]]
                            
                    if j == 0:
                        x_mult = x_mult_j
                    else:
                        x_mult = torch.cat([x_mult, x_mult_j], dim=1)
                
            if self.width[l+1][1] > 0:
                x = torch.cat([x[:,:dim_sum], x_mult], dim=1)
            
            # x = x + self.biases[l].weight
            # node affine transform
            x = self.node_scale[l][None,:] * x + self.node_bias[l][None,:]
            
            self.acts.append(x.detach())  
        return x
    
    def attribute(self, l=None, i=None, out_score=None, plot=True):
        '''
        get attribution scores

        Args:
        -----
            l : None or int
                layer index
            i : None or int
                neuron index
            out_score : None or 1D torch.float
                specify output scores
            plot : bool
                when plot = True, display the bar show
            
        Returns:
        --------
            attribution scores

        Example
        -------
        >>> from kan import *
        >>> model = KAN(width=[3,5,1], grid=5, k=3, noise_scale=0.3, seed=2)
        >>> f = lambda x: 1 * x[:,[0]]**2 + 0.3 * x[:,[1]]**2 + 0.0 * x[:,[2]]**2
        >>> dataset = create_dataset(f, n_var=3)
        >>> model.fit(dataset, opt='LBFGS', steps=20, lamb=0.001);
        >>> model.attribute()
        >>> model.feature_score
        '''
        # output (out_dim, in_dim)
        
        if l != None:
            self.attribute()
            out_score = self.node_scores[l]
       
        if self.acts == None:
            self.get_act()

        def score_node2subnode(node_score, width, mult_arity, out_dim):

            assert np.sum(width) == node_score.shape[1]
            if isinstance(mult_arity, int):
                n_subnode = width[0] + mult_arity * width[1]
            else:
                n_subnode = width[0] + int(np.sum(mult_arity))

            #subnode_score_leaf = torch.zeros(out_dim, n_subnode).requires_grad_(True)
            #subnode_score = subnode_score_leaf.clone()
            #subnode_score[:,:width[0]] = node_score[:,:width[0]]
            subnode_score = node_score[:,:width[0]]
            if isinstance(mult_arity, int):
                #subnode_score[:,width[0]:] = node_score[:,width[0]:][:,:,None].expand(out_dim, node_score[width[0]:].shape[0], mult_arity).reshape(out_dim,-1)
                subnode_score = torch.cat([subnode_score, node_score[:,width[0]:][:,:,None].expand(out_dim, node_score[:,width[0]:].shape[1], mult_arity).reshape(out_dim,-1)], dim=1)
            else:
                acml = width[0]
                for i in range(len(mult_arity)):
                    #subnode_score[:, acml:acml+mult_arity[i]] = node_score[:, width[0]+i]
                    subnode_score = torch.cat([subnode_score, node_score[:, width[0]+i].expand(out_dim, mult_arity[i])], dim=1)
                    acml += mult_arity[i]
            return subnode_score


        node_scores = []
        subnode_scores = []
        edge_scores = []
        
        l_query = l
        if l == None:
            l_end = self.depth
        else:
            l_end = l

        # back propagate from the queried layer
        out_dim = self.width_in[l_end]
        if out_score == None:
            node_score = torch.eye(out_dim).requires_grad_(True)
        else:
            node_score = torch.diag(out_score).requires_grad_(True)
        node_scores.append(node_score)
        
        device = self.act_fun[0].grid.device

        for l in range(l_end,0,-1):

            # node to subnode 
            if isinstance(self.mult_arity, int):
                subnode_score = score_node2subnode(node_score, self.width[l], self.mult_arity, out_dim=out_dim)
            else:
                mult_arity = self.mult_arity[l]
                #subnode_score = score_node2subnode(node_score, self.width[l], mult_arity)
                subnode_score = score_node2subnode(node_score, self.width[l], mult_arity, out_dim=out_dim)

            subnode_scores.append(subnode_score)
            # subnode to edge
            #print(self.edge_actscale[l-1].device, subnode_score.device, self.subnode_actscale[l-1].device)
            edge_score = torch.einsum('ij,ki,i->kij', self.edge_actscale[l-1], subnode_score.to(device), 1/(self.subnode_actscale[l-1]+1e-4))
            edge_scores.append(edge_score)

            # edge to node
            node_score = torch.sum(edge_score, dim=1)
            node_scores.append(node_score)

        self.node_scores_all = list(reversed(node_scores))
        self.edge_scores_all = list(reversed(edge_scores))
        self.subnode_scores_all = list(reversed(subnode_scores))

        self.node_scores = [torch.mean(l, dim=0) for l in self.node_scores_all]
        self.edge_scores = [torch.mean(l, dim=0) for l in self.edge_scores_all]
        self.subnode_scores = [torch.mean(l, dim=0) for l in self.subnode_scores_all]

        return self.subnode_scores[-1]
        # return self.subnode_scores_all[-1]
        # # return
        # if l_query != None:
        #     if i == None:
        #         return self.node_scores_all[0]
        #     else:
        #         return self.node_scores_all[0][i]


KAN = MultKAN