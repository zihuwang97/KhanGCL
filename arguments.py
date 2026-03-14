import argparse

def arg_parse():
    parser = argparse.ArgumentParser(description='KHAN-GCL')
    parser.add_argument('--DS', dest='DS', type=str, default='MUTAG', help='Dataset')
    parser.add_argument('--local', dest='local', action='store_const', const=True, default=False)
    parser.add_argument('--glob', dest='glob', action='store_const', const=True, default=False)
    parser.add_argument('--prior', dest='prior', action='store_const', const=True, default=False)

    parser.add_argument('--lr', dest='lr', type=float, default=0.01, help='Learning rate.')
    parser.add_argument('--num-gc-layers', dest='num_gc_layers', type=int, default=3,
                        help='Number of graph convolution layers before each pooling')

    parser.add_argument('--hidden-dim', dest='hidden_dim', type=int, default=32, help='hidden dimension')

    parser.add_argument('--aug', type=str, default='drop_ra')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--log_interval', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--rho', type=float, default=0.9)
    parser.add_argument('--num_workers', type=int, default=8)

    parser.add_argument('--pooling', type=str, default='all')
    parser.add_argument('--log', type=str, default='full')
    parser.add_argument('--beta', type=float, default=0.6)
    parser.add_argument('--ema_decay', type=float, default=0.8)

    # KAN related configs
    parser.add_argument('--kan_mlp', type = str, default='kan', help="mlp or kan")
    parser.add_argument('--kan_mp', type = str, default='none', help="kan or none")
    parser.add_argument('--kan_type1', type = str, default='ori', help="ori, bsrbf")
    parser.add_argument('--kan_type2', type = str, default='eff', help="ori, bsrbf")
    parser.add_argument('--grid', type = int, default = 5, help="bspline grid")
    parser.add_argument('--k', type = int, default = 3, help="bspline order")
    parser.add_argument('--neuron_fun', type = str, default = 'sum', help="kan's neuron_fun, in mean or sum")
    parser.add_argument('--use_transformer', type = str, default = 'mlp' , help="Use transformer: none, mlp or kan")
    parser.add_argument('--kan_mlp_proj', type = str, default = 'mlp' , help="Proj head: mlp or kan")
    parser.add_argument('--kan_pred_type', type = str, default='mlp', help="ori, bsrbf")
    parser.add_argument('--grid_pred', type = int, default = 5, help="bspline grid")
    parser.add_argument('--k_pred', type = int, default = 3, help="bspline order")
    parser.add_argument('--neuron_fun_pred', type = str, default = 'sum', help="kan's neuron_fun, in mean or sum")

    parser.add_argument('--saliency_method', type = str, default = ['hosvd', 'coef_var'], nargs="+" , help="std, coef_var, coef_corr, hosvd, or random")
    parser.add_argument('--hosvd_update_freq', type = int, default = 10 , help="Coefficient HOSVD based saliency update frequency. Ignored when not using HOSVD")

    parser.add_argument('--eval_which', type = str, default='encoder1', help="Which encoder to evaluate. Encoder1 or Encoder2")
    parser.add_argument('--model_save_path', type = str, default='None', help="model save path. If 'None', no savings")
    parser.add_argument('--update_grid', action='store_true', help="whether to update grid. Default: not")
    parser.add_argument('--update_grid_freq', type = int, default = 10 , help="grid update frequency")
    return parser.parse_args()

