
import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    # federated arguments
    parser.add_argument('--epochs', type=int, default=100,
                        help="number of training epochs")
    parser.add_argument('--num_users', type=int, default=50,
                        help="number of users: n")
    parser.add_argument('--frac', type=float, default=1.0,
                        help='the fraction of clients: C')
    parser.add_argument('--local_iter', type=int, default=1,
                        help="the number of local iterations: E")
    parser.add_argument('--local_bs', type=int, default=256,
                        help="local batch size: b")
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum')

    # byzantine arguments
    parser.add_argument('--num_byzs', type=int, default=25,
                        help='number of byzantine nodes: m')
    parser.add_argument('--agg_rule', type=str, default='Mean',
                        help='the gradient aggregation rule')
    parser.add_argument('--attack', type=str, nargs='*', default=[],
                        help='the byzantine attack method(s). If none specified, runs all.')

    # other arguments
    parser.add_argument('--dataset', type=str, default='mnist',
                        help="name of dataset")
    parser.add_argument('--model', type=str, default='cnn',
                        help="name of model (e.g. cnn, resnet9, densenet121)")
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--device', default='cpu', help="To use cuda, set to a specific GPU ID. Default set to use CPU.")#cuda:0
    parser.add_argument('--optimizer', type=str, default='sgd', help="type of optimizer")
    parser.add_argument('--iid', type=int, default=0,
                        help='Default set to IID. Set to 0 for non-IID.')
    parser.add_argument('--skew', type=float, default=0.5,
                        help='Default set to IID. Set to 0 for non-IID.')
    parser.add_argument('--seed', type=int, default=1, help='random seed')

    # ── Semantic Distribution Analysis ──
    parser.add_argument('--semantic', action='store_true', default=True,
                        help='Enable semantic distribution analysis (default: True)')
    parser.add_argument('--no-semantic', dest='semantic', action='store_false',
                        help='Disable semantic analysis (zero overhead short-circuit)')
    parser.add_argument('--gi_iterations', type=int, default=-1,
                        help='Gradient inversion iterations. -1 = auto-scale by device '
                             '(GPU: 30, CPU: 15)')
    parser.add_argument('--gi_batch_size', type=int, default=8,
                        help='Dummy batch size for gradient inversion')
    parser.add_argument('--semantic_sample_ratio', type=float, default=0.3,
                        help='Fraction of clients analyzed per round (0.3 = 30%%)')
    parser.add_argument('--semantic_decay', type=float, default=0.9,
                        help='Temporal decay factor for unsampled client scores')
    
    parser.add_argument('--semantic_veto_threshold', type=float, default=-0.015,
                        help='The MMD penalty threshold that triggers a hard reward override.')
    parser.add_argument('--semantic_penalty_value', type=float, default=-1.0,
                        help='The hard negative reward applied when the veto fires.')

    parser.add_argument('--direction_veto_threshold', type=float, default=0.0,
                        help='Cosine similarity to EMA below this is flagged as directional risk '
                             '(logged in gate_info["direction"]). Diagnostic-only: does NOT '
                             'currently override alpha — only semantic check enforces.')
    parser.add_argument('--mag_veto_factor', type=float, default=2.0,
                        help='Cluster magnitude / EMA magnitude ratio above this is flagged as '
                             'magnitude risk (logged in gate_info["magnitude"]). Diagnostic-only: '
                             'does NOT currently override alpha — only semantic check enforces.')

    args = parser.parse_args()
    return args
