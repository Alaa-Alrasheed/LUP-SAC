
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
    parser.add_argument('--lr', type=float, default=0.01,
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
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--device', default='cpu', help="To use cuda, set to a specific GPU ID. Default set to use CPU.")#cuda:0
    parser.add_argument('--optimizer', type=str, default='sgd', help="type of optimizer")
    parser.add_argument('--iid', type=int, default=0,
                        help='Default set to IID. Set to 0 for non-IID.')
    parser.add_argument('--skew', type=float, default=0.5,
                        help='Default set to IID. Set to 0 for non-IID.')
    # parser.add_argument('--seed', type=int, default=1, help='random seed')
    args = parser.parse_args()
    return args
