import argparse
import os


def get_args():
    parser = argparse.ArgumentParser(description='RENet')
    parser.add_argument("--dataset_dir", type=str, default='data')
    parser.add_argument("-d", "--dataset", type=str, default='GDELT')
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--hidden_size", type=int, default=100)
    parser.add_argument("--embed_size", type=int, default=100)
    parser.add_argument("--num-k", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=1000)
    # LSTM parameters
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--grad-norm", type=float, default=1.0)
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--n-bases", type=int, default=50,
                       help="number of weight blocks for each relation")
    parser.add_argument("--rgcn-layers", type=int, default=2,
                        help="number of propagation rounds")
    parser.add_argument("--seq-len", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--negative-rate", type=int, default=10)
    parser.add_argument("--graph-batch-size", type=int, default=3000, help="number of edges to sample in each iteration")
    return parser.parse_args()


def process_args():
    args = get_args()
    args.dataset = os.path.join(args.dataset_dir, args.dataset)
    return args