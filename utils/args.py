import argparse
import os


def get_args():
    parser = argparse.ArgumentParser(description='TKG-VAE')
    parser.add_argument("--dataset-dir", type=str, default='interpolation')
    parser.add_argument("-d", "--dataset", type=str, default='icews14')
    parser.add_argument("--score-function", type=str, default='complex')
    parser.add_argument("--module", type=str, default='GRRGCN')
    parser.add_argument("--n-gpu", type=int, default=0)
    parser.add_argument("--distributed_backend", type=str, default='ddp')
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--embed_size", type=int, default=128)
    parser.add_argument("--max-nb-epochs", type=int, default=1000)

    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--rate-lower", type=float, default=0.2)
    parser.add_argument("--rate-upper", type=float, default=0.8)
    parser.add_argument("--lambda-1", type=float, default=2)
    parser.add_argument("--lambda-2", type=float, default=10)
    parser.add_argument("--lambda-3", type=float, default=20)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gradient-clip-val", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--n-bases", type=int, default=128, help="number of weight blocks for each relation")
    parser.add_argument("--rgcn-layers", type=int, default=2, help="number of propagation rounds")
    parser.add_argument("--train-seq-len", type=int, default=15)
    parser.add_argument("--test-seq-len", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--negative-rate", type=int, default=100) # different
    parser.add_argument("--num-pos-facts", type=int, default=3000, help="number of edges to sample in each iteration")
    parser.add_argument('--log-gpu-memory', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--rec-only-last-layer', action='store_true')
    parser.add_argument('--fast-dev-run', action='store_true')
    parser.add_argument('--use-time-embedding', action='store_true')
    parser.add_argument("--inv-temperature", type=float, default=0.1)
    parser.add_argument('--use-embed-for-non-active', action='store_true')
    parser.add_argument("--edge-dropout", action='store_true')
    parser.add_argument("--random-dropout", action='store_true')
    parser.add_argument("--type1", action='store_true')
    parser.add_argument("--post-ensemble", action='store_true')
    parser.add_argument("--post-aggregation", action='store_true')
    parser.add_argument("--learnable-lambda", action='store_true')
    parser.add_argument("--impute", action='store_true')
    parser.add_argument("--EMA", action='store_true')

    parser.add_argument("--vote", type=str, default='recency')
    parser.add_argument("--future", action='store_true')
    parser.add_argument("--filtered", action='store_true')
    parser.add_argument("--all", action='store_true')
    parser.add_argument("--resume", action='store_true')
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--version", type=str, default=None)

    parser.add_argument('--config', '-c', type=str, default=None, help='JSON file with argument for the run.')
    parser.add_argument("--checkpoint-path", type=str, default=None)

    parser.add_argument("--spatial-checkpoint", type=str, default=None)
    parser.add_argument("--temporal-checkpoint", type=str, default=None)
    parser.add_argument("--temporal-module", type=str, default="BiGRRGCN")

    return parser.parse_args()


def process_args():
    args = get_args()
    args.dataset = os.path.join(args.dataset_dir, args.dataset)
    return args