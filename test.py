from utils.dataset import *
from utils.args import process_args
from models.TKG_VRE import TKG_VAE
from baselines.Static import Static
from baselines.Simple import SimplE
from baselines.Hyte import Hyte
from baselines.DiachronicEmbedding import DiachronicEmbedding
from baselines.StaticRGCN import StaticRGCN
from baselines.RecurrentRGCN import RecurrentRGCN
from baselines.DRGCN import DRGCN
import glob
import json
from pytorch_lightning import Trainer

if __name__ == '__main__':
    args = process_args()
    torch.manual_seed(args.seed)

    experiment_path = args.checkpoint_path
    checkpoint_path = glob.glob(os.path.join(experiment_path, "checkpoints", "*.ckpt"))[0]
    config_path = os.path.join(experiment_path, "config.json")
    args_json = json.load(open(config_path))
    args.__dict__.update(dict(args_json))

    print(checkpoint_path)
    # print(config_path)
    args.use_VAE = False
    use_cuda = args.use_cuda = args.n_gpu >= 0 and torch.cuda.is_available()
    num_ents, num_rels = get_total_number(args.dataset, 'stat.txt')

    graph_dict_train, graph_dict_val, graph_dict_test = build_interpolation_graphs(args)

    module = {
              'VKGRNN': TKG_VAE,
              "Simple": SimplE,
              "Static": Static,
              "DE": DiachronicEmbedding,
              "Hyte": Hyte,
              "SRGCN": StaticRGCN,
              "RRGCN": RecurrentRGCN,
              "DRGCN": DRGCN
              }[args.module]

    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    model = module(args, num_ents, num_rels, graph_dict_train, graph_dict_val, graph_dict_test)
    model.load_state_dict(checkpoint['state_dict'])

    # give model a chance to load something
    model.on_load_checkpoint(checkpoint)
    trainer = Trainer(gpus=0 if not use_cuda else 1)
    trainer.test(model)
    # trainer.run_evaluation(test=True)