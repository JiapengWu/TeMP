# from comet_ml import Experiment, ExistingExperiment
from utils.dataset import *
from utils.args import process_args
import pdb
from models.TKG_VRE import VKG_VRE
from sklearn.utils import shuffle
from utils.utils import make_batch


def train_epoch():
    model.train()
    train_times_shuffled = shuffle(train_times)
    epoch_loss = []
    for batch_time in make_batch(train_times_shuffled, args.batch_size):
        batch_time = torch.from_numpy(batch_time)

        # model(batch_time, train_graph_dict)
        loss = model(batch_time, train_graph_dict)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)  # clip gradients
        optimizer.step()
        epoch_loss.append(loss.item())
        print("Batch loss: {}\r".format(loss.item()), end='\r')


if __name__ == '__main__':
    args = process_args()
    torch.manual_seed(args.seed)
    use_cuda = args.use_cuda = args.gpu >= 0 and torch.cuda.is_available()

    if use_cuda:
        print("Using GPU")
        torch.cuda.set_device(args.gpu)
    else:
        print("Using CPU")

    num_ents, num_rels = get_total_number(args.dataset, 'stat.txt')
    if args.dataset == 'ICEWS14':
        train_data, train_times = load_quadruples(args.dataset, 'train.txt')
        valid_data, valid_times = load_quadruples(args.dataset, 'test.txt')
        test_data, test_times = load_quadruples(args.dataset, 'test.txt')
        total_data, total_times = load_quadruples(args.dataset, 'train.txt', 'test.txt')
    else:
        train_data, train_times = load_quadruples(args.dataset, 'train.txt')
        valid_data, valid_times = load_quadruples(args.dataset, 'valid.txt')
        test_data, test_times = load_quadruples(args.dataset, 'test.txt')
        total_data, total_times = load_quadruples(args.dataset, 'train.txt', 'valid.txt','test.txt')

    model = VKG_VRE(args, num_ents, num_rels)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0001)
    train_graph_dict = build_time_stamp_graph(args)

    if use_cuda:
        model.cuda()
    for i in range(args.epochs):
        train_epoch()
