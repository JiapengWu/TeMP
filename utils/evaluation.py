import torch
from utils.CorrptTriples import CorruptTriples


class EvaluationFilter:
    def __init__(self, args, calc_score, graph_dict_train, graph_dict_val, graph_dict_test):
        self.args = args
        self.calc_score = calc_score
        self.graph_dict_train = graph_dict_train
        self.graph_dict_val = graph_dict_val
        self.graph_dict_test = graph_dict_test
        self.get_true_head_and_tail_all()

    def get_true_head_and_tail_all(self):
        self.true_heads = dict()
        self.true_tails = dict()
        times = list(self.graph_dict_train.keys())
        for t in times:
            triples = []
            for g in self.graph_dict_train[t], self.graph_dict_val[t], self.graph_dict_test[t]:
                triples.append(torch.stack([g.edges()[0], g.edata['type_s'], g.edges()[1]]).transpose(0, 1))
            triples = torch.cat(triples, dim=0)
            true_head, true_tail = CorruptTriples.get_true_head_and_tail_per_graph(triples)
            self.true_heads[t] = true_head
            self.true_tails[t] = true_tail

    def calc_metrics_single_graph(self, ent_mean, rel_enc_means, test_triplets, time, eval_bz=100):
        with torch.no_grad():
            s = test_triplets[:, 0]
            r = test_triplets[:, 1]
            o = test_triplets[:, 2]
            test_size = test_triplets.shape[0]
            num_ent = ent_mean.shape[0]
            o_mask = self.mask_eval_set(test_triplets, test_size, num_ent, time, mode="tail")
            s_mask = self.mask_eval_set(test_triplets, test_size, num_ent, time, mode="head")
            # perturb object
            ranks_o = self.perturb_and_get_rank(ent_mean, rel_enc_means, s, r, o, test_size, o_mask, eval_bz, mode='tail')
            # perturb subject
            ranks_s = self.perturb_and_get_rank(ent_mean, rel_enc_means, s, r, o, test_size, s_mask, eval_bz, mode='head')
            ranks = torch.cat([ranks_s, ranks_o])
            ranks += 1 # change to 1-indexed

            mrr = torch.mean(1.0 / ranks.float()).item()
            hit_1 = torch.mean((ranks <= 1).float()).item()
            hit_3 = torch.mean((ranks <= 3).float()).item()
            hit_10 = torch.mean((ranks <= 10).float()).item()

        return mrr, hit_1, hit_3, hit_10

    def perturb_and_get_rank(self, ent_mean, rel_enc_means, s, r, o, test_size, mask, batch_size=100, mode ='tail'):
        """ Perturb one element in the triplets
        """
        n_batch = (test_size + batch_size - 1) // batch_size
        ranks = []
        for idx in range(n_batch):
            batch_start = idx * batch_size
            batch_end = min(test_size, (idx + 1) * batch_size)
            batch_r = rel_enc_means[r[batch_start: batch_end]]

            if mode == 'tail':
                batch_s = ent_mean[s[batch_start: batch_end]]
                batch_o = ent_mean
                target = o[batch_start: batch_end]
            else:
                batch_s = ent_mean
                batch_o = ent_mean[o[batch_start: batch_end]]
                target = s[batch_start: batch_end]

            unmasked_score = self.calc_score(batch_s, batch_r, batch_o, mode=mode)
            # import pdb; pdb.set_trace()
            masked_score = torch.where(mask[batch_start: batch_end], -10e6 * unmasked_score.new_ones(unmasked_score.shape), unmasked_score)
            score = torch.sigmoid(masked_score)  # bsz, n_ent

            ranks.append(self.sort_and_rank(score, target))
        return torch.cat(ranks)

    def mask_eval_set(self, test_triplets, test_size, num_ent, time, mode='tail'):
        time = time.item()
        mask = test_triplets.new_zeros(test_size, num_ent)
        for i in range(test_size):
            h, r, t = test_triplets[i]
            h, r, t = h.item(), r.item(), t.item()
            if mode == 'tail':
                tails = self.true_tails[time][(h, r)]
                mask[i][tails] = 1
                mask[i][t] = 0
            elif mode == 'head':
                heads = self.true_heads[time][(r, t)]
                mask[i][heads] = 1
                mask[i][h] = 0
        return mask.byte()

    def sort_and_rank(self, score, target):
        _, indices = torch.sort(score, dim=1, descending=True)
        indices = torch.nonzero(indices == target.view(-1, 1))
        indices = indices[:, 1].view(-1)
        return indices
