from torch import nn
from models.TKG_Module import TKG_Module
import torch
from utils.DropEdge import DropEdge

from utils.dataset import *
from baselines.StaticRGCN import StaticRGCN
from models.DynamicRGCN import DynamicRGCN
import glob
import json
from models.BiDynamicRGCN import BiDynamicRGCN
from models.SelfAttentionRGCN import SelfAttentionRGCN
from models.BiSelfAttentionRGCN import BiSelfAttentionRGCN
from utils.utils import cuda
from utils.CorrptTriples import CorruptTriples
import torch.nn.functional as F
from models.PostDynamicRGCN import ImputeDynamicRGCN, PostDynamicRGCN, PostEnsembleDynamicRGCN
from models.PostBiDynamicRGCN import ImputeBiDynamicRGCN, PostBiDynamicRGCN, PostEnsembleBiDynamicRGCN
from models.PostSelfAttentionRGCN import PostBiSelfAttentionRGCN


class Aggregator(TKG_Module):
    def __init__(self, args, num_ents, num_rels, graph_dict_train, graph_dict_val, graph_dict_test):

        super(Aggregator, self).__init__(args, num_ents, num_rels, graph_dict_train, graph_dict_val, graph_dict_test)
        module = {
            "GRRGCN": DynamicRGCN,
            "RRGCN": DynamicRGCN,
            "SARGCN": SelfAttentionRGCN,
            "BiGRRGCN": BiDynamicRGCN,
            "BiRRGCN": BiDynamicRGCN,
            "BiSARGCN": BiSelfAttentionRGCN
        }[args.temporal_module]

        # import pdb; pdb.set_trace()

        self.graph_dict_total = {**self.graph_dict_train, **self.graph_dict_val, **self.graph_dict_test}

        self.get_true_head_and_tail_all()

        spatial_path = args.spatial_checkpoint
        temporal_path = args.temporal_checkpoint

        if args.debug:
            self.bidirectional = True
            self.drop_edge = DropEdge(args, graph_dict_train, graph_dict_val, graph_dict_test)
            # self.drop_edge.count_frequency()
            self.spatial_model = StaticRGCN(args, num_ents, num_rels, graph_dict_train, graph_dict_val, graph_dict_test)
            args.module = 'BiGRRGCN'
            self.temporal_model = BiDynamicRGCN(args, num_ents, num_rels, graph_dict_train, graph_dict_val, graph_dict_test)

            self.train_seq_len = args.train_seq_len
            self.test_seq_len = args.train_seq_len
        else:
            checkpoint_path = glob.glob(os.path.join(temporal_path, "checkpoints", "*.ckpt"))[0]
            temporal_checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
            temporal_config_path = os.path.join(temporal_path, "config.json")
            temporal_args_json = json.load(open(temporal_config_path))
            temporal_args = process_args()
            temporal_args.__dict__.update(dict(temporal_args_json))
            self.drop_edge = DropEdge(temporal_args, graph_dict_train, graph_dict_val, graph_dict_test)
            # self.drop_edge.count_frequency()

            if module == BiDynamicRGCN:
                if temporal_args.post_aggregation:
                    module = PostBiDynamicRGCN
                elif temporal_args.post_ensemble:
                    module = PostEnsembleBiDynamicRGCN
                elif temporal_args.impute:
                    module = ImputeBiDynamicRGCN


            elif module == DynamicRGCN:
                if temporal_args.post_aggregation:
                    module = PostDynamicRGCN
                if temporal_args.post_ensemble:
                    module = PostEnsembleDynamicRGCN
                elif temporal_args.impute:
                    module = ImputeDynamicRGCN

            elif module == BiSelfAttentionRGCN:
                if temporal_args.post_aggregation:
                    module = PostBiSelfAttentionRGCN

            self.temporal_model = module(temporal_args, num_ents, num_rels, graph_dict_train, graph_dict_val, graph_dict_test)
            self.temporal_model.load_state_dict(temporal_checkpoint['state_dict'])

            self.train_seq_len = temporal_args.train_seq_len
            self.test_seq_len = temporal_args.train_seq_len
            self.bidirectional = "Bi" in temporal_args.module

            checkpoint_path = glob.glob(os.path.join(spatial_path, "checkpoints", "*.ckpt"))[0]
            local_checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
            local_config_path = os.path.join(spatial_path, "config.json")
            local_args_json = json.load(open(local_config_path))
            local_args = process_args()
            local_args.__dict__.update(dict(local_args_json))
            self.spatial_model = StaticRGCN(local_args, num_ents, num_rels, graph_dict_train, graph_dict_val, graph_dict_test)
            self.spatial_model.load_state_dict(local_checkpoint['state_dict'])

        for para in self.spatial_model.parameters():
            para.requires_grad = False
        for para in self.temporal_model.parameters():
            para.requires_grad = False

        # self.subject_linear = torch.nn.Linear(2, 1)
        # self.object_linear = torch.nn.Linear(2, 1)

        self.subject_linear = nn.Sequential(
            nn.Linear(3, 3),
            nn.ReLU(),
            nn.Linear(3, 1)
        )
        self.object_linear = nn.Sequential(
            nn.Linear(3, 3),
            nn.ReLU(),
            nn.Linear(3, 1)
        )

    def build_model(self):
        pass

    def get_true_head_and_tail_all(self):
        self.true_heads = dict()
        self.true_tails = dict()
        for t in self.total_time:
            triples = []
            for g in self.graph_dict_train[t], self.graph_dict_val[t], self.graph_dict_test[t]:
                triples.append(torch.stack([g.edges()[0], g.edata['type_s'], g.edges()[1]]).transpose(0, 1))
            triples = torch.cat(triples, dim=0)
            true_head, true_tail = CorruptTriples.get_true_head_and_tail_per_graph(triples)
            self.true_heads[t] = true_head
            self.true_tails[t] = true_tail

    def calc_ensemble_ratio(self, triples, t, g):
        sub_feature_vecs = []
        obj_feature_vecs = []
        t = t.item()
        for s, r, o in triples:

            s = g.ids[s.item()]
            r = r.item()
            o = g.ids[o.item()]
            # triple_freq = self.drop_edge.triple_freq_per_time_step_agg[t][(s, r, o)]
            # ent_pair_freq = self.drop_edge.ent_pair_freq_per_time_step_agg[t][(s, o)]
            sub_freq = self.drop_edge.sub_freq_per_time_step_agg[t][s]
            obj_freq = self.drop_edge.obj_freq_per_time_step_agg[t][o]
            rel_freq = self.drop_edge.rel_freq_per_time_step_agg[t][r]
            sub_rel_freq = self.drop_edge.sub_rel_freq_per_time_step_agg[t][(s, r)]
            obj_rel_freq = self.drop_edge.obj_rel_freq_per_time_step_agg[t][(o, r)]
            # 0: no local, 1: no temporal

            sub_feature_vecs.append(torch.tensor([obj_freq, rel_freq, obj_rel_freq]))
            obj_feature_vecs.append(torch.tensor([sub_freq, rel_freq, sub_rel_freq]))
        # pdb.set_trace()

        try:
            sub_features = torch.stack(sub_feature_vecs).float()
            obj_features = torch.stack(obj_feature_vecs).float()
            if self.use_cuda:
                sub_features = cuda(sub_features)
                obj_features = cuda(obj_features)
            weight_subject = torch.sigmoid(self.subject_linear(sub_features))
            weight_object = torch.sigmoid(self.object_linear(obj_features))
        except:
            weight_subject = cuda(torch.tensor([]).long()) if self.use_cuda else torch.tensor([]).long()
            weight_object = cuda(torch.tensor([]).long()) if self.use_cuda else torch.tensor([]).long()

        return weight_subject, weight_object

    def forward(self, t_list, reverse=False):
        # pdb.set_trace()
        t_list = t_list.sort(descending=True)[0]
        per_graph_ent_embeds_local, g_list = self.spatial_model.train_embed(t_list)
        if self.bidirectional:
            per_graph_ent_embeds_temporal, train_graphs, time_list, hist_embeddings_forward, start_time_tensor_forward, hist_embeddings_backward, start_time_tensor_backward = self.temporal_model.train_embed(t_list)
        else:
            per_graph_ent_embeds_temporal, train_graphs, time_list, hist_embeddings, start_time_tensor = self.temporal_model.train_embed(t_list)

        assert t_list.tolist() == time_list
        rel_enc_local = self.spatial_model.rel_embeds
        rel_enc_temp = self.temporal_model.rel_embeds
        reconstruct_loss = 0
        i = 0
        for t, g, ent_embed_local, ent_embed_temp in zip(t_list, train_graphs, per_graph_ent_embeds_local, per_graph_ent_embeds_temporal):
            triplets, neg_tail_samples, neg_head_samples, labels = self.corrupter.single_graph_negative_sampling(t, g, self.num_ents)
            # with torch.no_grad():
            if self.bidirectional:
                time_diff_tensor_forward = self.train_seq_len - 1 - start_time_tensor_forward[i]
                time_diff_tensor_backward = self.train_seq_len - 1 - start_time_tensor_backward[i]
                all_embeds_g_temp = self.temporal_model.get_all_embeds_Gt(ent_embed_temp, g, t, hist_embeddings_forward[i][0], hist_embeddings_forward[i][1], time_diff_tensor_forward,
                                                                            hist_embeddings_backward[i][0], hist_embeddings_backward[i][1], time_diff_tensor_backward)
            else:
                time_diff_tensor = self.train_seq_len - 1 - start_time_tensor[i]
                all_embeds_g_temp = self.temporal_model.get_all_embeds_Gt(ent_embed_temp, g, t, hist_embeddings[i][0], hist_embeddings[i][1], time_diff_tensor)

            all_embeds_g_local = self.spatial_model.get_all_embeds_Gt(t, g, ent_embed_local)

            weight_subject, weight_object = self.calc_ensemble_ratio(triplets, t, g)
            score_tail_local = self.train_link_prediction(ent_embed_local, rel_enc_local, triplets, neg_tail_samples, labels, all_embeds_g_local, corrupt_tail=True)
            score_head_local = self.train_link_prediction(ent_embed_local, rel_enc_local, triplets, neg_head_samples, labels, all_embeds_g_local, corrupt_tail=False)
            score_tail_temporal = self.train_link_prediction(ent_embed_temp, rel_enc_temp, triplets, neg_tail_samples, labels, all_embeds_g_temp, corrupt_tail=True)
            score_head_temporal = self.train_link_prediction(ent_embed_temp, rel_enc_temp, triplets, neg_head_samples, labels, all_embeds_g_temp, corrupt_tail=False)
            # pdb.set_trace()
            loss_tail = self.combined_scores(score_tail_local, score_tail_temporal, labels, weight_object)
            loss_head = self.combined_scores(score_head_local, score_head_temporal, labels, weight_subject)
            reconstruct_loss += loss_tail + loss_head
            i += 1
        return reconstruct_loss

    def evaluate(self, t_list, val=True):
        t_list = t_list.sort(descending=True)[0]
        per_graph_ent_embeds_local, g_list = self.spatial_model.evaluate_embed(t_list, val)
        if self.bidirectional:
            per_graph_ent_embeds_temporal, test_graphs, time_list, hist_embeddings_forward, start_time_tensor_forward, hist_embeddings_backward, start_time_tensor_backward = self.temporal_model.evaluate_embed(t_list, val)
            # if self.drop_edge
            #     per_graph_ent_embeds_temporal, test_graphs, time_list, hist_embeddings_loc, hist_embeddings_rec, start_time_tensor
        else:
            per_graph_ent_embeds_temporal, test_graphs, time_list, hist_embeddings, start_time_tensor = self.temporal_model.evaluate_embed(t_list, val)
        # assert t_list.tolist() == time_list
        mrrs, hit_1s, hit_3s, hit_10s, losses = [], [], [], [], []
        ranks = []
        i = 0
        cur_t = self.test_seq_len - 1
        rel_enc_local = self.spatial_model.rel_embeds
        rel_enc_temporal = self.temporal_model.rel_embeds

        for g, t, ent_embed_local, ent_embed_temporal in zip(g_list, t_list, per_graph_ent_embeds_local, per_graph_ent_embeds_temporal):
            # pdb.set_trace()
            all_embeds_g_local = self.spatial_model.get_all_embeds_Gt(t, g, ent_embed_local)
            if self.bidirectional:
                time_diff_tensor_forward = cur_t - start_time_tensor_forward[i]
                time_diff_tensor_backward = cur_t - start_time_tensor_backward[i]
                all_embeds_g_temporal = self.temporal_model.get_all_embeds_Gt(ent_embed_temporal, g, t, hist_embeddings_forward[i][0], hist_embeddings_forward[i][1], time_diff_tensor_forward,
                                                  hist_embeddings_backward[i][0], hist_embeddings_backward[i][1], time_diff_tensor_backward)
            else:
                time_diff_tensor = cur_t - start_time_tensor[i]
                all_embeds_g_temporal = self.temporal_model.get_all_embeds_Gt(ent_embed_temporal, g, t, hist_embeddings[i][0], hist_embeddings[i][1], time_diff_tensor)
            index_sample = torch.stack([g.edges()[0], g.edata['type_s'], g.edges()[1]]).transpose(0, 1)

            weight_subject, weight_object = self.calc_ensemble_ratio(index_sample, t, g)

            if self.use_cuda:
                index_sample = cuda(index_sample)
            if index_sample.shape[0] == 0: continue

            rank = self.calc_metrics_single_graph(ent_embed_local, ent_embed_temporal, rel_enc_local, rel_enc_temporal, all_embeds_g_local, all_embeds_g_temporal, weight_subject, weight_object, index_sample, g, t)
            # loss = self.link_classification_loss(ent_embed, self.rel_embeds, index_sample, label)
            ranks.append(rank)
            # losses.append(loss.item())
            i += 1
        try:
            ranks = torch.cat(ranks)
        except:
            ranks = cuda(torch.tensor([]).long()) if self.use_cuda else torch.tensor([]).long()

        return ranks, np.mean(losses)

    def calc_metrics_single_graph(self, ent_embed_local, ent_embed_temporal, rel_enc_local, rel_enc_temporal, all_embeds_g_local, all_embeds_g_temporal, weight_subject, weight_object, samples, graph, time, eval_bz=100):
        with torch.no_grad():
            s = samples[:, 0]
            r = samples[:, 1]
            o = samples[:, 2]
            test_size = samples.shape[0]
            num_ent = all_embeds_g_local.shape[0]
            o_mask = self.mask_eval_set(samples, test_size, num_ent, time, graph, mode="tail")
            s_mask = self.mask_eval_set(samples, test_size, num_ent, time, graph, mode="head")

            # perturb object
            ranks_o = self.emsemble_and_get_rank(ent_embed_local, ent_embed_temporal, rel_enc_local, rel_enc_temporal, all_embeds_g_local, all_embeds_g_temporal, weight_subject, s, r, o, test_size, o_mask, graph, eval_bz, mode='tail')
            # perturb subject
            ranks_s = self.emsemble_and_get_rank(ent_embed_local, ent_embed_temporal, rel_enc_local, rel_enc_temporal, all_embeds_g_local, all_embeds_g_temporal, weight_object, s, r, o, test_size, s_mask, graph, eval_bz, mode='head')

            ranks = torch.cat([ranks_s, ranks_o])
            ranks += 1 # change to 1-indexed
            # print("Graph {} mean ranks {}".format(time.item(), ranks.float().mean().item()))
        return ranks

    def emsemble_and_get_rank(self, ent_embed_local, ent_embed_temporal, rel_enc_local, rel_enc_temporal, all_embeds_g_local, all_embeds_g_temporal, weight, s, r, o, test_size, mask, graph, batch_size=100, mode ='tail'):
        """ Perturb one element in the triplets
        """
        n_batch = (test_size + batch_size - 1) // batch_size
        ranks = []
        local_scores = []
        temporal_scores = []
        targets = []
        for idx in range(n_batch):
            local_score, _ = self.get_score(ent_embed_local, rel_enc_local, all_embeds_g_local, s, r, o, test_size, mask, graph, idx, batch_size, mode)
            temporal_score, target = self.get_score(ent_embed_temporal, rel_enc_temporal, all_embeds_g_temporal, s, r, o, test_size, mask, graph, idx, batch_size, mode)
            local_scores.append(local_score)
            temporal_scores.append(temporal_score)
            targets.append(target)
        scores = weight * torch.cat(local_scores) + (1 - weight) * torch.cat(temporal_scores)
        targets = torch.cat(targets)
        ranks.append(self.sort_and_rank(torch.sigmoid(scores), targets))
        return torch.cat(ranks)

    def get_score(self, ent_mean, rel_enc_means, all_ent_embeds, s, r, o, test_size, mask, graph, idx, batch_size=100, mode ='tail'):
        batch_start = idx * batch_size
        batch_end = min(test_size, (idx + 1) * batch_size)
        batch_r = rel_enc_means[r[batch_start: batch_end]]

        if mode == 'tail':
            batch_s = ent_mean[s[batch_start: batch_end]]
            batch_o = all_ent_embeds
            target = o[batch_start: batch_end]
        else:
            batch_s = all_ent_embeds
            batch_o = ent_mean[o[batch_start: batch_end]]
            target = s[batch_start: batch_end]
        target = torch.tensor([graph.ids[i.item()] for i in target])

        if self.args.use_cuda:
            target = cuda(target)

        unmasked_score = self.calc_score(batch_s, batch_r, batch_o, mode=mode)
        masked_score = torch.where(mask[batch_start: batch_end], -10e6 * unmasked_score.new_ones(unmasked_score.shape), unmasked_score)
        return masked_score, target  # bsz, n_ent

    def mask_eval_set(self, test_triplets, test_size, num_ent, time, graph, mode='tail'):
        time = int(time.item())
        mask = test_triplets.new_zeros(test_size, num_ent)
        for i in range(test_size):
            h, r, t = test_triplets[i]
            h, r, t = h.item(), r.item(), t.item()
            if mode == 'tail':
                tails = self.true_tails[time][(h, r)]
                tail_idx = np.array(list(map(lambda x: graph.ids[x], tails)))
                mask[i][tail_idx] = 1
                mask[i][graph.ids[t]] = 0
            elif mode == 'head':
                heads = self.true_heads[time][(r, t)]
                head_idx = np.array(list(map(lambda x: graph.ids[x], heads)))
                mask[i][head_idx] = 1
                mask[i][graph.ids[h]] = 0
            # pdb.set_trace()
        return mask.byte()

    def combined_scores(self, local_score, temporal_score, labels, weight):
        score = weight * local_score + (1 - weight) * temporal_score
        predict_loss = F.cross_entropy(score, labels)
        return predict_loss

    def train_link_prediction(self, ent_embed, rel_embeds, triplets, neg_samples, labels, all_embeds_g, corrupt_tail=True):
        r = rel_embeds[triplets[:, 1]]
        if corrupt_tail:
            s = ent_embed[triplets[:, 0]]
            neg_o = all_embeds_g[neg_samples]
            score = self.calc_score(s, r, neg_o, mode='tail')
        else:
            neg_s = all_embeds_g[neg_samples]
            o = ent_embed[triplets[:, 2]]
            score = self.calc_score(neg_s, r, o, mode='head')
        return score

    def sort_and_rank(self, score, target):
        # pdb.set_trace()
        _, indices = torch.sort(score, dim=1, descending=True)
        indices = torch.nonzero(indices == target.view(-1, 1))
        indices = indices[:, 1].view(-1)
        return indices
