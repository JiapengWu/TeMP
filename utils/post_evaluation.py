import torch
from utils.CorrptTriples import CorruptTriples
from utils.utils import cuda
import numpy as np
from utils.evaluation import EvaluationFilter

class PostEvaluationFilter(EvaluationFilter):
    def __init__(self, args, calc_score, graph_dict_train, graph_dict_val, graph_dict_test):
        super(PostEvaluationFilter, self).__init__(args, calc_score, graph_dict_train, graph_dict_val, graph_dict_test)

    def calc_metrics_single_graph(self, ent_embed_loc, ent_embed_rec, rel_enc_means, all_embeds_g_loc, all_embeds_g_rec,
                                  samples, weight_subject_query_subject_embed, weight_subject_query_object_embed, weight_object_query_subject_embed, weight_object_query_object_embed, graph, time, eval_bz=100):
        with torch.no_grad():
            s = samples[:, 0]
            r = samples[:, 1]
            o = samples[:, 2]
            test_size = samples.shape[0]
            num_ent = all_embeds_g_loc.shape[0]
            # pdb.set_trace()
            o_mask = self.mask_eval_set(samples, test_size, num_ent, time, graph, mode="tail")
            s_mask = self.mask_eval_set(samples, test_size, num_ent, time, graph, mode="head")
            # perturb object
            ranks_o = self.emsemble_and_get_rank(ent_embed_loc, ent_embed_rec, rel_enc_means, all_embeds_g_loc, all_embeds_g_rec, weight_object_query_subject_embed, weight_object_query_object_embed, s, r, o, test_size, o_mask, graph, eval_bz, mode='tail')
            # perturb subject
            ranks_s = self.emsemble_and_get_rank(ent_embed_loc, ent_embed_rec, rel_enc_means, all_embeds_g_loc, all_embeds_g_rec, weight_subject_query_subject_embed, weight_subject_query_object_embed, s, r, o, test_size, s_mask, graph, eval_bz, mode='head')

            ranks = torch.cat([ranks_s, ranks_o])
            ranks += 1 # change to 1-indexed
            # print("Graph {} mean ranks {}".format(time.item(), ranks.float().mean().item()))
        return ranks

    def emsemble_and_get_rank(self, ent_embed_loc, ent_embed_rec, rel_enc_means, all_embeds_g_loc, all_embeds_g_rec, weight_subject, weight_object, s, r, o, test_size, mask, graph, batch_size=100, mode ='tail'):
        """ Perturb one element in the triplets
        """
        n_batch = (test_size + batch_size - 1) // batch_size
        ranks = []
        for idx in range(n_batch):
            batch_start = idx * batch_size
            batch_end = min(test_size, (idx + 1) * batch_size)
            batch_r = rel_enc_means[r[batch_start: batch_end]]
            batch_weight_subject = weight_subject[batch_start: batch_end]
            batch_weight_object = weight_object[batch_start: batch_end]
            if mode == 'tail':
                batch_s_loc = ent_embed_loc[s[batch_start: batch_end]]
                batch_s_rec = ent_embed_rec[s[batch_start: batch_end]]
                batch_o_loc = all_embeds_g_loc
                batch_o_rec = all_embeds_g_rec
                batch_s = batch_weight_subject * batch_s_loc + (1 - batch_weight_subject) * batch_s_rec
                batch_o = (torch.matmul(batch_weight_object, batch_o_loc.unsqueeze(1)) + torch.matmul((1 - batch_weight_object), batch_o_rec.unsqueeze(1))).transpose(0,1)
                target = o[batch_start: batch_end]
            else:
                batch_s_loc = all_embeds_g_loc
                batch_s_rec = all_embeds_g_rec
                batch_o_loc = ent_embed_loc[o[batch_start: batch_end]]
                batch_o_rec = ent_embed_rec[o[batch_start: batch_end]]
                batch_s = (torch.matmul(batch_weight_subject, batch_s_loc.unsqueeze(1)) + torch.matmul((1 - batch_weight_subject), batch_s_rec.unsqueeze(1))).transpose(0,1)
                batch_o = batch_weight_object * batch_o_loc + (1 - batch_weight_object) * batch_o_rec
                target = s[batch_start: batch_end]

            target = torch.tensor([graph.ids[i.item()] for i in target])

            if self.args.use_cuda:
                target = cuda(target)

            unmasked_score = self.calc_score(batch_s, batch_r, batch_o, mode=mode)
            masked_score = torch.where(mask[batch_start: batch_end], -10e6 * unmasked_score.new_ones(unmasked_score.shape), unmasked_score)
            score = torch.sigmoid(masked_score)  # bsz, n_ent
            ranks.append(self.sort_and_rank(score, target))
        return torch.cat(ranks)


class PostEnsembleEvaluationFilter(EvaluationFilter):
    def __init__(self, args, calc_score, graph_dict_train, graph_dict_val, graph_dict_test):
        super(PostEnsembleEvaluationFilter, self).__init__(args, calc_score, graph_dict_train, graph_dict_val, graph_dict_test)

    def calc_metrics_single_graph(self, ent_embed_local, ent_embed_temporal, rel_enc_mean, all_embeds_g_local, all_embeds_g_temporal, weight_subject, weight_object, samples, graph, time, eval_bz=100):
        with torch.no_grad():
            s = samples[:, 0]
            r = samples[:, 1]
            o = samples[:, 2]
            test_size = samples.shape[0]
            num_ent = all_embeds_g_local.shape[0]
            o_mask = self.mask_eval_set(samples, test_size, num_ent, time, graph, mode="tail")
            s_mask = self.mask_eval_set(samples, test_size, num_ent, time, graph, mode="head")
            # perturb object
            ranks_o = self.emsemble_and_get_rank(ent_embed_local, ent_embed_temporal, rel_enc_mean, all_embeds_g_local, all_embeds_g_temporal, weight_subject, s, r, o, test_size, o_mask, graph, eval_bz, mode='tail')
            # perturb subject
            ranks_s = self.emsemble_and_get_rank(ent_embed_local, ent_embed_temporal, rel_enc_mean, all_embeds_g_local, all_embeds_g_temporal, weight_object, s, r, o, test_size, s_mask, graph, eval_bz, mode='head')

            ranks = torch.cat([ranks_s, ranks_o])
            ranks += 1 # change to 1-indexed
            # print("Graph {} mean ranks {}".format(time.item(), ranks.float().mean().item()))
        return ranks

    def emsemble_and_get_rank(self, ent_embed_loc, ent_embed_temporal, rel_enc_mean, all_embeds_g_local, all_embeds_g_temporal, weight, s, r, o, test_size, mask, graph, batch_size=100, mode ='tail'):
        """ Perturb one element in the triplets
        """
        n_batch = (test_size + batch_size - 1) // batch_size
        ranks = []
        local_scores = []
        temporal_scores = []
        targets = []
        for idx in range(n_batch):
            local_score, _ = self.get_score(ent_embed_loc, rel_enc_mean, all_embeds_g_local, s, r, o, test_size, mask, graph, idx, batch_size, mode)
            temporal_score, target = self.get_score(ent_embed_temporal, rel_enc_mean, all_embeds_g_temporal, s, r, o, test_size, mask, graph, idx, batch_size, mode)
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
