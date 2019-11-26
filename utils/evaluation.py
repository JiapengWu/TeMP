import torch


def calc_metrics(ent_mean, rel_enc_means, test_triplets, eval_bz=100):
    with torch.no_grad():
        s = test_triplets[:, 0]
        r = test_triplets[:, 1]
        o = test_triplets[:, 2]
        test_size = test_triplets.shape[0]

        # perturb subject
        ranks_s = perturb_and_get_rank(ent_mean, rel_enc_means, o, r, s, test_size, eval_bz)
        # perturb object
        ranks_o = perturb_and_get_rank(ent_mean, rel_enc_means, s, r, o, test_size, eval_bz)

        ranks = torch.cat([ranks_s, ranks_o])
        ranks += 1 # change to 1-indexed

        mrr = torch.mean(1.0 / ranks.float()).item()
        # print("MRR (raw): {:.6f}".format(mrr.item()))
        hit_1 = torch.mean((ranks <= 1).float()).item()
        hit_3 = torch.mean((ranks <= 3).float()).item()
        hit_10 = torch.mean((ranks <= 10).float()).item()
        pos_facts = torch.cat([ent_mean[s], rel_enc_means[r], ent_mean[o]], dim=1)

    return mrr, hit_1, hit_3, hit_10, torch.max(pos_facts, dim=0)[0]


def perturb_and_get_rank(ent_mean, rel_enc_means, a, r, b, test_size, batch_size=100):
    """ Perturb one element in the triplets
    """
    n_batch = (test_size + batch_size - 1) // batch_size
    ranks = []
    for idx in range(n_batch):
        print("batch {} / {}".format(idx, n_batch))
        batch_start = idx * batch_size
        batch_end = min(test_size, (idx + 1) * batch_size)
        batch_a = a[batch_start: batch_end]
        batch_r = r[batch_start: batch_end]
        emb_ar = ent_mean[batch_a] * rel_enc_means[batch_r] # E x D
        emb_ar = emb_ar.transpose(0, 1).unsqueeze(2) # size: D x E x 1
        emb_c = ent_mean.transpose(0, 1).unsqueeze(1) # size: D x 1 x V
        # out-prod and reduce sum
        out_prod = torch.bmm(emb_ar, emb_c) # size D x E x V
        score = torch.sum(out_prod, dim=0) # size E x V
        score = torch.sigmoid(score)
        target = b[batch_start: batch_end]
        ranks.append(sort_and_rank(score, target))
    return torch.cat(ranks)


def sort_and_rank(score, target):
    _, indices = torch.sort(score, dim=1, descending=True)
    indices = torch.nonzero(indices == target.view(-1, 1))
    indices = indices[:, 1].view(-1)
    return indices
