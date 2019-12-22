import torch


def distmult(s, r, o, mode='single'):
    # import pdb; pdb.set_trace()
    if mode == 'tail':
        return torch.sum((s * r).unsqueeze(1) * o, dim=-1)
    elif mode == 'head':
        return torch.sum(s * (r * o).unsqueeze(1), dim=-1)
    else:
        return torch.sum(s * r * o, dim=-1)


def simple(head, head_inv, rel, rel_inv, tail, tail_inv, mode='tail'):
    if mode == 'tail':
        scores1 = torch.sum((head * rel).unsqueeze(1) * tail_inv, dim=-1)
        scores2 = torch.sum((head_inv * rel_inv).unsqueeze(1) * tail, dim=-1)
    elif mode == 'head':
        scores1 = torch.sum(head * (rel * tail_inv).unsqueeze(1), dim=-1)
        scores2 = torch.sum(head_inv * (rel_inv * tail).unsqueeze(1), dim=-1)
    else:
        scores1 = torch.sum(head * rel * tail_inv, dim=-1)
        scores2 = torch.sum(head_inv * rel_inv * tail, dim=-1)
    return (scores1 + scores2) / 2


def complex(head, relation, tail, mode='single'):
    re_head, im_head = torch.chunk(head, 2, dim=-1)
    re_relation, im_relation = torch.chunk(relation, 2, dim=-1)
    re_tail, im_tail = torch.chunk(tail, 2, dim=-1)

    if mode == 'tail':
        # import pdb; pdb.set_trace()
        re_score = re_head * re_relation - im_head * im_relation
        im_score = re_head * im_relation + im_head * re_relation
        score = re_score.unsqueeze(1) * re_tail + im_score.unsqueeze(1) * im_tail
    elif mode == 'head':
        re_score = re_relation * re_tail + im_relation * im_tail
        im_score = re_relation * im_tail - im_relation * re_tail
        score = re_head * re_score.unsqueeze(1) + im_head * im_score.unsqueeze(1)
    else:
        re_score = re_head * re_relation - im_head * im_relation
        im_score = re_head * im_relation + im_head * re_relation
        score = re_score * re_tail + im_score * im_tail

    return score.sum(dim = -1)


def transE(head, relation, tail, mode='single'):
    if mode == 'tail':
        score = (head + relation).unsqueeze(1) - tail
    elif mode == 'head':
        score = head + (relation - tail).unsqueeze(1)
    else:
        score = head + relation - tail
    score = - torch.norm(score, p=1, dim=-1)
    return score
