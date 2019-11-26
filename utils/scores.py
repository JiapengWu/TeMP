import torch
def distmult(s, r, o):
    return torch.sum(s * r * o, dim=1)


def complex(s, r, o):
    half_size = s.shape[1] / 2
    s_re = s[:, :half_size]
    s_im = s[:, half_size:]
    r_re = r[:, :half_size]
    r_im = r[:, half_size:]
    o_re = o[:, :half_size]
    o_im = o[:, half_size:]
    # pdb.set_trace()
    return torch.sum(
        s_re * o_re * r_re + s_im * o_im * r_re
        + s_re * o_im * r_im - s_im * o_re * r_im, dim=-1
    )