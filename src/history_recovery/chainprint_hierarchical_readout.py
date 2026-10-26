"""Anonymous within-query combination of set and order probabilities."""
import torch


def membership(banks,device):
    masks=[]
    for bank in banks:
        if not bank or len({tuple(word) for word in bank})!=len(bank):
            raise ValueError('distinct nonempty candidate histories required')
        groups=sorted({tuple(sorted(word)) for word in bank})
        masks.append([[float(tuple(sorted(word))==g) for word in bank] for g in groups])
    maximum=max(len(rows) for rows in masks)
    masks=[rows+[[0.]*len(rows[0]) for _ in range(maximum-len(rows))] for rows in masks]
    result=torch.tensor(masks,dtype=torch.float64,device=device)
    if not bool((result.sum(-2)==1).all()): raise ValueError('candidate partition is invalid')
    return result


def readout(scores,mask,weights,hierarchical):
    # Axes: block, expert, query, candidate. Each query is processed independently.
    probabilities=scores.softmax(-1)
    if not hierarchical:
        return torch.einsum('k,nkqc->nqc',weights[0].softmax(-1),probabilities).clamp_min(1e-30).log()
    group=torch.einsum('nkqc,ngc->nkqg',probabilities,mask)
    conditional=probabilities/torch.einsum('nkqg,ngc->nkqc',group,mask).clamp_min(1e-30)
    combined_group=torch.einsum('k,nkqg->nqg',weights[0].softmax(-1),group)
    combined_order=torch.einsum('k,nkqc->nqc',weights[1].softmax(-1),conditional)
    combined=torch.einsum('nqg,ngc->nqc',combined_group,mask)*combined_order
    return combined.clamp_min(1e-30).log()


def fit(scores,mask,hierarchical,steps,penalty):
    weights=torch.zeros((2 if hierarchical else 1,scores.shape[1]),dtype=scores.dtype,device=scores.device,requires_grad=True)
    optimizer=torch.optim.Adam([weights],lr=.05)
    for _ in range(steps):
        values=readout(scores,mask,weights,hierarchical)
        loss=-values.diagonal(dim1=-2,dim2=-1).mean()+penalty*weights.square().mean()
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    return weights.detach()
