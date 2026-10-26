"""Frozen r70 inference with explicit independent and complete-set boundaries."""
import numpy as np
import torch
from .chainprint_hierarchical_readout import membership,readout
from .chainprint_local_response import LocalResponse
from .chainprint_prototype_calibration import score_components
from .chainprint_spatial_transition import retain_or_pool


class FrozenChainPrintCandidate:
    def __init__(self,base_model,base_head,fine_model,fine_head,weights,*,blend=.5,bins=128,pseudocount=2.):
        self.experts=((base_model,base_head),(fine_model,fine_head))
        self.device=next(base_head.parameters()).device
        self.weights=torch.as_tensor(weights,dtype=torch.float64)
        if self.weights.shape!=(2,3): raise ValueError('two frozen three-expert weight rows required')
        self.blend=float(blend)
        if not 0<=self.blend<=1: raise ValueError('invalid blend')
        self.operator=LocalResponse(self.device,bins,pseudocount)
        for model,head in self.experts:
            model.eval(); head.eval()

    @torch.no_grad()
    def score(self,*,query_before,query_after,support_before,support_after,candidates):
        """Return independent log probabilities; accepts one or several independent endpoints."""
        if query_before is None:
            raise ValueError('r70 requires query-before evidence; endpoint-only needs a separate method')
        query_before=np.asarray(query_before,dtype=np.float32)
        query_after=np.asarray(query_after,dtype=np.float32)
        support_before=np.asarray(support_before,dtype=np.float32)
        support_after=np.asarray(support_after,dtype=np.float32)
        if query_after.ndim==3: query_after=query_after[None]
        if (query_before.ndim!=3 or query_before.shape[0]!=3 or support_before.shape!=query_before.shape
            or support_after.shape!=(4,*query_before.shape) or query_after.shape[1:]!=query_before.shape
            or query_after.ndim!=4 or len(query_after)<1):
            raise ValueError('RGB endpoint/support shapes disagree')
        if (len(candidates)!=4 or len({tuple(x) for x in candidates})!=4
            or len({len(x) for x in candidates})!=1 or len(candidates[0]) not in (2,3)
            or any(not isinstance(g,(int,np.integer)) or not 0<=g<4 for x in candidates for g in x)):
            raise ValueError('four distinct length-two/three anonymous histories required')
        images=np.concatenate((support_before[None],support_after,query_before[None],query_after),0)
        if not np.isfinite(images).all(): raise ValueError('nonfinite image values')
        raw=[]
        for model,head in self.experts:
            evidence=model.frontend(model._float_tensor(np.ascontiguousarray(images)))
            tokens=retain_or_pool(evidence.tokens,evidence.pooled,spatial=True)
            encoded={'support_before':tokens[:1].expand(4,-1,-1),'support_after':tokens[1:5],
                     'query_before':tokens[5:6],'query_after':tokens[6:]}
            raw.append(score_components(head,encoded,candidates)[0].cpu().numpy())
        block={'query':query_before,'query_after':query_after,'support':support_before,
               'support_after':support_after,'bank':candidates}
        response=self.operator.scores(block)['nonlinear']
        panel=torch.as_tensor(np.stack((raw[0],raw[1],(1-self.blend)*raw[0]+self.blend*response),0)[None],dtype=torch.float64)
        return readout(panel,membership([candidates],'cpu'),self.weights,True)[0].numpy()

    @staticmethod
    def complete_set_posterior(independent_scores,temperature=.25):
        """Explicitly use the additional known four-query bijection constraint."""
        import itertools
        values=np.asarray(independent_scores,dtype=np.float64)
        if values.shape!=(4,4) or temperature<=0: raise ValueError('complete four-by-four set required')
        permutations=np.asarray(list(itertools.permutations(range(4))))
        totals=values[np.arange(4)[None],permutations].sum(-1)/temperature
        probability=np.exp(totals-totals.max()); probability/=probability.sum()
        marginals=np.zeros((4,4))
        for mass,permutation in zip(probability,permutations):
            marginals[np.arange(4),permutation]+=mass
        return np.log(np.maximum(marginals,1e-300))
