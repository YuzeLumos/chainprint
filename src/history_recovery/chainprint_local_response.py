"""Anonymous block-transform response identification from one support pair."""
import math
import torch
from torch.nn import functional as F


class LocalResponse:
    def __init__(self,device,bins=128,pseudocount=2.):
        self.device=device; self.bins=bins; self.pseudocount=pseudocount
        k=torch.arange(8,device=device,dtype=torch.float32)[:,None]
        n=torch.arange(8,device=device,dtype=torch.float32)[None]
        self.basis=torch.cos(math.pi*(n+.5)*k/8)*.5
        self.basis[0]*=2**-.5

    def encode(self,image):
        n,c,h,w=image.shape
        if c!=3 or h%8 or w%8: raise ValueError('divisible RGB image required')
        patches=image.reshape(n,c,h//8,8,w//8,8).permute(0,1,2,4,3,5)
        coefficients=self.basis@patches@self.basis.T
        return coefficients.permute(0,1,4,5,2,3).reshape(n,192,-1)

    def decode(self,coefficients,height,width):
        n=coefficients.shape[0]
        patches=coefficients.reshape(n,3,8,8,height//8,width//8).permute(0,1,4,5,2,3)
        pixels=self.basis.T@patches@self.basis
        return pixels.permute(0,1,2,4,3,5).reshape(n,3,height,width)

    def fit(self,before,after):
        source=self.encode(before).expand(len(after),-1,-1)
        delta=self.encode(after)-source
        mean=source.mean(-1,keepdim=True)
        scale=source.std(-1,keepdim=True,unbiased=False).clamp_min(1e-3)
        z=(source-mean)/scale; target=delta/scale
        shift=target.mean(-1,keepdim=True)
        slope=(z*(target-shift)).mean(-1,keepdim=True)/(z.square().mean(-1,keepdim=True)+1e-3)
        residual=target-(shift+slope*z)
        coordinates=((z.clamp(-4,4)+4)*(self.bins-1)/8).round().long()
        sums=torch.zeros((*z.shape[:2],self.bins),device=self.device)
        counts=torch.zeros_like(sums)
        sums.scatter_add_(-1,coordinates,residual)
        counts.scatter_add_(-1,coordinates,torch.ones_like(residual))
        curve=sums/(counts+self.pseudocount)
        return {'mean':mean,'scale':scale,'slope':slope,'shift':shift,'curve':curve}

    def predict(self,query,fit,bank,nonlinear):
        current=query.expand(len(bank),-1,-1,-1).clone()
        for position in range(len(bank[0])):
            refs=torch.tensor([word[position] for word in bank],device=self.device)
            coefficients=self.encode(current)
            z=(coefficients-fit['mean'][refs])/fit['scale'][refs]
            delta=fit['shift'][refs]+fit['slope'][refs]*z
            if nonlinear:
                coordinates=(z.clamp(-4,4)+4)*(self.bins-1)/8
                lower=coordinates.floor().long(); upper=(lower+1).clamp(max=self.bins-1)
                alpha=coordinates-lower
                curve=fit['curve'][refs]
                delta=delta+(1-alpha)*curve.gather(-1,lower)+alpha*curve.gather(-1,upper)
            coefficients=coefficients+fit['scale'][refs]*delta
            current=self.decode(coefficients,*query.shape[-2:]).clamp(0,1)
            current=torch.round(current*255)/255
        return current

    def scores(self,block):
        before=torch.as_tensor(block['support'][None],device=self.device)
        after=torch.as_tensor(block['support_after'],device=self.device)
        query=torch.as_tensor(block['query'][None],device=self.device)
        observed=torch.as_tensor(block['query_after'],device=self.device)-query
        observed=F.normalize(observed.flatten(1),dim=-1,eps=1e-6)
        fit=self.fit(before,after)
        scores={}
        for name,nonlinear in (('linear',False),('nonlinear',True)):
            predicted=self.predict(query,fit,block['bank'],nonlinear)-query
            predicted=F.normalize(predicted.flatten(1),dim=-1,eps=1e-6)
            scores[name]=(5*(observed@predicted.T)).cpu().numpy()
        return scores
