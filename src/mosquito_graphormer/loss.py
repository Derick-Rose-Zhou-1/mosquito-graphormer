"""Four-component coefficient-of-variation weighted forecast loss."""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F

BASE_NAMES = ('backbone_log_huber','backbone_poisson_deviance','final_log_huber','final_poisson_deviance')

class CoVFourLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.names = BASE_NAMES
        self.eps = eps
        self.register_buffer('steps', torch.zeros((),dtype=torch.long))
        for name in ('mean_losses','mean_ratios','m2_ratios'):
            self.register_buffer(name, torch.zeros(len(self.names)))
    def weights(self):
        if self.steps.item() < 2: return torch.full_like(self.mean_losses,1/len(self.names))
        cov = torch.sqrt(self.m2_ratios/(self.steps-1)+self.eps)/self.mean_ratios.clamp_min(self.eps)
        if not torch.isfinite(cov).all() or cov.sum().item()<=self.eps:
            return torch.full_like(self.mean_losses,1/len(self.names))
        return cov/cov.sum()
    @torch.no_grad()
    def _update(self, losses):
        n=int(self.steps.item())
        ratios=torch.ones_like(losses) if n==0 else losses/self.mean_losses.clamp_min(self.eps)
        self.mean_losses.add_((losses-self.mean_losses)/(n+1))
        delta=ratios-self.mean_ratios
        self.mean_ratios.add_(delta/(n+1))
        self.m2_ratios.add_(delta*(ratios-self.mean_ratios))
        self.steps.add_(1)
    def _deviance(self,p,y):
        mu=p+self.eps
        term=torch.where(y>0,y*(y.clamp_min(self.eps).log()-mu.log()),torch.zeros_like(y))
        return (mu-y+term).clamp_min(0)
    def forward(self,output,target,mask=None):
        b,f=output['backbone_prediction'],output['raw_prediction']
        if b.shape!=f.shape or f.shape!=target.shape or f.ndim!=3 or f.shape[-1]!=7:
            raise ValueError('Expected common [batch,nodes,7]')
        if mask is not None:
            if mask.shape!=target.shape or mask.dtype!=torch.bool or not mask.any(): raise ValueError('Invalid mask')
            b,f,target=b[mask],f[mask],target[mask]
        for value in (b,f,target):
            if not torch.isfinite(value).all() or (value<0).any(): raise FloatingPointError('Invalid counts')
        terms=[F.smooth_l1_loss(torch.log1p(b),torch.log1p(target)),self._deviance(b,target).mean(),
               F.smooth_l1_loss(torch.log1p(f),torch.log1p(target)),self._deviance(f,target).mean()]
        terms=torch.stack(terms); alpha=self.weights().to(terms).detach()
        total=torch.dot(alpha,terms)
        if self.training: self._update(terms.detach().to(self.mean_losses))
        return total, {'terms':{n:float(x.detach().cpu()) for n,x in zip(self.names,terms)},
                       'weights':{n:float(x.cpu()) for n,x in zip(self.names,alpha)}}

