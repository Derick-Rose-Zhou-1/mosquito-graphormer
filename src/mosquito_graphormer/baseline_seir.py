"""Archived comparator architecture; caller supplies training-only scaled features."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .baseline_temporal import _TemporalEncoder

class GraphormerSEIRObservation(nn.Module):
    """Graphormer-derived SEIR latent incidence with a train-fitted case observation map."""
    def __init__(self, num_features, num_cities, horizon=7, hidden=64):
        super().__init__()
        self.temporal=_TemporalEncoder(num_features,hidden,0.1)
        self.city_embedding=nn.Embedding(num_cities,hidden)
        layer=nn.TransformerEncoderLayer(hidden,4,2*hidden,0.1,batch_first=True,activation='gelu',norm_first=True)
        self.graph_encoder=nn.TransformerEncoder(layer,num_layers=2)
        self.beta_head=nn.Sequential(nn.LayerNorm(hidden),nn.Linear(hidden,hidden),nn.GELU(),nn.Linear(hidden,1))
        self.exposed_scale_head=nn.Sequential(nn.LayerNorm(hidden),nn.Linear(hidden,1))
        self.log_sigma=nn.Parameter(torch.tensor(np.log(1/5),dtype=torch.float32))
        self.log_gamma=nn.Parameter(torch.tensor(np.log(1/6),dtype=torch.float32))
        self.log_mobility=nn.Parameter(torch.tensor(np.log(0.2),dtype=torch.float32))
        # log1p incidence -> log1p reported cases. Parameters are fit only on train data.
        self.raw_obs_slope=nn.Parameter(torch.tensor(0.54132485)) # softplus ~= 1
        self.obs_intercept=nn.Parameter(torch.tensor(0.0))
        self.horizon=int(horizon)

    def forward(self,x,history,population,city_indices,adjacency,mobility):
        nodes=self.temporal(x)+self.city_embedding(city_indices).unsqueeze(0)
        allowed=adjacency.to(device=x.device)>0
        allowed=allowed|torch.eye(allowed.shape[0],dtype=torch.bool,device=x.device)
        h=self.graph_encoder(nodes,mask=~allowed)
        beta=0.8*torch.sigmoid(self.beta_head(h).squeeze(-1))
        e_scale=0.25+1.5*torch.sigmoid(self.exposed_scale_head(h).squeeze(-1))
        sigma=self.log_sigma.exp().clamp(1e-3,1.0)
        gamma=self.log_gamma.exp().clamp(1e-3,1.0)
        eta=self.log_mobility.exp().clamp(0.0,2.0)
        recent=history[:,-5:,:].mean(dim=1).clamp_min(0)
        infectious=history[:,-1,:].clamp_min(0)+1e-3
        exposed=(recent*e_scale).clamp_min(1e-3)
        pop=population.clamp_min(1.0)
        susceptible=(pop-infectious-exposed).clamp_min(0)
        slope=F.softplus(self.raw_obs_slope).clamp(0.1,3.0)
        forecasts=[]
        for _ in range(self.horizon):
            moved_i=torch.matmul(infectious,mobility.T)
            pressure=(infectious+eta*moved_i)/pop
            new_exposed=(beta*pressure*susceptible).clamp_min(0)
            new_exposed=torch.minimum(new_exposed,susceptible)
            new_infectious=sigma*exposed
            recovered=gamma*infectious
            susceptible=(susceptible-new_exposed).clamp_min(0)
            exposed=(exposed+new_exposed-new_infectious).clamp_min(0)
            infectious=(infectious+new_infectious-recovered).clamp_min(0)
            # Explicit observation/output head; trains against the same case-count labels.
            cases=torch.expm1((slope*torch.log1p(new_infectious.clamp_min(0))+self.obs_intercept).clamp(-20,20)).clamp_min(0)
            forecasts.append(cases)
        return torch.stack(forecasts,dim=-1)
