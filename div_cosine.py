import torch
import geomloss

a, b = torch.randn((100, 512)), torch.randn((100, 512))
p = 2
entreg = .1 # entropy regularization factor for Sinkhorn

OTLoss = geomloss.SamplesLoss(
    loss='sinkhorn', p=p,
    # 对于p=1或p=2的情形
    cost=geomloss.utils.distances if p==1 else geomloss.utils.squared_distances,
    blur=entreg**(1/p), backend='tensorized')
pW = OTLoss(a, b)


def cost_func(a, b, p=2, metric='cosine'):
    """ a, b in shape: (B, N, D) or (N, D)
    """ 
    assert type(a)==torch.Tensor and type(b)==torch.Tensor, 'inputs should be torch.Tensor'
    if metric=='euclidean' and p==1:
        return geomloss.utils.distances(a, b)
    elif metric=='euclidean' and p==2:
        return geomloss.utils.squared_distances(a, b)
    else:
        if a.dim() == 3:
            x_norm = a / a.norm(dim=2)[:, :, None]
            y_norm = b / b.norm(dim=2)[:, :, None]
            M = 1 - torch.bmm(x_norm, y_norm.transpose(-1, -2))
        elif a.dim() == 2:
            x_norm = a / a.norm(dim=1)[:, None]
            y_norm = b / b.norm(dim=1)[:, None]
            M = 1 - torch.mm(x_norm, y_norm.transpose(0, 1))
        M = pow(M, p)
        return M

metric = 'cosine'

OTLoss = geomloss.SamplesLoss(
    loss='sinkhorn', p=p,
    cost=lambda a, b: cost_func(a, b, p=p, metric=metric),
    blur=entreg**(1/p), backend='tensorized')

pW = OTLoss(a, b)
print('pW={}'.format(pW))

