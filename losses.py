

import torch
import torch.nn as nn
from torch.nn import functional as F

class InstProtoCLR(nn.Module):
    def __init__(self, temperature,devices):
        super(InstProtoCLR, self).__init__()
        self.ce = nn.CrossEntropyLoss(reduction='none')
        self.temperature = temperature
        self.devices = devices

    def forward(self, anchor, prototypes, peso_labels):

        batch_size = anchor.size(0)

        anchor = F.normalize(anchor) 

        prototypes = F.normalize(prototypes).to(self.devices)

        pos_proto_id = peso_labels
 
        pos_prototypes = prototypes[pos_proto_id]

        proto_similarity_matrix = torch.matmul(anchor, pos_prototypes.T) 

     

        mask = torch.eye(batch_size, dtype=torch.bool)  
        assert proto_similarity_matrix.shape == mask.shape

        proto_positives = proto_similarity_matrix[mask].view(batch_size, -1) 

        proto_negatives = proto_similarity_matrix[~mask].view(batch_size, -1) 


        proto_logits = torch.cat([proto_positives, proto_negatives], dim=1)  
        proto_labels = torch.zeros(batch_size, dtype=torch.long).to(self.devices)  


        proto_logits /= self.temperature
    
        loss_proto = self.ce(proto_logits, proto_labels)

        w = torch.ones(batch_size).to(self.devices)
        loss_proto = torch.sum(w*(loss_proto))/ torch.sum(w)

        return loss_proto

class ContrastiveLoss(nn.Module):


    def __init__(self, margin=0, shift=2., measure=False, max_violation=False):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.shift = shift
        if measure == 'order':
            self.sim = order_sim
        else:
            self.sim = lambda x, y: x.mm(y.t())

        self.max_violation = max_violation
        self.count = 1

    def set_margin(self, margin):
        self.margin = margin

    def loss_func(self, cost, tau):
        cost = (cost - cost.diag().reshape([-1, 1])).exp()
        I = (cost.diag().diag() == 0)
        return cost[I].sum() / (cost.shape[0] * (cost.shape[0] - 1))

    def forward(self, im, s=None, tau=1., lab=None):
        if s is None:
            scores = im
            diagonal = im[:, 0].view(im.size(0), 1)
            d1 = diagonal.expand_as(scores)

            cost = (self.margin + scores - d1).clamp(min=0)

            if self.max_violation:
                cost = cost.max(1)[0]

            return cost.sum()

        else:
 
            scores = self.sim(im, s)
            self.count += 1
            
            diagonal = scores.diag().view(im.size(0), 1)
            d1 = diagonal.expand_as(scores)
            d2 = diagonal.t().expand_as(scores)
            mask_s = (scores >= (d1 - self.margin)).float().detach()
            cost_s = scores * mask_s + (1. - mask_s) * (scores - self.shift)
            mask_im = (scores >= (d2 - self.margin)).float().detach()
            cost_im = scores * mask_im + (1. - mask_im) * (scores - self.shift)
            loss = (-cost_s.diag() + tau * (cost_s / tau).exp().sum(1).log() + self.margin).mean() + (-cost_im.diag() + tau * (cost_im / tau).exp().sum(0).log() + self.margin).mean()
            return loss
        
class InstProtoRC(nn.Module):
    def __init__(self,devices):
        super(InstProtoRC, self).__init__()
        self.mse = nn.MSELoss()
        self.devices = devices

    def forward(self,args,F_I,F_T,code_I,code_T,prototypes):

        batch_size = F_I.size(0)

        F_I = F.normalize(F_I)
        F_T = F.normalize(F_T)

        prototypes = F.normalize(prototypes).to(self.devices)

        q1 = torch.matmul(F_I,prototypes.T)
        qi = torch.softmax(q1,dim=1)
        S_ii = torch.mm(qi,qi.T)
  


        q2 = torch.matmul(F_T,prototypes.T)
        qt = torch.softmax(q2,dim=1)
        S_tt = torch.mm(qt,qt.T)



        B_I = F.normalize(code_I)
        B_T = F.normalize(code_T)

        BI_BI = B_I.mm(B_I.t())
        BT_BT = B_T.mm(B_T.t())
        BI_BT = B_I.mm(B_T.t())


        S = S_ii + S_tt





        loss2 = self.mse(BI_BT, S)






        return loss2



