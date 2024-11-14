import torch
import torch.nn as nn
import numpy as np

def sinkhorn(Q, nmb_iters):
    '''sinkhorn algorithm '''
    with torch.no_grad():
        Q = shoot_infs(Q)
        sum_Q = torch.sum(Q)
        Q /= sum_Q
        if torch.cuda.is_available():
            r = torch.ones(Q.shape[0]).cuda(non_blocking=True) / Q.shape[0]
            c = torch.ones(Q.shape[1]).cuda(non_blocking=True) / (-1 * Q.shape[1])
        else:
            r = torch.ones(Q.shape[0]) / Q.shape[0]
            c = torch.ones(Q.shape[1]) / (-1 * Q.shape[1])
        for it in range(nmb_iters):
            u = torch.sum(Q, dim=1)
            u = r / u
            u = shoot_infs(u)
            Q *= u.unsqueeze(1)
            Q *= (c / torch.sum(Q, dim=0)).unsqueeze(0)
        return (Q / torch.sum(Q, dim=0, keepdim=True)).t().float()

def shoot_infs(inp_tensor):
    """Replaces inf by maximum of tensor"""
    mask_inf = torch.isinf(inp_tensor)
    ind_inf = torch.nonzero(mask_inf)
    if len(ind_inf) > 0:
        for ind in ind_inf:
            if len(ind) == 2:
                inp_tensor[ind[0], ind[1]] = 0
            elif len(ind) == 1:
                inp_tensor[ind[0]] = 0
        m = torch.max(inp_tensor)
        for ind in ind_inf:
            if len(ind) == 2:
                inp_tensor[ind[0], ind[1]] = m
            elif len(ind) == 1:
                inp_tensor[ind[0]] = m
    return inp_tensor

def process_output(output1, output2):
    epsilon = 0.3
    temprature = 0.1
    n_ite = 10 
    softmax = nn.Softmax(dim=1).cuda()
    q1 = output1 / epsilon
    q2 = output2 / epsilon
    q1 = torch.exp(q1).t()
    q2 = torch.exp(q2).t()
    q1 = sinkhorn(q1, n_ite)
    q2 = sinkhorn(q2, n_ite)

    p1 = softmax(output1 / temprature)
    p2 = softmax(output2 / temprature)
    return p1, p2, q1, q2


def result_std(acc, nmi, pur):
    acc = np.array(acc)
    nmi = np.array(nmi)
    pur = np.array(pur)
    print('acc_mean:{:.4f} acc_std:{:.4f} nmi_mean:{:.4f} nmi_std:{:.4f} pur_mean:{:.4f} pur_std:{:.4f}'
          .format(acc.mean(), acc.std(), nmi.mean(), nmi.std(), pur.mean(), pur.std()))
    return