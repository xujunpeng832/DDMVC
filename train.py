import torch
from network import Network
from metric import valid
from torch.utils.data import Dataset
import numpy as np
import argparse
import random
from loss import Loss
from dataloader import load_data
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
import random
import utils
import time
import matplotlib.pyplot as plt
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# -----------parameters:alpha, beta, lamb-----------------
# handwritten   0.1 10 1
# Scene_15  1 1 1
# Caltech-5V    1 1 0.1 

Dataname = 'handwritten'
parser = argparse.ArgumentParser(description='train')
parser.add_argument('--dataset', default=Dataname)
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument("--temperature_f", default=0.5)
parser.add_argument("--temperature_l", default=1.0)
parser.add_argument("--learning_rate", default=0.0003)
parser.add_argument("--weight_decay", default=0.)
parser.add_argument("--workers", default=8)
parser.add_argument("--mse_epochs", default=200)
parser.add_argument("--con_epochs", default=50)
parser.add_argument("--feature_dim", default=512)
parser.add_argument("--high_feature_dim", default=128)
parser.add_argument("--alpha", default=0.1)
parser.add_argument("--beta", default=10)
parser.add_argument("--lamb", default=1)
parser.add_argument("--pretrain", default=True)
parser.add_argument("--T", default=1) 

args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if args.dataset == "Scene_15":
    args.con_epochs = 100
    seed = 5
if args.dataset == "Caltech-5V":
    args.con_epochs = 100
    seed = 5
if args.dataset == "handwritten":
    args.con_epochs = 100
    seed = 5

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(seed)


dataset, dims, view, data_size, class_num = load_data(args.dataset)

data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )

def contrastive_train(epoch):
    tot_loss = 0. 
    mes = torch.nn.MSELoss()
    for batch_idx, (xs, _, _) in enumerate(data_loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        optimizer.zero_grad()
        hs, qs, xrs, zs, ps = model.forward(xs)

        loss_list = []
        for v in range(view):
            for w in range(v+1, view):
                # -------------------------LP------------------------------
                p1, p2, q1, q2 = utils.process_output(ps[v], ps[w])
                loss_con = -1 * (torch.mean(torch.sum(q1 * torch.log(qs[w]), dim=1))+torch.mean(torch.sum(q2 * torch.log(qs[v]), dim=1))) / 2
                loss_list.append(loss_con)

                # --------------------------CL----------------------------
                loss_list.append(args.alpha*criterion.forward_feature(hs[v], hs[w]))
                loss_list.append(args.beta*criterion.forward_label(qs[v], qs[w]))	

                # -------------------------diversity-----------------------
                loss_list.append(args.alpha*criterion.VIC_loss(hs[v], hs[w],args.high_feature_dim)) 
                loss_list.append(args.beta*criterion.VIC_q_loss(qs[v], qs[w],args.batch_size))


                # -------------------------Con------------------------------
                loss_list.append(args.lamb*mes(zs[v], zs[w]))



            loss_list.append(mes(xs[v], xrs[v]))

        loss = sum(loss_list)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
    print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(tot_loss/len(data_loader)))
    return np.round(tot_loss/len(data_loader),4) 


def pretrain(epoch):
    tot_loss = 0.
    criterion = torch.nn.MSELoss()
    for batch_idx, (xs, _, _) in enumerate(data_loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        optimizer.zero_grad()
        _, _, xrs, _,_= model(xs)
        loss_list = []
        for v in range(view):
            loss_list.append(criterion(xs[v], xrs[v]))
        loss = sum(loss_list)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
    # print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(tot_loss / len(data_loader)))

losses = []
accs = []
nmis = []
purs = []
zses = []
hses = []
comZs = []
best_acc = 0
best_epoch_nmi = 0
best_epoch_pur = 0
best_epoch_ari = 0
best_epoch = 0
best_epoch_loss = 0
if not os.path.exists('./pretrain_models'):
    os.makedirs('./pretrain_models')
T = args.T
for i in range(T):
    print("ROUND:{}".format(i+1))
    T1 = time.time() 
    model = Network(view, dims, args.feature_dim, args.high_feature_dim, class_num, device)
    # print(model)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = Loss(args.batch_size, class_num, args.temperature_f, args.temperature_l, device).to(device)
    epoch = 1
    if(args.pretrain):
        while epoch <= args.mse_epochs: 
            pretrain(epoch)
            epoch += 1
        state = model.state_dict()
        torch.save(state, './pretrain_models/' + args.dataset + '.pth') 
    else:
        print("loading pretrain model...")
        checkpoint = torch.load('./pretrain_models/' + args.dataset + '.pth')
        model.load_state_dict(checkpoint)
        epoch = args.mse_epochs + 1  
        print("pretrain model loaded...")

    while epoch <= args.mse_epochs + args.con_epochs:
        loss_= contrastive_train(epoch)
        losses.append(np.round(loss_, 4))
        acc, nmi, pur, ari = valid(model, device, dataset, view, data_size, class_num, eval_h=False)
        accs.append(np.round(acc, 4))
        nmis.append(np.round(nmi, 4))
        purs.append(np.round(pur, 4))
        if acc>best_acc:
            best_acc = np.copy(acc)
            best_epoch_nmi = np.copy(nmi)
            best_epoch_pur = np.copy(pur)
            best_epoch_ari = np.copy(ari)
            best_epoch = epoch
            best_epoch_loss = loss_
        if epoch == args.mse_epochs + args.con_epochs:
            print('---------train over---------')
            print(Dataname)
            print('Clustering results: ACC, NMI, ARI, PUR, EPOCH, LOSS')
            print('{:.4f} {:.4f} {:.4f} {:.4f} {} {:.6f}'.format(best_acc, best_epoch_nmi, best_epoch_ari, best_epoch_pur, best_epoch, best_epoch_loss))
        epoch += 1
    T2 = time.time()
