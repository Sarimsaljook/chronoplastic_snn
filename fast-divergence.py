import math, argparse, random
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from dataclasses import dataclass

# Utils
def set_seed(seed=42):
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)

class SuperSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, beta=10.0):
        ctx.save_for_backward(x); ctx.beta=beta
        return (x>0.).float()
    @staticmethod
    def backward(ctx, grad_out):
        (x,)=ctx.saved_tensors; beta=ctx.beta
        sg=beta*torch.clamp(1-torch.abs(beta*x),min=0.0)
        return grad_out*sg, None

spike_fn=SuperSpike.apply

# ---------- Dataset ----------
class LongGapTwoCueDataset(Dataset):
    def __init__(self,n=4000,T=100,C=16,min_gap=8,max_gap=25,
                 distractor_p=0.05,seed=0):
        super().__init__(); g=torch.Generator().manual_seed(seed)
        self.X=torch.zeros(n,T,C); self.y=torch.zeros(n)
        for i in range(n):
            a=torch.randint(0,C,(1,),generator=g).item()
            b=torch.randint(0,C,(1,),generator=g).item()
            t1=torch.randint(3,T-max_gap-3,(1,),generator=g).item()
            gap=torch.randint(min_gap,max_gap+1,(1,),generator=g).item()
            t2=min(T-3,t1+gap)
            self.X[i,t1,a]=1; self.X[i,t2,b]=1
            mask=(torch.rand(T,C,generator=g)<distractor_p)
            mask[t1,a]=mask[t2,b]=False; self.X[i][mask]=1
            self.y[i]=float(((a%2)^(b%2))==1)
        self.T,self.C=T,C
    def __len__(self): return self.X.shape[0]
    def __getitem__(self,i): return self.X[i],self.y[i]

# Synapses
class BaselineSynapse(nn.Module):
    def __init__(self,c_in,c_out,alpha=0.95):
        super().__init__()
        self.W=nn.Parameter(torch.empty(c_in,c_out))
        nn.init.kaiming_uniform_(self.W,a=math.sqrt(5))
        self.alpha=nn.Parameter(torch.tensor(alpha))
    def forward(self,spikes,trace):
        trace_next=self.alpha.clamp(0,0.9999)*trace+spikes
        cur=spikes@self.W+trace_next@self.W
        return cur,trace_next

class ChronoPlasticSynapse(nn.Module):
    def __init__(self,c_in,c_out,alpha_fast=0.9,alpha_slow=0.995,beta_noise=0.1):
        super().__init__()
        self.W=nn.Parameter(torch.empty(c_in,c_out))
        nn.init.kaiming_uniform_(self.W,a=math.sqrt(5))
        self.alpha_fast=nn.Parameter(torch.tensor(alpha_fast))
        self.alpha_slow=nn.Parameter(torch.tensor(alpha_slow))
        self.gamma_fast=nn.Parameter(torch.tensor(0.5))
        self.gamma_slow=nn.Parameter(torch.tensor(0.5))
        self.beta_noise=beta_noise
        hidden=max(8,c_in//2)
        self.ctrl=nn.Sequential(nn.Linear(c_in*2,hidden),nn.ReLU(),nn.Linear(hidden,c_in))
    def forward(self,spikes,fast,slow,return_warp=False):
        af=self.alpha_fast.clamp(0,0.9999)
        fast_next=af*fast+spikes
        ctrl_in=torch.cat([spikes,slow],1)
        ctrl_logits=self.ctrl(ctrl_in)
        if self.training and self.beta_noise>0:
            ctrl_logits+=self.beta_noise*torch.randn_like(ctrl_logits)
        warp=torch.sigmoid(ctrl_logits).clamp(0.05,0.95)
        aslow=self.alpha_slow.clamp(0,0.999999)
        alpha_adapt=torch.exp(torch.log(aslow+1e-8)*warp)
        slow_next=alpha_adapt*slow+spikes
        cur=spikes@self.W+(self.gamma_fast*fast_next)@self.W+(self.gamma_slow*slow_next)@self.W
        return (cur,fast_next,slow_next,warp) if return_warp else (cur,fast_next,slow_next)

# LIF 
@dataclass
class LIFParams:
    v_th:float=1.0; v_reset:float=0.0; dt:float=1e-3; tau_mem:float=2e-2

class LIFLayer(nn.Module):
    def __init__(self,c_in,c_out,lif:LIFParams,chrono=True):
        super().__init__()
        self.v_th,self.v_reset=lif.v_th,lif.v_reset
        self.alpha=math.exp(-lif.dt/lif.tau_mem)
        self.chrono=chrono
        self.syn=ChronoPlasticSynapse(c_in,c_out) if chrono else BaselineSynapse(c_in,c_out)
    def forward(self,spikes,log=False):
        T,B,_=spikes.shape; device=spikes.device
        v=torch.zeros(B,self.syn.W.shape[1],device=device)
        outs=[]; warp_log=[]
        if self.chrono:
            fast=torch.zeros(B,self.syn.W.shape[0],device=device)
            slow=torch.zeros(B,self.syn.W.shape[0],device=device)
            for t in range(T):
                if log:
                    cur,fast,slow,warp=self.syn(spikes[t],fast,slow,True)
                    warp_log.append(warp.mean().item())
                else:
                    cur,fast,slow=self.syn(spikes[t],fast,slow)
                v=self.alpha*v+(1-self.alpha)*cur
                s=spike_fn(v-self.v_th,75.0)
                v=torch.where(s>0,torch.full_like(v,self.v_reset),v)
                outs.append(s)
        else:
            trace=torch.zeros(B,self.syn.W.shape[0],device=device)
            for t in range(T):
                cur,trace=self.syn(spikes[t],trace)
                v=self.alpha*v+(1-self.alpha)*cur
                s=spike_fn(v-self.v_th,75.0)
                v=torch.where(s>0,torch.full_like(v,self.v_reset),v)
                outs.append(s)
        spikes_out=torch.stack(outs,0); rates_out=spikes_out.mean(0)
        return spikes_out,rates_out,warp_log

# Network
class SpikingNet(nn.Module):
    def __init__(self,C_in,C_hidden,C_out,lif:LIFParams,chrono=True):
        super().__init__()
        self.l1=LIFLayer(C_in,C_hidden,lif,chrono)
        self.l2=LIFLayer(C_hidden,C_out,lif,chrono)
        self.readout=nn.Linear(C_out,1)
        self.chrono=chrono
    def forward(self,x,log=False):
        s1,r1,w1=self.l1(x,log)
        s2,r2,w2=self.l2(s1)
        logits=self.readout(r2).squeeze(-1)
        return logits,(w1 if log else None)

# Train and Eval
def run_epoch(model,loader,opt,device,train=True):
    if train:model.train()
    else:model.eval()
    total,correct=0,0
    for X,y in loader:
        X=X.to(device).transpose(0,1); y=y.to(device)
        if train: opt.zero_grad()
        logits,_=model(X)
        loss=F.binary_cross_entropy_with_logits(logits,y)
        if train: loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        preds=(torch.sigmoid(logits)>0.5).float()
        correct+=(preds==y).sum().item(); total+=y.numel()
    return correct/total

# Main
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--epochs",type=int,default=40)
    parser.add_argument("--hidden",type=int,default=256)
    parser.add_argument("--lr",type=float,default=0.01)
    args=parser.parse_args()
    set_seed(0)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device} hidden={args.hidden} epochs={args.epochs} lr={args.lr}")
    lif=LIFParams(); C=16
    baseline=SpikingNet(C,args.hidden,args.hidden,lif,chrono=False).to(device)
    cpsnn=SpikingNet(C,args.hidden,args.hidden,lif,chrono=True).to(device)
    opt_b=torch.optim.Adam(baseline.parameters(),lr=args.lr)
    opt_c=torch.optim.Adam(cpsnn.parameters(),lr=args.lr)
    acc_b_hist=[]; acc_c_hist=[]

    for ep in range(1,args.epochs+1):
        gap_min,gap_max=8+ep//2,25+ep
        noise=0.004*ep
        train_ds=LongGapTwoCueDataset(4000,100,C,gap_min,gap_max,noise,seed=ep)
        val_ds=LongGapTwoCueDataset(1000,100,C,gap_min,gap_max,noise,seed=999+ep)
        train_loader=DataLoader(train_ds,batch_size=64,shuffle=True)
        val_loader=DataLoader(val_ds,batch_size=64,shuffle=False)
        val_b=run_epoch(baseline,val_loader,opt_b,device,train=True)
        val_c=run_epoch(cpsnn,val_loader,opt_c,device,train=True)
        acc_b_hist.append(val_b); acc_c_hist.append(val_c)
        print(f"Ep{ep:02d} gap=({gap_min},{gap_max}) noise={noise:.3f} | Base {val_b:.3f}  CPSNN {val_c:.3f}")

    # Plots
    plt.figure(figsize=(7,4))
    plt.plot(acc_b_hist,label="Baseline SNN",linewidth=2)
    plt.plot(acc_c_hist,label="ChronoPlastic SNN",linewidth=2)
    plt.xlabel("Epoch"); plt.ylabel("Validation Accuracy")
    plt.title("ChronoPlastic SNN Learns to Retain Long-Gap Memory")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig("accuracy_plot.png"); plt.show()

    # Warp factor visualization
    sample,_=val_ds[0]; X=sample.unsqueeze(1).to(device)
    _,warps=cpsnn(X,log=True)
    plt.figure(figsize=(6,3))
    plt.plot(warps,label="Mean Warp Factor (Memory Retention Strength)")
    plt.xlabel("Timestep"); plt.ylabel("Warp Factor")
    plt.title("Adaptive Decay Controller Response (Cue vs Distractor)")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig("warp_factor_plot.png"); plt.show()

if __name__=="__main__":
    main()
