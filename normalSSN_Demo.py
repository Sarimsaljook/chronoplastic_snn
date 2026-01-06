import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

# Dataset
class LongGapTwoCueDataset(Dataset):
    """Two cue spikes with variable gap XOR label."""
    def __init__(self, n=800, T=40, C=12, min_gap=3, max_gap=10, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.X = torch.zeros(n, T, C)
        self.y = torch.zeros(n)
        for i in range(n):
            a = torch.randint(0, C, (1,), generator=g).item()
            b = torch.randint(0, C, (1,), generator=g).item()
            t1 = torch.randint(1, T - max_gap - 1, (1,), generator=g).item()
            gap = torch.randint(min_gap, max_gap + 1, (1,), generator=g).item()
            t2 = min(T - 2, t1 + gap)
            self.X[i, t1, a] = 1.0
            self.X[i, t2, b] = 1.0
            self.y[i] = float(((a % 2) ^ (b % 2)) == 1)
        self.T, self.C = T, C
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

# SuperSpike Surrogate
class SuperSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, beta=10.0):
        ctx.save_for_backward(x)
        ctx.beta = beta
        return (x > 0.).float()
    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        beta = ctx.beta
        sg = beta * torch.clamp(1 - torch.abs(beta * x), min=0.0)
        return grad_output * sg, None

spike_fn = SuperSpike.apply

# Basic Synapse with single exponential trace
class SimpleSynapse(nn.Module):
    def __init__(self, c_in, c_out, alpha=0.9):
        super().__init__()
        self.W = nn.Parameter(torch.empty(c_in, c_out))
        nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))
        self.alpha = nn.Parameter(torch.tensor(alpha))
    def forward(self, spikes, trace):
        a = torch.clamp(self.alpha, 0.0, 0.9999)
        trace_next = a * trace + spikes
        current = spikes @ self.W + trace_next @ self.W
        return current, trace_next

# LIF Neuron Layer
class LIFLayer(nn.Module):
    def __init__(self, c_in, c_out, dt=1e-3, tau=2e-2):
        super().__init__()
        self.syn = SimpleSynapse(c_in, c_out)
        self.alpha = math.exp(-dt / tau)
        self.v_th = 1.0
        self.v_reset = 0.0
    def forward(self, x_seq):
        T, B, _ = x_seq.shape
        v = torch.zeros(B, self.syn.W.shape[1], device=x_seq.device)
        trace = torch.zeros(B, self.syn.W.shape[0], device=x_seq.device)
        outs = []
        for t in range(T):
            cur, trace = self.syn(x_seq[t], trace)
            v = self.alpha * v + (1 - self.alpha) * cur
            s = spike_fn(v - self.v_th, 25.0)
            v = torch.where(s > 0, torch.full_like(v, self.v_reset), v)
            outs.append(s)
        spikes_out = torch.stack(outs, dim=0)
        rates = spikes_out.mean(dim=0)
        return spikes_out, rates

# Network and Training
class StandardSNN(nn.Module):
    def __init__(self, C_in, hidden, out):
        super().__init__()
        self.l1 = LIFLayer(C_in, hidden)
        self.l2 = LIFLayer(hidden, out)
        self.readout = nn.Linear(out, 1)
    def forward(self, x):
        s1, r1 = self.l1(x)
        s2, r2 = self.l2(s1)
        return self.readout(r2).squeeze(-1)


def train(model, loader, opt, device):
    model.train()
    total, correct = 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        X = X.transpose(0, 1).contiguous()
        opt.zero_grad()
        logits = model(X)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        loss.backward()
        opt.step()
        preds = (torch.sigmoid(logits) > 0.5).float()
        correct += (preds == y).sum().item()
        total += y.numel()
    return correct / total

# Main
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = LongGapTwoCueDataset(n=400, T=40, C=12)
    loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    model = StandardSNN(C_in=12, hidden=64, out=64).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)

    accs = []
    for ep in range(505):
        acc = train(model, loader, opt, device)
        accs.append(acc)
        print(f"Epoch {ep + 1} | acc={acc:.3f}")

    # Plot results
    plt.figure(figsize=(7,4))
    plt.plot(accs, label="Standard SNN Accuracy", color="red")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Standard SNN Demo")
    plt.ylim(0.5, 1.0)
    plt.legend()
    plt.grid(True)
    plt.show()
    print("[INFO] Training Done")

if __name__ == "__main__":
    main()
