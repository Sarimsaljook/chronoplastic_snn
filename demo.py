import torch, torch.nn as nn, torch.nn.functional as F
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import time


# -------------------------------
# Simple surrogate spike function
# -------------------------------
class SpikeFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return (x > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        grad = grad_output * torch.clamp(1 - x.abs(), min=0)
        return grad


spike_fn = SpikeFn.apply


# -------------------------------
# Tiny synthetic dataset generator
# -------------------------------
def make_dataset(n=500, T=30, C=8, min_gap=3, max_gap=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    X = torch.zeros(n, T, C)
    y = torch.zeros(n)
    for i in range(n):
        a, b = torch.randint(0, C, (1,), generator=g), torch.randint(0, C, (1,), generator=g)
        t1 = torch.randint(0, T - max_gap - 1, (1,), generator=g)
        gap = torch.randint(min_gap, max_gap + 1, (1,), generator=g)
        t2 = t1 + gap
        X[i, t1, a] = 1
        X[i, t2, b] = 1
        y[i] = ((a % 2) ^ (b % 2)).float()  # label = XOR of parity
    return X, y


# -------------------------------
# Simple CPSNN core components
# -------------------------------
class ChronoSynapse(nn.Module):
    """Fast+Slow traces with learned adaptive decay"""

    def __init__(self, c_in, c_out):
        super().__init__()
        self.W = nn.Parameter(torch.randn(c_in, c_out) * 0.3)
        self.alpha_fast = 0.9
        self.alpha_slow = 0.995
        self.ctrl = nn.Sequential(nn.Linear(c_in * 2, c_in), nn.Sigmoid())

    def forward(self, spikes, fast, slow):
        fast = self.alpha_fast * fast + spikes
        ctrl_in = torch.cat([spikes, slow], dim=1)
        warp = self.ctrl(ctrl_in) * 0.9 + 0.05
        slow = torch.pow(self.alpha_slow, warp) * slow + spikes
        cur = spikes @ self.W + 0.5 * fast @ self.W + 0.5 * slow @ self.W
        return cur, fast, slow


class LIFLayer(nn.Module):
    def __init__(self, c_in, c_out, chrono=True):
        super().__init__()
        self.chrono = chrono
        self.syn = ChronoSynapse(c_in, c_out) if chrono else nn.Linear(c_in, c_out, bias=False)
        self.alpha = 0.9
        self.v_th = 1.0

    def forward(self, seq):
        T, B, _ = seq.shape
        v = torch.zeros(B, self.syn.W.shape[1], device=seq.device)
        fast = torch.zeros(B, self.syn.W.shape[0], device=seq.device)
        slow = torch.zeros_like(fast)
        outs = []
        for t in range(T):
            if self.chrono:
                cur, fast, slow = self.syn(seq[t], fast, slow)
            else:
                cur = seq[t] @ self.syn.weight.T
            v = self.alpha * v + (1 - self.alpha) * cur
            s = spike_fn(v - self.v_th)
            v = torch.where(s > 0, torch.zeros_like(v), v)
            outs.append(s)
        outs = torch.stack(outs)
        return outs.mean(0)


class CPSNN(nn.Module):
    def __init__(self, C_in, C_hidden):
        super().__init__()
        self.l1 = LIFLayer(C_in, C_hidden, chrono=True)
        self.readout = nn.Linear(C_hidden, 1)

    def forward(self, x_seq):
        h = self.l1(x_seq)
        return self.readout(h).squeeze(-1)


# -------------------------------
# Quick training loop
# -------------------------------
def run_demo():
    torch.manual_seed(0)
    device = "cpu"
    X, y = make_dataset()
    X = X.transpose(0, 1)  # [T,B,C]
    model = CPSNN(C_in=X.shape[2], C_hidden=32).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)

    print("[INFO] Training CPSNN demo (fast mode)...")
    acc_hist = []
    start = time.time()
    for epoch in range(1, 506):
        opt.zero_grad()
        logits = model(X)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        loss.backward();
        opt.step()
        with torch.no_grad():
            acc = ((torch.sigmoid(logits) > 0.5) == y).float().mean().item()
            acc_hist.append(acc)
        print(f"Epoch {epoch:02d} | loss={loss.item():.4f} | acc={acc:.3f}")
    print(f"[DONE] Trained in {time.time() - start:.2f}s | Final acc={acc_hist[-1]:.3f}")

    plt.plot(acc_hist, label='CPSNN Accuracy', color='royalblue')
    plt.xlabel('Epoch');
    plt.ylabel('Accuracy');
    plt.title('Fast CPSNN Demo');
    plt.legend()
    plt.tight_layout();
    plt.show()


if __name__ == "__main__":
    run_demo()
