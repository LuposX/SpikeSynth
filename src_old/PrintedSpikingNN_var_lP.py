import torch
import torch.nn as nn
from MyTransformer import GPT  # Replace with the actual module and class name
torch.serialization.add_safe_globals([GPT])  # Trust this class for unpickling


# ==============================================================================
# =========================== Single Spike Generator ===========================
# ==============================================================================

class pSpikeGenerator(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        # Load frozen spike generator
        self.spike_generator = torch.load(
            './simulation_small/NNs/predictor_gpt-nano_lr_-5_seed_0',
            map_location=torch.device(self.DEVICE), weights_only=False
        )
        self.spike_generator.train(False)
        for param in self.spike_generator.parameters():
            param.requires_grad = False

        self.spike_generator = self.spike_generator.to(self.DEVICE)

        # Define raw trainable parameters (unconstrained)
        self.raw_params = nn.Parameter(torch.randn(1, 6))

        # Define target per-parameter ranges (shape: (6,))
        self.low = torch.tensor([0.1, 0.1, 0.1, 0.4, 0.4, 0.6], device=self.DEVICE)
        self.high = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], device=self.DEVICE)

    @property
    def DEVICE(self):
        return self.args.DEVICE

    def transform_params(self):
        r = (self.high - self.low) / 2
        c = (self.high + self.low) / 2
        base = c + r * torch.tanh(self.raw_params)  # Apply tanh transformation

        if hasattr(self.args, 'e_train') and hasattr(self.args, 'N_train'):
            eps = self.args.e_train * (self.high - self.low)  # variation range
            noise = (torch.rand((self.args.N_train, 1, 6), device=self.DEVICE) * 2 - 1) * eps  # (N, 1, 6)
            varied = base + noise  # (N, 1, 6)
            return varied  # shape: (N, 1, 6)
        else:
            return base  # fallback without variation

    def forward(self, x):
        batch_size = x.shape[0]
        T = x.shape[2]

        transformed = self.transform_params()  # (N, 1, 6) or (1, 6)

        if transformed.dim() == 3:
            # Variation is enabled
            transformed = transformed.expand(-1, batch_size, -1)  # (N, B, 6)
            transformed = transformed.permute(1, 0, 2)  # (B, N, 6)
            transformed = transformed.unsqueeze(3).expand(-1, -1, -1, T)  # (B, N, 6, T)
            x = x.unsqueeze(1).expand(-1, self.args.N_train, -1, -1)  # (B, N, C, T)
            x = torch.cat([x, transformed], dim=2)  # (B, N, C+6, T)
            if True:
                x = x.mean(1)  # average over variations
            else:
                x = x[:, 0]  # use first variation
        else:
            # No variation
            expanded_params = transformed.expand(batch_size, -1)  # (B, 6)
            expanded_params = expanded_params.unsqueeze(2).expand(-1, -1, T)  # (B, 6, T)
            x = torch.cat([x, expanded_params], dim=1)  # (B, C+6, T)
        return self.spike_generator(x)

    def UpdateArgs(self, args):
        self.args = args

    def SetParamAveraging(self, average: bool):
        self.args.average_param_variations = average

# ==============================================================================
# ============================== SG Layer ======================================
# ==============================================================================

class SGLayer(torch.nn.Module):
    def __init__(self, N, args):
        super().__init__()
        self.args = args
        self.SG_Group = torch.nn.ModuleList(
            [pSpikeGenerator(args) for _ in range(N)])

    @property
    def DEVICE(self):
        return self.args.DEVICE

    def forward(self, x):
        result = []
        for n in range(len(self.SG_Group)):
            x_temp = x[:, n, :].unsqueeze(-1)
            result.append(self.SG_Group[n](x_temp))
        return torch.stack(result).permute(1, 0, 2)

    def UpdateArgs(self, args):
        self.args = args

# ==============================================================================
# =====================  Learnable Negative Weight Circuit  =====================
# ==============================================================================

class Inv(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def forward(self, z):
        return - torch.tanh(z)

# ==============================================================================
# ============================= Printed Layer ===================================
# ==============================================================================

class pLayer(torch.nn.Module):
    def __init__(self, n_in, n_out, args, INV):
        super().__init__()
        self.args = args
        self.N = args.N_train  # Added: number of variations
        self.epsilon = args.e_train  # Added: noise level for conductance variation

        self.SG = SGLayer(n_out, args)
        self.INV = INV
        theta = torch.rand([n_in + 2, n_out])/10. + args.gmin
        theta[-2, :] = args.gmax - theta[-2, :]
        self.theta_ = torch.nn.Parameter(theta, requires_grad=True)

    def UpdateVariation(self, N, epsilon):  # Added: method to update variation settings
        self.N = N
        self.epsilon = epsilon

    @property
    def device(self):
        return self.args.DEVICE

    @property
    def theta(self):
        self.theta_.data.clamp_(-self.args.gmax, self.args.gmax)
        theta_temp = self.theta_.clone()
        theta_temp[theta_temp.abs() < self.args.gmin] = 0.
        return theta_temp.detach() + self.theta_ - self.theta_.detach()

    @property
    def theta_noisy(self):
        mean = self.theta.repeat(self.N, 1, 1).to(self.device)
        noise = ((torch.rand(mean.shape, device=self.device) * 2.) - 1.) * self.epsilon + 1.
        return mean * noise

    @property
    def W(self):
        G = torch.sum(self.theta_noisy.abs(), axis=1, keepdim=True)
        W = self.theta_noisy.abs() / (G + 1e-10)
        return W.to(self.device)

    def MAC(self, a):
        a = a.to(self.device)
        positive = self.theta_noisy.clone().to(self.device)
        positive[positive >= 0] = 1.
        positive[positive < 0] = 0.
        negative = 1. - positive

        a_extend = torch.cat([
            a,
            torch.ones([a.shape[0], 1], device=self.device),
            torch.zeros([a.shape[0], 1], device=self.device)
        ], dim=1)
        a_neg = self.INV(a_extend)
        a_neg[:, -1] = 0.

        z = torch.matmul(a_extend.unsqueeze(0), self.W * positive) + \
            torch.matmul(a_neg.unsqueeze(0), self.W * negative)

        if True:
            return z.mean(0)
        else:
            return z[0]

    def forward(self, x):
        T = x.shape[2]
        result = []
        self.power = torch.tensor(0.).to(self.device)
        for t in range(T):
            mac = self.MAC(x[:, :, t])
            result.append(mac)
            self.power += self.MACPower(x[:, :, t], mac)
        z_new = torch.stack(result, dim=2)
        self.power = self.power / T
        a_new = self.SG(z_new)
        return a_new

    @property
    def g_tilde(self):
        g_initial = self.theta_.abs()
        g_min = g_initial.min(dim=0, keepdim=True)[0]
        scaler = self.args.pgmin / g_min
        return g_initial * scaler

    def MACPower(self, x, y):
        x_extend = torch.cat([
            x,
            torch.ones([x.shape[0], 1], device=self.device),
            torch.zeros([x.shape[0], 1], device=self.device)], dim=1)
        x_neg = self.INV(x_extend)
        x_neg[:, -1] = 0.

        E = x_extend.shape[0]
        M = x_extend.shape[1]
        N = y.shape[1]

        positive = self.theta.clone().detach().to(self.device)
        positive[positive >= 0] = 1.
        positive[positive < 0] = 0.
        negative = 1. - positive

        Power = torch.tensor(0.).to(self.device)

        for m in range(M):
            for n in range(N):
                Power += self.g_tilde[m, n] * (
                    (x_extend[:, m]*positive[m, n]+x_neg[:, m]*negative[m, n])-y[:, n]).pow(2.).sum()
        Power = Power / E
        return Power

# ==============================================================================
# ======================== Printed Spiking Neural Network ======================
# ==============================================================================

class PrintedSpikingNeuralNetwork(torch.nn.Module):
    def __init__(self, topology, args):
        super().__init__()
        self.args = args
        self.INV = Inv(args)
        self.model = torch.nn.Sequential()
        for i in range(len(topology)-1):
            self.model.add_module(str(i)+'_pLayer', pLayer(topology[i], topology[i+1], args, self.INV))

    @property
    def DEVICE(self):
        return self.args.DEVICE

    def forward(self, x):
        return self.model(x)

    @property
    def power(self):
        power = torch.tensor(0.).to(self.DEVICE)
        for layer in self.model:
            if hasattr(layer, 'power'):
                power += layer.power
        return power

    def UpdateArgs(self, args):
        self.args = args
        for layer in self.model:
            if hasattr(layer, 'UpdateArgs'):
                layer.UpdateArgs(args)

    def SetAveraging(self, average: bool):  # Added: toggle for averaging over variations
        self.args.average_variations = average
        for layer in self.model:
            if hasattr(layer, 'UpdateVariation'):
                layer.UpdateVariation(self.args.N_train, self.args.e_train)

    def GetParam(self):
        weights = [p for name, p in self.named_parameters()
                   if name.endswith('theta_') or name.endswith('beta') or name.endswith('raw_params')]
        nonlinear = [p for name, p in self.named_parameters()
                     if name.endswith('rt_')]
        if self.args.lnc:
            return weights + nonlinear
        else:
            return weights

# ==============================================================================
# ============================= Loss Function ===================================
# ==============================================================================

class LossFN(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def standard(self, prediction, label):
        label = label.reshape(-1, 1)
        fy = prediction.gather(1, label).reshape(-1, 1)
        fny = prediction.clone()
        fny = fny.scatter_(1, label, -10 ** 10)
        fnym = torch.max(fny, axis=1).values.reshape(-1, 1)
        l = torch.max(self.args.m + self.args.T - fy, torch.tensor(0)) + torch.max(self.args.m + fnym, torch.tensor(0))
        return torch.mean(l)

    def celoss(self, prediction, label):
        return torch.nn.CrossEntropyLoss()(prediction, label)

    def forward(self, prediction, label):
        if self.args.loss == 'pnnloss':
            return self.standard(prediction, label)
        elif self.args.loss == 'celoss':
            return self.celoss(prediction, label)

class LFLoss(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.loss_fn = LossFN(args)

    def forward(self, model, x, label):
        prediction = model(x)
        L = [self.loss_fn(prediction[:, :, step], label) for step in range(prediction.shape[2])]
        return torch.stack(L).mean() + 0.1 * model.power
