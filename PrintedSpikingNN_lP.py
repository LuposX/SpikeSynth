import torch
import torch.nn as nn
from MyTransformer import GPT  
torch.serialization.add_safe_globals([GPT])  

# ===============================================================================
# ============================ Single Spike Generator ===========================
# ===============================================================================


class pSpikeGenerator(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        # Load frozen spike generator
        self.spike_generator = torch.load(
            './surrogate/NNs/predictor_final',
            map_location=self.DEVICE, weights_only=False
        )
        self.spike_generator.train(False)
        for param in self.spike_generator.parameters():
            param.requires_grad = False

        # Define raw trainable parameters (unconstrained)
        self.raw_params = nn.Parameter(torch.randn(1, 6))

        # Define target per-parameter ranges (shape: (6,))
        self.low = torch.tensor([0.1, 0.1, 0.1, 0.4, 0.4, 0.6], device=self.DEVICE)
        self.high = torch.tensor([1.0, 1.0,  1.0, 1.0, 1.0,  1.0], device=self.DEVICE)

    @property
    def DEVICE(self):
        return self.args.DEVICE

    def transform_params(self):
        # Apply tanh transformation and scale to [low, high] for each parameter
        # raw_params shape: (1, 6) → transformed_params shape: (1, 6)
        r = (self.high - self.low) / 2
        c = (self.high + self.low) / 2
        return c + r * torch.tanh(self.raw_params)

    def forward(self, x):
        batch_size = x.shape[0]
        T = x.shape[2]

        # Transform and expand trainable parameters
        extra_params = self.transform_params()  # (1, 6)
        #print(extra_params)
        expanded_params = extra_params.expand(batch_size, -1)  # (B, 6)
        expanded_params = expanded_params.unsqueeze(2).expand(-1, -1, T)  # (B, 6, T)

        # Concatenate with input
        x = torch.cat([x, expanded_params], dim=1)  # (B, C+6, T)
        return self.spike_generator(x)

    def UpdateArgs(self, args):
        self.args = args


# ===============================================================================
# ============================== SG Layer =======================================
# ===============================================================================

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


# ===============================================================================
# =====================  Learnable Negative Weight Circuit  =====================
# ===============================================================================

class Inv(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        
    def forward(self, z):
        return - torch.tanh(z)


# ===============================================================================
# ============================= Printed Layer ===================================
# ===============================================================================

class pLayer(torch.nn.Module):
    def __init__(self, n_in, n_out, args, INV):
        super().__init__()
        self.args = args
        # define spike generators
        self.SG = SGLayer(n_out, args)
        # define nonlinear circuits
        self.INV = INV
        # initialize conductances for weights
        theta = torch.rand([n_in + 2, n_out])/10. + args.gmin
        theta[-2, :] = args.gmax - theta[-2, :]
        self.theta_ = torch.nn.Parameter(theta, requires_grad=True)

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
    def W(self):
        return self.theta.abs() / torch.sum(self.theta.abs(), axis=0, keepdim=True)

    def MAC(self, a):
        # 0 and positive thetas are corresponding to no negative weight circuit
        positive = self.theta.clone().to(self.device)
        positive[positive >= 0] = 1.
        positive[positive < 0] = 0.
        negative = 1. - positive
        a_extend = torch.cat([a,
                              torch.ones([a.shape[0], 1]).to(self.device),
                              torch.zeros([a.shape[0], 1]).to(self.device)], dim=1)
        a_neg = self.INV(a_extend)
        a_neg[:, -1] = 0.
        z = torch.matmul(a_extend, self.W * positive) + \
            torch.matmul(a_neg, self.W * negative)
        return z

    @property
    def neg_power(self):
        # Exclude bias and dummy from power computation
        theta = self.theta.clone().detach()[:-2, :]  # [input_dim, output_dim]
        
        # Identify negative weights
        negative_mask = (theta < 0).float()
        N_neg_hard = negative_mask.sum()

        # Soft (gradient-aware) count of negative weights
        soft_count = 1 - torch.sigmoid(self.theta[:-2, :])
        soft_count = soft_count * negative_mask
        soft_N_neg = soft_count.max(dim=1)[0].sum()

        # Surrogate power from InvRT
        inv_power_scalar = self.INV.power.item() if hasattr(self, "INV") else 0.0

        # Compute final power (hard + relaxed)
        power_hard = inv_power_scalar * N_neg_hard
        power_soft = inv_power_scalar * soft_N_neg
        
        return power_hard + power_soft - power_soft.detach()

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
        x_extend = torch.cat([x,
                              torch.ones([x.shape[0], 1]).to(self.device),
                              torch.zeros([x.shape[0], 1]).to(self.device)], dim=1)
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

    def UpdateArgs(self, args):
        self.args = args

# ===============================================================================
# ======================== Printed Neural Network ===============================
# ===============================================================================


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

    def GetParam(self):
        weights = [p for name, p in self.named_parameters()
                if name.endswith('theta_') or name.endswith('beta') or name.endswith('raw_params')]
        nonlinear = [p for name, p in self.named_parameters()
                    if name.endswith('rt_')]
        if self.args.lnc:
            return weights + nonlinear
        else:
            return weights

# ===============================================================================
# ============================= Loss Functin ====================================
# ===============================================================================


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
        l = torch.max(self.args.m + self.args.T - fy, torch.tensor(0)
                      ) + torch.max(self.args.m + fnym, torch.tensor(0))
        L = torch.mean(l)
        return L

    def celoss(self, prediction, label):
        lossfn = torch.nn.CrossEntropyLoss()
        return lossfn(prediction, label)

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
        L = []
        for step in range(prediction.shape[2]):
            L.append(self.loss_fn(prediction[:, :, step], label))
        return torch.stack(L).mean() + 0.1 * model.power
