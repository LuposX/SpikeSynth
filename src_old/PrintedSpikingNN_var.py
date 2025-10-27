import torch
from MyTransformer import GPT  # Replace with the actual module and class name
torch.serialization.add_safe_globals([GPT])  # Trust this class for unpickling

# ===============================================================================
# ============================ Single Spike Generator ===========================
# ===============================================================================


class pSpikeGenerator(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.spike_generator = torch.load(
            './utils/final_SG.model', map_location=self.DEVICE, weights_only=False)
        self.spike_generator.train(False)
        for param in self.spike_generator.parameters():
            param.requires_grad = False

    @property
    def DEVICE(self):
        return self.args.DEVICE

    def forward(self, x):
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
        self.N = args.N_train  # Added: store number of variations
        self.epsilon = args.e_train  # Added: store epsilon for variation

        self.SG = SGLayer(n_out, args)
        self.INV = INV

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
    def theta_noisy(self):
        mean = self.theta.repeat(self.N, 1, 1).to(self.device)  # Added: replicate for each variation
        noise = ((torch.rand(mean.shape, device=self.device) * 2.) - 1.) * self.epsilon + 1.  # Added: multiplicative noise
        return mean * noise  # Added: apply noise to conductances

    @property
    def W(self):
        G = torch.sum(self.theta_noisy.abs(), axis=1, keepdim=True)  # Modified: batch-wise sum for N variations
        W = self.theta_noisy.abs() / (G + 1e-10)  # Modified: normalization over variations
        return W.to(self.device)

    def MAC(self, a):
        positive = self.theta_noisy.clone().to(self.device)  # Modified: use noisy theta
        #print(positive[:,0,0])
        positive[positive >= 0] = 1.
        positive[positive < 0] = 0.
        negative = 1. - positive

        a_extend = torch.cat([a,
                              torch.ones([a.shape[0], 1]).to(self.device),
                              torch.zeros([a.shape[0], 1]).to(self.device)], dim=1)
        a_neg = self.INV(a_extend)
        a_neg[:, -1] = 0.

        z = torch.matmul(a_extend.unsqueeze(0), self.W * positive) + \
            torch.matmul(a_neg.unsqueeze(0), self.W * negative)  # Modified: broadcast for N variations
        return z.mean(0)  # Modified: average across variations

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

    def UpdateVariation(self, N, epsilon):  # Added: method to update variation
        self.N = N
        self.epsilon = epsilon

# ===============================================================================
# ======================== Printed Neural Network ===============================
# ===============================================================================


class PrintedSpikingNeuralNetwork(torch.nn.Module):
    def __init__(self, topology, args):
        super().__init__()
        self.args = args
        self.N = args.N_train  # Added: variation count
        self.epsilon = args.e_train  # Added: noise level

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

    def UpdateVariation(self, N, epsilon):  # Added: propagate variation updates to submodules
        self.N = N
        self.epsilon = epsilon
        for layer in self.model:
            if hasattr(layer, 'UpdateVariation'):
                layer.UpdateVariation(N, epsilon)

    def GetParam(self):
        weights = [p for name, p in self.named_parameters() if name.endswith('theta_') or name.endswith('beta')]
        nonlinear = [p for name, p in self.named_parameters() if name.endswith('rt_')]
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
