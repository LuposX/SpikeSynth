#!/usr/bin/env python

import snntorch as snn
from snntorch._layers.bntt import BatchNormTT1d

import pytorch_lightning as L
import torch
import torch.nn as nn


class SpikeSynth(L.LightningModule):
    def __init__(self,
                 num_hidden_layers,
                 num_hidden,
                 beta,
                 optimizer_class,
                 optimizer_kwargs,
                 lr,
                 batch_size,
                 dropout,
                 surrogate_gradient,
                 max_epochs,
                 temporal_skip,
                 layer_skip,
                 use_bntt,
                 scheduler_class,
                 log_every_n_steps,
                 use_layernorm,
                 neuron_type="LeakyParallel",
                 scheduler_kwargs=None,
                 bntt_time_steps=None,
                 train_dataset=None,
                 valid_dataset=None,
                 ):
        """
        Initializes the SpikeSynth model, a spiking neural network (SNN). 
        The model supports temporal and layer-wise skip connections,
        customizable surrogate gradients, and flexible optimizer/lr scheduling.

        Args:
            num_hidden_layers (int): 
                Number of recurrent Leaky Integrate-and-Fire (LIF) layers in the network.

            num_hidden (int): 
                Number of hidden neurons (units) in each LIF layer.

            beta (float): 
                Membrane potential decay constant for the LIF neurons (typically 0 < beta < 1).

            optimizer_class (type): 
                Optimizer class to use (e.g., `torch.optim.Adam`, `torch.optim.SGD`).

            neuron_type: str, one of {"LeakyParallel", "Leaky", "SLSTM"}
               "LeakyParallel" uses snn.LeakyParallel and accepts a full (T,B,F) sequence. 
               "Leaky" uses snn.Leaky and is stepped through time manually (per-time-step). SLSTM" uses snn.SLSTM (the existing code path).

            optimizer_kwargs(dict):
                The paramaters of the optimizer.

            lr (float): 
                Inital Learning rate for the optimizer.

            batch_size (int): 
                Batch size used for both training and validation DataLoaders.

            dropout (float):
                Value between [0, 1] if non zero, introduces a dropout layer.

            surrogate_gradient (torch.autograd.Function): 
                Surrogate gradient function for backpropagation through spikes
                (e.g., `snn.surrogate.atan()`).

            max_epochs (int): 
                Maximum number of training epochs. Used for scheduling when employing
                cosine annealing or similar schedulers.

            temporal_skip (int or -1): 
                Temporal skip interval in time steps for residual connections within
                each LIF layer. If `-1`, no temporal skip is applied.

            layer_skip (int or 0): 
                Number of layers to skip for inter-layer residual connections.
                For example, `layer_skip=1` adds a connection from each layer
                to the one immediately above it. If `0`, no layer skip is applied.

            use_bntt (bool):
                Whether to use 1D Batch Normalization Layer with length time_steps. 
                Uses Batch Normalisation Through Time (BNTT).

            bntt_time_steps (None or int):
                Batch Normalization Layer requries the amount of time steps that will be used i.e. sequecne length.

            train_dataset (torch.utils.data.Dataset, optional): 
                Dataset used for training. Required by `train_dataloader()`.

            valid_dataset (torch.utils.data.Dataset, optional): 
                Dataset used for validation. Required by `val_dataloader()`.

            scheduler_class (torch.optim.lr_scheduler):
                What Learning Rate Scheduler should be used for training.

            use_slstm (bool): 
                If True, use snn.SLSTM layers for the recurrent layers.

            log_every_n_steps (int):
                Log every n steps.

            use_layernorm (bool):
                If layernorm should be used afetr every Lif layer.
        """
        super().__init__()

        self.num_params = 6
        self.num_inputs = 1
        self.num_outputs = self.num_inputs
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.max_epochs = max_epochs
        self.surrogate_gradient = surrogate_gradient

        # save hyperparams (exclude big objects)
        self.save_hyperparameters(ignore=["surrogate_gradient", "train_dataset", "valid_dataset"])

        # quick validation
        if neuron_type == "SLSTM" and dropout != 0:
            raise ValueError("SLSTM doesn't support dropout.")

        # modules containers
        self.norm = nn.LayerNorm(self.num_inputs + self.num_params)
        self.lif_layers = nn.ModuleList()
        self.leaky_linears = nn.ModuleList()    # used by Leaky and RLeaky
        self.temp_skip_projs = nn.ModuleList()  # temporal skip projections per layer
        self.layer_skip_projs = nn.ModuleList() # inter-layer skip projections
        self.layer_bntt = nn.ModuleList()       # per-layer BNTT (or None)
        self.layer_norms = nn.ModuleList()      # per-layer LayerNorm (or None)

        # build layers/projections
        self._build_layers(neuron_type, bntt_time_steps)

        # final projection to outputs
        self.output_layer = nn.Linear(self.hparams.num_hidden, self.num_outputs)

    # Initialization helpers
    def _build_layers(self, neuron_type, bntt_time_steps):
        """
        Build lif_layers, leaky_linears, temp_skip_projs, layer_skip_projs,
        layer_bntt and layer_norms based on hparams.
        """
        input_size = self.num_inputs + self.num_params
        for i in range(self.hparams.num_hidden_layers):
            # create neuron layer depending on type
            if neuron_type == "SLSTM":
                layer = snn.SLSTM(
                    input_size=input_size,
                    hidden_size=self.hparams.num_hidden,
                    spike_grad=self.surrogate_gradient
                )
            elif neuron_type == "LeakyParallel":
                layer = snn.LeakyParallel(
                    input_size=input_size,
                    hidden_size=self.hparams.num_hidden,
                    beta=self.hparams.beta,
                    spike_grad=self.surrogate_gradient,
                    dropout=self.hparams.dropout
                )
            elif neuron_type == "Leaky":
                layer = snn.Leaky(
                    beta=self.hparams.beta,
                    spike_grad=self.surrogate_gradient
                )
                # linear projections for Leaky (per-step)
                if i == 0:
                    self.leaky_linears.append(nn.Linear(input_size, self.hparams.num_hidden))
                else:
                    self.leaky_linears.append(nn.Linear(self.hparams.num_hidden, self.hparams.num_hidden))
            elif neuron_type == "RLeaky":
                layer = snn.RLeaky(
                    beta=self.hparams.beta,
                    linear_features=self.hparams.num_hidden,
                    spike_grad=self.surrogate_gradient,
                    learn_recurrent=True
                )
                if i == 0:
                    self.leaky_linears.append(nn.Linear(input_size, self.hparams.num_hidden))
                else:
                    self.leaky_linears.append(nn.Linear(self.hparams.num_hidden, self.hparams.num_hidden))
            else:
                raise ValueError(f"Unknown neuron_type: {neuron_type}")

            self.lif_layers.append(layer)

            # temporal skip projection: if sizes differ and temporal skip enabled, project; else Identity
            if input_size != self.hparams.num_hidden and self.hparams.temporal_skip != -1:
                self.temp_skip_projs.append(nn.Linear(input_size, self.hparams.num_hidden))
            else:
                self.temp_skip_projs.append(nn.Identity())

            # layer-skip projection (only available when layer_skip > 0 and enough earlier layers exist)
            if self.hparams.layer_skip > 0 and i >= self.hparams.layer_skip:
                self.layer_skip_projs.append(nn.Linear(self.hparams.num_hidden, self.hparams.num_hidden))
            else:
                self.layer_skip_projs.append(None)

            # BNTT
            if self.hparams.use_bntt:
                if bntt_time_steps is None and self.hparams.bntt_time_steps is None:
                    raise ValueError("bntt_time_steps must be specified if use_bntt=True")
                time_steps = self.hparams.bntt_time_steps or bntt_time_steps
                self.layer_bntt.append(BatchNormTT1d(self.hparams.num_hidden, time_steps))
            else:
                self.layer_bntt.append(None)

            # layernorm
            if self.hparams.use_layernorm:
                self.layer_norms.append(nn.LayerNorm(self.hparams.num_hidden))
            else:
                self.layer_norms.append(None)

            # next layer sees hidden-size input
            input_size = self.hparams.num_hidden

    # Helper for forward
    def _norm_and_bntt(self, idx, tensor, t=None, per_timestep=False):
        """
        Apply LayerNorm (per-feature) and BNTT if present for layer idx.
        - If per_timestep=True, tensor is (B, F) and layernorm expects (B, F) => ok.
        - If per_timestep=False and tensor is (T, B, F), LayerNorm is applied across last dim by permuting.
        - BNTT expects indexing by time step: BNTT[idx][t](features)
        """
        # LayerNorm
        ln = self.layer_norms[idx]
        if ln is not None:
            if per_timestep:
                tensor = ln(tensor)
            else:
                # tensor shape: (T, B, F) -> apply ln over (B, F) view per-timestep
                tensor = tensor.permute(1, 0, 2)  # (B, T, F)
                tensor = ln(tensor)
                tensor = tensor.permute(1, 0, 2)  # (T, B, F)

        # BNTT
        bntt = self.layer_bntt[idx]
        if bntt is not None:
            if per_timestep:
                if t is None:
                    raise ValueError("time index t required for per-timestep BNTT")
                tensor = bntt[t](tensor)
            else:
                normed = []
                for tt in range(tensor.shape[0]):  # iterate T
                    normed.append(bntt[tt](tensor[tt]))
                tensor = torch.stack(normed, dim=0)

        return tensor

    def _temporal_proj(self, idx, t, x_seq, raw_input_t=None):
        """
        Return temporal projection for layer idx at time t.
        If t < skip_k or temporal skip disabled -> returns zeros/identity behavior by design.
        raw_input_t: (B, F_in) - used by per-step branches to project current raw input from previous time step.
        Note: temp_skip_projs[idx] expects input-sized features (T,B,F_in) or (B,F_in) depending on neuron type.
        """
        skip_k = self.hparams.temporal_skip
        if skip_k == -1 or t < skip_k:
            # return zero tensor of appropriate output size when needed by additions.
            proj = None
        else:
            prev = x_seq[t - skip_k]
            # temp_skip_projs might be Linear expecting feature dim; it accepts (B, F_in) or (T, B, F_in).
            proj = self.temp_skip_projs[idx](prev)
        return proj

     # Helper for forward
    def _layer_skip_proj(self, idx, i, t, track_layer_time, layer_outputs_time):
        """
        Return projected skip tensor from a previous layer (i - layer_skip) if available.
        """
        if self.hparams.layer_skip <= 0 or i < self.hparams.layer_skip:
            return None

        if track_layer_time:
            skip_layer_out = layer_outputs_time[i - self.hparams.layer_skip][t]
        else:
            # when not tracking time, rely on layer_outputs_time holding single-step entries (not typical)
            skip_layer_out = None

        if skip_layer_out is None:
            return None

        proj_mod = self.layer_skip_projs[i]
        if proj_mod is not None:
            return proj_mod(skip_layer_out)
        else:
            return skip_layer_out

     # Helper for forward
    def _stack_last_layer_seq(self, layer_outputs_time, T, curr_input):
        """
        Build a (T, B, F) tensor for returning.
        If we have tracked per-time outputs, stack them; otherwise replicate last `curr_input`.
        """
        if layer_outputs_time is not None:
            # last layer outputs_time is a list of T tensors
            return torch.stack(layer_outputs_time[-1], dim=0)
        else:
            return curr_input.unsqueeze(0).repeat(T, 1, 1)

    # Helper for forward
    def forward(self, x, params=None):
        self.spike_counts = []

        # ensure 3D: (batch, seq_len, channels) ; channels == 1
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        elif x.dim() != 3:
            raise ValueError(f"x must be 2D or 3D, got {x.shape}")

        batch, seq_len, _ = x.shape

        # handle static params in sequence dimension (same as original)
        if params is not None:
            if params.dim() < 2 or params.shape[1] < self.num_params:
                raise ValueError(f"params must have shape (batch, >= {self.num_params})")
            params_expanded = params.unsqueeze(2)  # (batch, num_params, 1)
            x = torch.cat([params_expanded, x], dim=1)
        elif seq_len <= self.num_params:
            raise ValueError(f"x sequence too short to contain static params, got seq_len={seq_len}")

        static_params = x[:, :self.num_params, 0]  # (batch, num_params)
        time_series = x[:, self.num_params:, 0]    # (batch, time_steps)
        time_steps = time_series.shape[1]

        static_repeated = static_params.unsqueeze(1).repeat(1, time_steps, 1)
        time_series_features = time_series.unsqueeze(2)
        x_transformed = torch.cat([static_repeated, time_series_features], dim=2)  # (batch, time_steps, F_in)

        # reorder to (T, B, F)
        x_seq = x_transformed.permute(1, 0, 2)
        T, B, F_in = x_seq.shape

        skip_k = self.hparams.temporal_skip
        neuron_type = getattr(self.hparams, "neuron_type", "LeakyParallel")

        # ------------------ SLSTM branch ------------------
        if neuron_type == "SLSTM":
            num_layers = len(self.lif_layers)
            syn_states = []
            mem_states = []
            for layer in self.lif_layers:
                syn, mem = layer.reset_mem()
                syn_states.append(syn.to(x_seq.device))
                mem_states.append(mem.to(x_seq.device))

            track_layer_time = (skip_k != -1)
            layer_outputs_time = [ [] for _ in range(num_layers) ] if track_layer_time else None
            spike_acc = [0.0] * num_layers

            for t in range(T):
                curr_input = x_seq[t]
                for i, layer in enumerate(self.lif_layers):
                    spk, syn_states[i], mem_states[i] = layer(curr_input, syn_states[i], mem_states[i])

                    # normalization & bntt (per-timestep)
                    spk = self._norm_and_bntt(i, spk, t=t, per_timestep=True)

                    # temporal skip
                    prev_proj = self._temporal_proj(i, t, x_seq)
                    if prev_proj is not None:
                        spk = spk + prev_proj

                    # layer skip
                    skip_proj = self._layer_skip_proj(i, i, t, track_layer_time, layer_outputs_time)
                    if skip_proj is not None:
                        spk = spk + skip_proj

                    if track_layer_time:
                        layer_outputs_time[i].append(spk)

                    curr_input = spk
                    spike_acc[i] += spk.mean().item()

            last_layer_seq = self._stack_last_layer_seq(layer_outputs_time, T, curr_input)
            self.spike_counts = [c / T for c in spike_acc]
            x_seq = last_layer_seq

        # ------------------ Leaky (per-step) branch ------------------
        elif neuron_type == "Leaky":
            num_layers = len(self.lif_layers)
            # initialize mem states
            mem_states = [layer.init_leaky().to(x_seq.device) for layer in self.lif_layers]
            track_layer_time = (skip_k != -1)
            layer_outputs_time = [[] for _ in range(num_layers)] if track_layer_time else None
            spike_acc = [0.0] * num_layers
            curr_input = None

            for t in range(T):
                raw_input_t = x_seq[t]  # (B, F_in)
                for i, layer in enumerate(self.lif_layers):
                    lin = self.leaky_linears[i]
                    if i == 0:
                        weighted_input = lin(raw_input_t)
                    else:
                        weighted_input = lin(curr_input)

                    # temporal skip
                    prev_proj = self._temporal_proj(i, t, x_seq)
                    if prev_proj is not None:
                        weighted_input = weighted_input + prev_proj

                    # layer skip
                    skip_proj = self._layer_skip_proj(i, i, t, track_layer_time, layer_outputs_time)
                    if skip_proj is not None:
                        weighted_input = weighted_input + skip_proj

                    # normalization and BNTT (per-timestep)
                    weighted_input = self._norm_and_bntt(i, weighted_input, t=t, per_timestep=True)

                    # layer update
                    spk, mem_states[i] = layer(weighted_input, mem_states[i])

                    if track_layer_time:
                        layer_outputs_time[i].append(spk)

                    curr_input = spk
                    spike_acc[i] += spk.mean().item()

            last_layer_seq = self._stack_last_layer_seq(layer_outputs_time, T, curr_input)
            self.spike_counts = [c / T for c in spike_acc]
            x_seq = last_layer_seq

        # ------------------ RLeaky (per-step) branch ------------------
        elif neuron_type == "RLeaky":
            num_layers = len(self.lif_layers)
            spk_states = []
            mem_states = []
            for layer in self.lif_layers:
                spk0, mem0 = layer.init_rleaky()
                spk_states.append(spk0.to(x_seq.device))
                mem_states.append(mem0.to(x_seq.device))

            track_layer_time = (skip_k != -1)
            layer_outputs_time = [[] for _ in range(num_layers)] if track_layer_time else None
            spike_acc = [0.0] * num_layers
            curr_input = None

            for t in range(T):
                raw_input_t = x_seq[t]
                for i, layer in enumerate(self.lif_layers):
                    lin = self.leaky_linears[i]
                    if i == 0:
                        weighted_input = lin(raw_input_t)
                    else:
                        weighted_input = lin(curr_input)

                    # temporal skip
                    prev_proj = self._temporal_proj(i, t, x_seq)
                    if prev_proj is not None:
                        weighted_input = weighted_input + prev_proj

                    # layer skip
                    skip_proj = self._layer_skip_proj(i, i, t, track_layer_time, layer_outputs_time)
                    if skip_proj is not None:
                        weighted_input = weighted_input + skip_proj

                    # normalization & bntt (per-timestep)
                    weighted_input = self._norm_and_bntt(i, weighted_input, t=t, per_timestep=True)

                    # RLeaky expects (input, spk_state, mem_state)
                    spk, mem_states[i] = layer(weighted_input, spk_states[i], mem_states[i])
                    spk_states[i] = spk

                    if track_layer_time:
                        layer_outputs_time[i].append(spk)

                    curr_input = spk
                    spike_acc[i] += spk.mean().item()

            last_layer_seq = self._stack_last_layer_seq(layer_outputs_time, T, curr_input)
            self.spike_counts = [c / T for c in spike_acc]
            x_seq = last_layer_seq

        # ------------------ LeakyParallel / default sequence-processing neuron ------------------
        else:
            # 'LeakyParallel' case: lif(x_seq) returns (T, B, hidden)
            layer_outputs = []
            for i, lif in enumerate(self.lif_layers):
                lif_out = lif(x_seq)  # (T, B, F_hidden)
                
                self.spike_counts.append(lif_out.mean())

                # normalization (apply layernorm across (B, F) per time step) & BNTT
                lif_out = self._norm_and_bntt(i, lif_out, per_timestep=False)

                # Temporal skip: when skip_k invalid, just assign lif_out; else add projected previous states
                if skip_k == -1 or skip_k >= T:
                    x_seq = lif_out
                else:
                    prev = torch.zeros_like(x_seq)
                    prev[skip_k:] = x_seq[:-skip_k]
                    prev_proj = self.temp_skip_projs[i](prev)
                    x_seq = lif_out + prev_proj

                # Layer-wise skip (non-time-tracked)
                if self.hparams.layer_skip > 0 and i >= self.hparams.layer_skip:
                    skip_layer_out = layer_outputs[i - self.hparams.layer_skip]
                    if skip_layer_out is not None:
                        if self.layer_skip_projs[i] is not None:
                            skip_proj = self.layer_skip_projs[i](skip_layer_out)
                        else:
                            skip_proj = skip_layer_out
                        x_seq = x_seq + skip_proj

                layer_outputs.append(x_seq)

        # x_seq is now (T, B, F_hidden). Project and return (B, T, outputs) squeezed as original.
        x_seq_b = x_seq.permute(1, 0, 2)  # (B, T, F_hidden)
        out = self.output_layer(x_seq_b).squeeze()
        return out

    
    def training_step(self, batch, batch_idx):
        X_batch, y_batch = batch
        outputs = self(X_batch)
        loss = torch.nn.MSELoss()(outputs, y_batch.float())
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        spike_logs = {
            f"spikes/train_avg_layer_{i}": avg_spikes
            for i, avg_spikes in enumerate(self.spike_counts)
        }
        self.log_dict(spike_logs, on_step=False, on_epoch=True, sync_dist=True)

        if batch_idx % self.hparams.log_every_n_steps == 0 and batch_idx > 0:
            grad_means = {}
            for name, param in self.named_parameters():
                if param.grad is not None:
                    parts = name.split('.')
                    if parts[0] == "lif_layers" and len(parts) > 1:
                        layer_name = ".".join(parts[:2])
                    else:
                        layer_name = parts[0]
                    grad_means.setdefault(layer_name, []).append(param.grad.abs().mean().item())

            grad_means = {
                f"grad_mean/{layer}": sum(vals) / len(vals)
                for layer, vals in grad_means.items()
            }
            self.log_dict(grad_means, on_step=True, on_epoch=False, sync_dist=False)

        return loss

    def validation_step(self, batch, batch_idx):
        X_batch, y_batch = batch
        outputs = self(X_batch)
        loss = torch.nn.MSELoss()(outputs, y_batch.float())
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def on_train_epoch_end(self):
        opt = self.optimizers()
        lr = opt.param_groups[0]["lr"]
        self.log("lr", lr, prog_bar=True, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer_kwargs = dict(self.hparams.optimizer_kwargs or {})
        optimizer_kwargs.setdefault("lr", self.hparams.lr)

        optimizer = self.hparams.optimizer_class(self.parameters(), **optimizer_kwargs)

        if self.hparams.scheduler_class is not None:
            if self.hparams.scheduler_class == torch.optim.lr_scheduler.CosineAnnealingLR:
                self.hparams.scheduler_kwargs.setdefault("T_max", self.hparams.max_epochs)
            if self.hparams.scheduler_class == torch.optim.lr_scheduler.ExponentialLR:
                self.hparams.scheduler_kwargs.setdefault("gamma", 0.95)
            if self.hparams.scheduler_class == torch.optim.lr_scheduler.StepLR:
                self.hparams.scheduler_kwargs.setdefault("step_size", 5)
            scheduler = self.hparams.scheduler_class(optimizer, **self.hparams.scheduler_kwargs)

            return {
                "optimizer": optimizer,
                "lr_scheduler": scheduler,
                "monitor": "val_loss",
            }
        else:
            return optimizer

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=8,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.valid_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=4,
        )
