def forward(self, x, params=None):
    skip_k = self.hparams.temporal_skip
    
    
    if self.use_slstm:
    syn_states = []
    mem_states = []
    for layer in self.lif_layers:
    syn, mem = layer.reset_mem(batch_size=B)
    syn_states.append(syn.to(x_seq.device))
    mem_states.append(mem.to(x_seq.device))
    
    
    # Only track layer outputs if temporal skip is enabled
    track_layer_time = skip_k != -1
    if track_layer_time:
    layer_outputs_time = [ [] for _ in range(len(self.lif_layers)) ]
    
    
    spike_count_acc = [0.0 for _ in range(len(self.lif_layers))]
    
    
    for t in range(T):
    curr_input = x_seq[t]
    for i, layer in enumerate(self.lif_layers):
    spk, syn_states[i], mem_states[i] = layer(curr_input, syn_states[i], mem_states[i])
    
    
    if self.hparams.use_layernorm and self.layer_norms[i] is not None:
    spk = self.layer_norms[i](spk)
    
    
    if self.hparams.use_bntt and self.layer_bntt[i] is not None:
    spk = self.layer_bntt[i][t](spk)
    
    
    if skip_k != -1 and t >= skip_k:
    prev_t = x_seq[t - skip_k]
    prev_proj = self.temp_skip_projs[i](prev_t)
    spk = spk + prev_proj
    
    
    if self.hparams.layer_skip > 0 and i >= self.hparams.layer_skip and track_layer_time:
    skip_layer_out = layer_outputs_time[i - self.hparams.layer_skip][t]
    if self.layer_skip_projs[i] is not None:
    skip_proj = self.layer_skip_projs[i](skip_layer_out)
    else:
    skip_proj = skip_layer_out
    spk = spk + skip_proj
    
    
    if track_layer_time:
    layer_outputs_time[i].append(spk)
    
    
    curr_input = spk
    spike_count_acc[i] += spk.mean().item()
    
    
    if track_layer_time:
    last_layer_seq = torch.stack(layer_outputs_time[-1], dim=0)
    else:
    last_layer_seq = curr_input.unsqueeze(0).repeat(T,1,1)
    
    
    self.spike_counts = [c / T for c in spike_count_acc]
    x_seq = last_layer_seq
    
    
    else:
    layer_outputs = []
    for i, lif in enumerate(self.lif_layers):
    lif_out = lif(x_seq)
    self.spike_counts.append(lif_out.mean())
    
    
    if self.hparams.use_layernorm and self.layer_norms[i] is not None:
    lif_out = lif_out.permute(1, 0, 2)
    lif_out = self.layer_norms[i](lif_out)
    lif_out = lif_out.permute(1, 0, 2)
    
    
    if self.hparams.use_bntt:
    lif_out_norm = []
    for t in range(lif_out.shape[0]):
    lif_out_norm.append(self.layer_bntt[i][t](lif_out[t]))
    lif_out = torch.stack(lif_out_norm, dim=0)
    
    
    if skip_k == -1 or skip_k >= T:
    x_seq = lif_out
    else:
    prev = torch.zeros_like(x_seq)
    prev[skip_k:] = x_seq[:-skip_k]
    prev_proj = self.temp_skip_projs[i](prev)
    x_seq = lif_out + prev_proj
    
    
    if self.hparams.layer_skip > 0 and i >= self.hparams.layer_skip:
    skip_layer_out = layer_outputs[i - self.hparams.layer_skip]
    skip_proj = self.layer_skip_projs[i](skip_layer_out)
    x_seq = x_seq + skip_proj
    
    
    layer_outputs.append(x_seq)
    
    
    x_seq_b = x_seq.permute(1, 0, 2)
    out = self.output_layer(x_seq_b).squeeze()
    return out