import torch
import torch.nn as nn
import torch.nn.functional as F
from fecam import dct_channel_block,dct
class RWKVConfig:
    def __init__(self, n_layer, n_head, n_embd, dropout=0.02):
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout

        self.bias = False

class LayerNorm(nn.Module):
    def __init__(self, ndim, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-4)

class RWKV_TimeMix_x051a(nn.Module):
    def __init__(self, config, layer_id):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.version = 'v5'
        self.head_size = config.n_embd // config.n_head
        self.n_head = config.n_head

        with torch.no_grad():
            # ratio_0_to_1 = layer_id / (config.n_layer - 1)
            ratio_0_to_1 = layer_id / (config.n_layer)
            ratio_1_to_almost0 = 1.0 - (layer_id / config.n_layer)
            ddd = torch.ones(1, 1, config.n_embd)
            for i in range(config.n_embd):
                ddd[0, 0, i] = i / config.n_embd
            self.time_maa_k = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_v = nn.Parameter((torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1))
            self.time_maa_r = nn.Parameter(torch.pow(ddd, 0.5 * ratio_1_to_almost0))
            self.time_maa_g = nn.Parameter(torch.pow(ddd, 0.5 * ratio_1_to_almost0))

            if self.version == 'v5':
                decay_speed = torch.ones(self.n_head)
                for h in range(self.n_head):
                    decay_speed[h] = -6 + 5 * (h / (self.n_head - 1)) ** (0.7 + 0.3 * ratio_0_to_1)
                self.time_decay = nn.Parameter(decay_speed.unsqueeze(-1))

                tmp = torch.zeros(self.n_head)
                for h in range(self.n_head):
                    zigzag = ((h + 1) % 3 - 1) * 0.1
                    tmp[h] = ratio_0_to_1 * (1 - (h / (self.n_head - 1))) + zigzag
                self.time_faaaa = nn.Parameter(tmp.unsqueeze(-1))

            elif self.version == 'v6':
                self.time_maa_w = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
                # v6版本的动态初始化

                D_LORA_W = 32  # Low-rank dimension for w
                self.w_A = nn.Parameter(torch.zeros(config.n_embd, D_LORA_W))
                self.w_B = nn.Parameter(torch.zeros(D_LORA_W, config.n_head).uniform_(-0.01, 0.01))

                # 初始化动态 time_decay

                decay_speed = torch.ones(config.n_head)
                for n in range(config.n_head):
                    decay_speed[n] = -6 + 5 * (n / (config.n_head - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
                self.time_decay = nn.Parameter(decay_speed.unsqueeze(-1))

                # 引入低秩参数用于动态调整 time_decay

                D_DECAY_LORA = 64
                self.time_decay_w1 = nn.Parameter(torch.zeros(config.n_embd, D_DECAY_LORA))
                self.time_decay_w2 = nn.Parameter(torch.zeros(D_DECAY_LORA, config.n_head).uniform_(-0.01, 0.01))

                # 初始化动态 time_faaaa

                tmp = torch.zeros(config.n_head)
                for n in range(config.n_head):
                    zigzag = ((n + 1) % 3 - 1) * 0.1
                    tmp[n] = ratio_0_to_1 * (1 - (n / (config.n_head - 1))) + zigzag
                self.time_faaaa = nn.Parameter(tmp.unsqueeze(-1))

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.receptance = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        nn.init.orthogonal_(self.receptance.weight, gain=1)

        self.key = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        nn.init.orthogonal_(self.key.weight, gain=0.1)

        self.value = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        nn.init.orthogonal_(self.value.weight, gain=1)

        self.gate = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        nn.init.orthogonal_(self.gate.weight, gain=0.1)

        self.output = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        nn.init.zeros_(self.output.weight)
        scale = ((1 + layer_id) / config.n_layer) ** 0.7
        self.ln_x = nn.GroupNorm(self.n_head, config.n_embd, eps=(1e-5)*144)
        # self.ln_x.weight.data.fill_(scale)
        # self.ln_x.bias.data.zero_()

        self.dropout = nn.Dropout(config.dropout)

    def lora(self, x, A, B):
        return torch.tanh(torch.matmul(x, A)) @ B

    def forward(self, x):
        B, T, C = x.size()
        H, N = self.n_head, self.head_size

        if T % 256 == 0:
            Q = 256
        elif T % 128 == 0:
            Q = 128
        else:
            Q = T
        assert T % Q == 0
        xx = self.time_shift(x) - x
        xk = x + xx * self.time_maa_k
        xv = x + xx * self.time_maa_v
        xr = x + xx * self.time_maa_r
        xg = x + xx * self.time_maa_g
        r = self.receptance(xr).view(B, T, H, N).transpose(1, 2)
        k = self.key(xk).view(B, T, H, N).permute(0, 2, 3, 1)
        v = self.value(xv).view(B, T, H, N).transpose(1, 2)
        g = F.silu(self.gate(xg))
        # Compute w and u based on version
        if self.version == 'v6':
            xw = x + xx * self.time_maa_w
            dynamic_w = self.lora(xw, self.w_A, self.w_B)
            w = torch.exp(-torch.exp(self.time_decay.float() + dynamic_w.float()))
            u = self.time_faaaa.float()

            # 显式状态更新 (wkv 计算)
            wk = w.view(B, H, T)
            wb = wk.transpose(-2, -1).flip(2)

            y = torch.empty(B, H, T, N, device=x.device, dtype=x.dtype)

            for t in range(T):
                y[:, :, t, :] = (r[:, :, t, :] @ state) * wb[:, t, :].unsqueeze(-1)
                state = w[:, t, :].unsqueeze(-1) * state + (k[:, :, :, t] @ v[:, :, t, :])

        else:  # version == 'v5'
            w = torch.exp(-torch.exp(self.time_decay).float())

            u = self.time_faaaa.float()
            ws = w.pow(Q).view(1, H, 1, 1)
            ind = torch.arange(Q - 1, -1, -1, device=r.device).unsqueeze(0).repeat(H, 1)
            w = w.repeat(1, Q).pow(ind)
            wk = w.view(1, H, 1, Q)
            wb = wk.transpose(-2, -1).flip(2)
            w = torch.cat([w[:, 1:], u], dim=1)
            w = F.pad(w, (0, Q))
            w = torch.tile(w, [Q])
            w = w[:, :-Q].view(-1, Q, 2 * Q - 1)
            w = w[:, :, Q - 1:].view(1, H, Q, Q)
            w = w.to(dtype=r.dtype)
            wk = wk.to(dtype=r.dtype)
            wb = wb.to(dtype=r.dtype)
            ws = ws.to(dtype=r.dtype)

            state = torch.zeros(B, H, N, N, device=r.device, dtype=r.dtype)
            y = torch.empty(B, H, T, N, device=r.device, dtype=r.dtype)

            for i in range(T // Q):
                rr = r[:, :, i * Q:i * Q + Q, :]
                kk = k[:, :, :, i * Q:i * Q + Q]
                vv = v[:, :, i * Q:i * Q + Q, :]
                y[:, :, i * Q:i * Q + Q, :] = ((rr @ kk) * w) @ vv + (rr @ state) * wb
                state = ws * state + (kk * wk) @ vv
            # state shape [bs, n_head, head_size, head_size]
        y = y.transpose(1, 2).contiguous().view(B * T, C)
        y = self.ln_x(y).view(B, T, C) * g
        y = self.dropout(self.output(y))
        return y

class RWKV_ChannelMix_x051a(nn.Module):
    def __init__(self, config, layer_id):
        super().__init__()

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        with torch.no_grad():
            ratio_1_to_almost0 = 1.0 - (layer_id / config.n_layer)
            ddd = torch.ones(1, 1, config.n_embd)
            for i in range(config.n_embd):
                ddd[0, 0, i] = i / config.n_embd
            self.time_maa_k = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_r = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))

        self.key = nn.Linear(config.n_embd, 2 * config.n_embd, bias=config.bias)
        nn.init.orthogonal_(self.key.weight, gain=0.1)
        self.value = nn.Linear(2 * config.n_embd, config.n_embd, bias=config.bias)
        nn.init.orthogonal_(self.value.weight, gain=1)
        self.receptance = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        nn.init.orthogonal_(self.receptance.weight, gain=1)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        xx = self.time_shift(x) - x
        xk = x + xx * self.time_maa_k
        xr = x + xx * self.time_maa_r

        x = self.key(xk)
        x = torch.relu(x) ** 2
        x = self.value(x)
        x = torch.sigmoid(self.receptance(xr)) * x
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self, config, layer_id):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.tmix = RWKV_TimeMix_x051a(config, layer_id)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.cmix = RWKV_ChannelMix_x051a(config, layer_id)

    def forward(self, x):
        x = x + self.tmix(self.ln_1(x))
        x = x + self.cmix(self.ln_2(x))
        return x

class RWKV(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.blocks = nn.ModuleList([Block(config, i) for i in range(config.n_layer)])
        self.ln_out = LayerNorm(config.n_embd)
        # self.head = nn.Linear(config.n_embd, config.n_embd / 2, bias=config.bias)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)

        x = self.ln_out(x)
        # x = self.head(x)
        return x
