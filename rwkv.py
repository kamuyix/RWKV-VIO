import os
import torch
import torch.nn as nn
from torch.nn import functional as F
import pytorch_lightning as pl
from torch.utils.cpp_extension import load

def __nop(ob):
    return ob

MyModule = nn.Module
MyFunction = __nop

HEAD_SIZE = int(os.environ.get("RWKV_HEAD_SIZE_A", 64))
wkv6_cuda = load(name="wkv6", sources=["cuda/wkv6_op.cpp", f"cuda/wkv6_cuda.cu"],
                 verbose=True, extra_cuda_cflags=["-res-usage", "--use_fast_math", "-O3", "-Xptxas -O3",
                                                  "--extra-device-vectorization", f"-D_N_={HEAD_SIZE}",
                                                  f"-D_T_={int(os.environ.get('RWKV_CTXLEN', 512))}"])

class WKV_6(torch.autograd.Function):
    @staticmethod
    def forward(ctx, B, T, C, H, r, k, v, w, u):
        with torch.no_grad():
            assert r.dtype == torch.bfloat16
            assert k.dtype == torch.bfloat16
            assert v.dtype == torch.bfloat16
            assert w.dtype == torch.bfloat16
            assert u.dtype == torch.bfloat16
            assert HEAD_SIZE == C // H
            ctx.B = B
            ctx.T = T
            ctx.C = C
            ctx.H = H
            assert r.is_contiguous()
            assert k.is_contiguous()
            assert v.is_contiguous()
            assert w.is_contiguous()
            assert u.is_contiguous()
            ew = (-torch.exp(w.float())).contiguous()
            ctx.save_for_backward(r, k, v, ew, u)
            y = torch.empty((B, T, C), device=r.device, dtype=torch.bfloat16, memory_format=torch.contiguous_format)
            wkv6_cuda.forward(B, T, C, H, r, k, v, ew, u, y)
            return y

    @staticmethod
    def backward(ctx, gy):
        with torch.no_grad():
            assert gy.dtype == torch.bfloat16
            B = ctx.B
            T = ctx.T
            C = ctx.C
            H = ctx.H
            assert gy.is_contiguous()
            r, k, v, ew, u = ctx.saved_tensors
            gr = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16,
                             memory_format=torch.contiguous_format)
            gk = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16,
                             memory_format=torch.contiguous_format)
            gv = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16,
                             memory_format=torch.contiguous_format)
            gw = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16,
                             memory_format=torch.contiguous_format)
            gu = torch.empty((B, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16,
                             memory_format=torch.contiguous_format)
            wkv6_cuda.backward(B, T, C, H, r, k, v, ew, u, gy, gr, gk, gv, gw, gu)
            gu = torch.sum(gu, 0).view(H, C // H)
            return (None, None, None, None, gr, gk, gv, gw, gu)

def RUN_CUDA_RWKV6(B, T, C, H, r, k, v, w, u):
    return WKV_6.apply(B, T, C, H, r, k, v, w.to(torch.bfloat16), u)

class RWKV_Tmix_x060(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id

        self.head_size = args.head_size_a
        self.n_head = args.dim_att // self.head_size
        assert args.dim_att % self.n_head == 0

        with torch.no_grad():
            ratio_0_to_1 = layer_id / (args.n_layer - 1)
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)
            ddd = torch.ones(1, 1, args.n_embd, device='cuda', dtype=torch.bfloat16)
            for i in range(args.n_embd):
                ddd[0, 0, i] = i / args.n_embd

            self.time_maa_x = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0)).to(torch.bfloat16)
            self.time_maa_w = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0)).to(torch.bfloat16)
            self.time_maa_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0)).to(torch.bfloat16)
            self.time_maa_v = nn.Parameter(1.0 - (torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1)).to(torch.bfloat16)
            self.time_maa_r = nn.Parameter(1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0)).to(torch.bfloat16)
            self.time_maa_g = nn.Parameter(1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0)).to(torch.bfloat16)

            D_MIX_LORA = 32
            self.time_maa_w1 = nn.Parameter(torch.zeros(args.n_embd, D_MIX_LORA * 5)).to('cuda', dtype=torch.bfloat16)
            self.time_maa_w2 = nn.Parameter(torch.zeros(5, D_MIX_LORA, args.n_embd, device='cuda', dtype=torch.bfloat16).uniform_(-0.01, 0.01))

            decay_speed = torch.ones(args.dim_att).to('cuda', dtype=torch.bfloat16)
            for n in range(args.dim_att):
                decay_speed[n] = -6 + 5 * (n / (args.dim_att - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
            self.time_decay = nn.Parameter(decay_speed.reshape(1, 1, args.dim_att)).to(torch.bfloat16)

            D_DECAY_LORA = 64
            self.time_decay_w1 = nn.Parameter(torch.zeros(args.n_embd, D_DECAY_LORA, device='cuda', dtype=torch.bfloat16))
            self.time_decay_w2 = nn.Parameter(torch.zeros(D_DECAY_LORA, args.dim_att, device='cuda', dtype=torch.bfloat16)).uniform_(-0.01, 0.01)

            tmp = torch.zeros(args.dim_att, device='cuda', dtype=torch.bfloat16)
            for n in range(args.dim_att):
                zigzag = ((n + 1) % 3 - 1) * 0.1
                tmp[n] = ratio_0_to_1 * (1 - (n / (args.dim_att - 1))) + zigzag

            self.time_faaaa = nn.Parameter(tmp.reshape(self.n_head, self.head_size))

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.receptance = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.receptance.weight.data = self.receptance.weight.data.to('cuda', dtype=torch.bfloat16)

        self.key = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.key.weight.data = self.key.weight.data.to('cuda', dtype=torch.bfloat16)

        self.value = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.value.weight.data = self.value.weight.data.to('cuda', dtype=torch.bfloat16)

        self.output = nn.Linear(args.dim_att, args.n_embd, bias=False)
        self.output.weight.data = self.output.weight.data.to('cuda', dtype=torch.bfloat16)

        self.gate = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.gate.weight.data = self.gate.weight.data.to('cuda', dtype=torch.bfloat16)

        self.ln_x = nn.GroupNorm(self.n_head, args.dim_att, eps=(1e-5) * (args.head_size_divisor ** 2))
        self.ln_x.weight.data = self.ln_x.weight.data.to('cuda', dtype=torch.bfloat16)
        self.ln_x.bias.data = self.ln_x.bias.data.to('cuda', dtype=torch.bfloat16)

    @MyFunction
    def jit_func(self, x):
        B, T, C = x.size()

        xx = self.time_shift(x) - x

        xxx = x + xx * self.time_maa_x
        xxx = torch.tanh(xxx @ self.time_maa_w1).view(B * T, 5, -1).transpose(0, 1)
        xxx = torch.bmm(xxx, self.time_maa_w2).view(5, B, T, -1)
        mw, mk, mv, mr, mg = xxx.unbind(dim=0)

        xw = (x + xx * (self.time_maa_w + mw)).to(torch.bfloat16)
        xk = (x + xx * (self.time_maa_k + mk)).to(torch.bfloat16)
        xv = (x + xx * (self.time_maa_v + mv)).to(torch.bfloat16)
        xr = (x + xx * (self.time_maa_r + mr)).to(torch.bfloat16)
        xg = (x + xx * (self.time_maa_g + mg)).to(torch.bfloat16)

        r = self.receptance(xr).to(torch.bfloat16)
        k = self.key(xk).to(torch.bfloat16)
        v = self.value(xv).to(torch.bfloat16)
        g = F.silu(self.gate(xg)).to(torch.bfloat16)

        ww = torch.tanh(xw @ self.time_decay_w1) @ self.time_decay_w2
        w = self.time_decay + ww

        return r, k, v, g, w

    @MyFunction
    def jit_func_2(self, x, g):
        B, T, C = x.size()
        x = x.view(B * T, C)

        x = self.ln_x(x).view(B, T, C)
        x = self.output(x * g)
        return x

    def forward(self, x):
        B, T, C = x.size()
        H = self.n_head

        r, k, v, g, w = self.jit_func(x.to('cuda', dtype=torch.bfloat16))
        x = RUN_CUDA_RWKV6(B, T, C, H, r, k, v, w, u=self.time_faaaa.to('cuda', dtype=torch.bfloat16))

        return self.jit_func_2(x, g)

class RWKV_CMix_x060(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        with torch.no_grad():
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)
            ddd = torch.ones(1, 1, args.n_embd, device='cuda')
            for i in range(args.n_embd):
                ddd[0, 0, i] = i / args.n_embd
            self.time_maa_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0)).to(torch.bfloat16)
            self.time_maa_r = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0)).to(torch.bfloat16)

        self.key = nn.Linear(args.n_embd, args.dim_ffn, bias=False)
        self.key.weight.data = self.key.weight.data.to('cuda', dtype=torch.bfloat16)

        self.receptance = nn.Linear(args.n_embd, args.n_embd, bias=False)
        self.receptance.weight.data = self.receptance.weight.data.to('cuda', dtype=torch.bfloat16)

        self.value = nn.Linear(args.dim_ffn, args.n_embd, bias=False)
        self.value.weight.data = self.value.weight.data.to('cuda', dtype=torch.bfloat16)

    def forward(self, x):
        xx = self.time_shift(x) - x
        xk = x + xx * self.time_maa_k
        xr = x + xx * self.time_maa_r

        k = self.key(xk.to(torch.bfloat16))
        k = torch.relu(k) ** 2
        kv = self.value(k).to(torch.bfloat16)
        return torch.sigmoid(self.receptance(xr.to(torch.bfloat16))) * kv

class Block(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id

        self.ln1 = nn.LayerNorm(args.n_embd)
        self.ln1.weight.data = self.ln1.weight.data.to(device='cuda', dtype=torch.bfloat16)
        self.ln1.bias.data = self.ln1.bias.data.to(device='cuda', dtype=torch.bfloat16)

        self.ln2 = nn.LayerNorm(args.n_embd)
        self.ln2.weight.data = self.ln2.weight.data.to(device='cuda', dtype=torch.bfloat16)
        self.ln2.bias.data = self.ln2.bias.data.to(device='cuda', dtype=torch.bfloat16)

        if self.layer_id == 0:
            self.ln0 = nn.LayerNorm(args.n_embd)
            self.ln0.weight.data = self.ln0.weight.data.to(device='cuda', dtype=torch.bfloat16)
            self.ln0.bias.data = self.ln0.bias.data.to(device='cuda', dtype=torch.bfloat16)

        self.att = RWKV_Tmix_x060(args, layer_id)
        self.ffn = RWKV_CMix_x060(args, layer_id)

        if args.dropout > 0:
            self.drop0 = nn.Dropout(p=args.dropout)
            self.drop1 = nn.Dropout(p=args.dropout)

    def forward(self, x):
        if self.layer_id == 0:
            x = self.ln0(x.to(torch.bfloat16))

        x = x + self.att(self.ln1(x).to(torch.bfloat16))
        x = x + self.ffn(self.ln2(x).to(torch.bfloat16))

        return x

class L2Wrap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, loss, y):
        ctx.save_for_backward(y)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        y = ctx.saved_tensors[0]
        factor = 1e-4 / (y.shape[0] * y.shape[1])
        maxx, ids = torch.max(y, -1, keepdim=True)
        gy = torch.zeros_like(y)
        gy.scatter_(-1, ids, maxx * factor)
        return (grad_output, gy)

class RWKV(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.blocks = nn.ModuleList([Block(args, i) for i in range(args.n_layer)])
        self.ln_out = nn.LayerNorm(args.n_embd)
        self.ln_out.weight.data = self.ln_out.weight.data.to('cuda', dtype=torch.bfloat16)
        self.ln_out.bias.data = self.ln_out.bias.data.to('cuda', dtype=torch.bfloat16)

        self.head = nn.Linear(args.n_embd, args.rwkv_out_size, bias=False)
        self.head.weight.data = self.head.weight.data.to('cuda', dtype=torch.bfloat16)

        if args.dropout > 0:
            self.drop0 = nn.Dropout(p=args.dropout)

    def configure_optimizers(self):
        trainable_params = [p for p in self.parameters() if p.requires_grad]
        optim_groups = [{"params": trainable_params, "weight_decay": self.args.weight_decay}]
        if self.deepspeed_offload:
            return DeepSpeedCPUAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps,
                                    bias_correction=True, adamw_mode=True, amsgrad=False)
        return FusedAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps,
                         bias_correction=True, adam_w_mode=True, amsgrad=False)

    @property
    def deepspeed_offload(self) -> bool:
        strategy = self.trainer.strategy
        if isinstance(strategy, DeepSpeedStrategy):
            cfg = strategy.config["zero_optimization"]
            return cfg.get("offload_optimizer") or cfg.get("offload_param")
        return False

    def forward(self, x):
        args = self.args

        if args.dropout > 0:
            x = self.drop0(x)

        for block in self.blocks:
            if args.grad_cp == 1:
                x = deepspeed.checkpointing.checkpoint(block, x.to('cuda', dtype=torch.bfloat16))
            else:
                x = block(x.to('cuda', dtype=torch.bfloat16))

        x = self.ln_out(x.to('cuda', dtype=torch.bfloat16))
        x = self.head(x.to('cuda', dtype=torch.bfloat16))

        return x

    def training_step(self, batch, batch_idx):
        idx, targets = batch
        logits = self(idx)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return L2Wrap.apply(loss, logits)

    def training_step_end(self, batch_parts):
        if pl.__version__[0] != '2':
            all = self.all_gather(batch_parts)
            if self.trainer.is_global_zero:
                self.trainer.my_loss_all = all
