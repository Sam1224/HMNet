import sys
import time
import math
from functools import partial
from typing import Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref


class Mlp(nn.Module):
    """ Multilayer perceptron."""
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super(Mlp, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
    
class SS2D(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        ratio=4,
        layer_idx=0,
        d_conv=3,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        dropout=0.,
        conv_bias=True,
        bias=False,
        device=None,
        dtype=None,
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.ratio = ratio  # downsample/window size
        self.layer_idx = layer_idx
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj_q = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.in_proj_s = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        
        self.conv2d_q = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.conv2d_s = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        
        self.act = nn.SiLU()
        
        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs)
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0)) # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0)) # (K=4, inner)
        del self.dt_projs
        
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True) # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True) # (K=4, D, N)

        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        
        self.out_proj_q = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.out_proj_s = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True
        
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def window_partition(self, x, ratio=4):
        """
        Args:
            x: (B, H, W, C)
            ratio (int): Downsample ratio, default is 4
        Returns:
            windows: (B, ratio^2, size^2, C)
        """
        B, H, W, C = x.shape
        assert H == W, "Feature height should be consistent with its weight."
        size = H // ratio
        x = x.view(B, ratio, size, ratio, size, C)  # B, ratio (H), size (H), ratio (W), size (W), C
        
        windows_row = x.permute(0, 1, 3, 2, 4, 5).contiguous()  # B, ratio (H), ratio (W), size (H), size (W), C
        windows_col = x.permute(0, 3, 1, 4, 2, 5).contiguous()  # B, ratio (W), ratio (H), size (W), size (H), C
        
        windows_row = windows_row.view(B, ratio ** 2, size ** 2, C)  # B, ratio^2 (H then W), size^2 (H then W), C
        windows_col = windows_col.view(B, ratio ** 2, size ** 2, C)  # B, ratio^2 (W then H), size^2 (W then H), C
        return windows_row, windows_col

    def window_reverse(self, windows, ratio=4, H=64, W=64, mode="row"):
        """
        Args:
            windows: (B, ratio^2, size^2, C)
            ratio (int): Downsample ratio, default is 4
            H (int): Height of image
            W (int): Width of image
            mode (str): "row" or "col"
        Returns:
            x: (B, H, W, C)
        """
        assert mode in ["row", "col"], "mode should be either 'row' or 'col'."
        B, _, _, C = windows.size()
        assert H == W, "Feature height should be consistent with its weight."
        size = H // ratio
        x = windows.view(B, ratio, ratio, size, size, C)  # B, ratio (H/W), ratio (W/H), size (H/W), size (W/H), C
        if mode == "row":
            x = x.permute(0, 1, 3, 2, 4, 5).contiguous()  # B, ratio (H), size (H), ratio (W), size (W), C
        else:
            x = x.permute(0, 2, 4, 1, 3, 5).contiguous()  # B, ratio (H), size (H), ratio (W), size (W), C
        x = x.view(B, H, W, -1)
        return x

    def self_mamba(self, x):
        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (b, 4, d, l)
        
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)

        xs = xs.float().view(B, -1, L) # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1) # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)

        out_y = self.selective_scan(
            xs, dts, 
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)  # (b, 4, c, l)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        
        return (out_y[:, 0], inv_y[:, 0], wh_y, invwh_y)
    
    def mix_mamba(self, x_q_idt, x_q_dwn, x_s_dwn):
        # x_q_idt: (b, c, 64, 64)
        # x_q_dwn: (b, c, 16, 16)
        # x_s_dwn: (b, c, 16, 16)
        B, C, H, W = x_q_idt.shape
        K = 4  # 4 direction
        ratio = self.ratio  # downsample/window size
        size = H // ratio
        L_S = size ** 2
        L = H * W * 2 # q: h * w; s: h/4 * w/4

        # feature reshape
        x_q_idt = x_q_idt.permute(0, 2, 3, 1).contiguous()  # (b, 64, 64, d)
        x_s_dwn = x_s_dwn.permute(0, 2, 3, 1).contiguous()  # (b, 16, 16, d)
        
        # ========================================
        # split features into sequences
        # ========================================
        x_q_row_idt, x_q_col_idt = self.window_partition(x_q_idt, ratio=ratio)  # (b, ratio^2=4*4, size^2=16*16, d)
        x_s_row_dwn = x_s_dwn.view(B, 1, -1, C)  # (b, 1, 16*16, d)
        x_s_col_dwn = x_s_dwn.permute(0, 2, 1, 3).contiguous().view(B, 1, -1, C)  # (b, 1, 16*16, d)

        # ========================================
        # sequences with original ordering
        # ========================================
        # original ordering sequences
        x_q_ori_idt = torch.stack([x_q_row_idt, x_q_col_idt], dim=1)  # (b, 2, ratio^2, size^2, d)
        x_s_ori_dwn = torch.stack([x_s_row_dwn, x_s_col_dwn], dim=1)  # (b, 2, 1, size^2, d)
        
        # ========================================
        # sequences with inverse ordering
        # ========================================
        # inverse query idt sequences
        x_q_inv_idt = x_q_ori_idt.view(B, 2, -1, C)
        x_q_inv_idt = torch.flip(x_q_inv_idt, dims=[-2])  # inverse ordering
        x_q_inv_idt = x_q_inv_idt.view(B, 2, ratio ** 2, size ** 2, C)  # (b, 4*4, 16*16, d)

        # inverse support dwn sequences
        x_s_inv_dwn = x_s_ori_dwn
        x_s_inv_dwn = torch.flip(x_s_inv_dwn, dims=[-2])  # inverse ordering
        
        # ========================================
        # support recap
        # ========================================
        # repeat x_s_ori_dwn and x_s_inv_dwn
        x_s_ori_dwn = x_s_ori_dwn.repeat(1, 1, ratio ** 2, 1, 1).contiguous()  # (b, 2, 4*4, 16*16, d)
        x_s_inv_dwn = x_s_inv_dwn.repeat(1, 1, ratio ** 2, 1, 1).contiguous()  # (b, 2, 4*4, 16*16, d)
        
        # concat support and query patches
        xs_ori = torch.cat([x_s_ori_dwn, x_q_ori_idt], dim=-2)  # (b, 2, 4*4, 16*16*2, d)
        xs_inv = torch.cat([x_s_inv_dwn, x_q_inv_idt], dim=-2)  # (b, 2, 4*4, 16*16*2, d)
        
        # ========================================
        # 4 direction mamba
        # ========================================
        # reshape
        xs_ori = xs_ori.view(B, 2, -1, C)  # (b, 1, 4*4*16*16*2, d)
        xs_inv = xs_inv.view(B, 2, -1, C)  # (b, 1, 4*4*16*16*2, d)
        
        # merge two orderings
        xs = torch.cat([xs_ori, xs_inv], dim=1) # (b, k=4, l, d)
        xs = xs.permute(0, 1, 3, 2).contiguous()  # (b, k, d, l)
        assert xs.size(-1) == L, "Please double check the length of the input sequence."

        # ========================================
        # 3 mamba scans
        # 1st: self mamba for support - scan support for obtaining h_s
        # 2nd: cross mamba
        # 3rd: mix mamba
        # ========================================
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)

        xs = xs.float().view(B, -1, L) # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1) # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)
        
        # ########################################
        # 1st scan - self mamba
        # ########################################
        with torch.no_grad():
            xs_s = xs[:, :, :L_S].contiguous()
            dts_s = dts[:, :, :L_S].contiguous()
            Bs_s = Bs[:, :, :, :L_S].contiguous()
            Cs_s = Cs[:, :, :, :L_S].contiguous()

            _, h_s = self.selective_scan(
                xs_s, dts_s, 
                As, Bs_s, Cs_s, Ds, z=None,
                delta_bias=dt_projs_bias,
                delta_softplus=True,
                return_last_state=True
            )
        
        # ########################################
        # 2nd scan - mix mamba
        # ########################################
        out_y = self.selective_scan(
            xs, dts, 
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        )
        out_y = out_y.view(B, K, -1, L)  # (b, k, d, l)
        assert out_y.dtype == torch.float
        
        # ========================================
        # sequence split
        # 0: original ordering + row
        # 1: original ordering + col
        # 2: inverse ordering + row
        # 3: inverse ordering + col
        # ========================================
        out_ori_row, out_ori_col = out_y[:, 0], out_y[:, 1]  # (b, d, l=4*4*16*16*2)
        out_inv_row, out_inv_col = out_y[:, 2], out_y[:, 3]
        
        # ========================================
        # separate query and support
        # ========================================
        # original ordering
        out_s_ori_row = out_ori_row[:, :, :L_S].contiguous()  # (b, d, 16*16)
        out_s_ori_col = out_ori_col[:, :, :L_S].contiguous()
        
        out_ori_row = out_ori_row.view(B, C, ratio ** 2, -1)  # (b, d, 4*4, 16*16*2)
        out_ori_col = out_ori_col.view(B, C, ratio ** 2, -1)
        
        out_q_ori_row = out_ori_row[:, :, :, L_S:].contiguous()  # (b, d, 4*4, 16*16)
        out_q_ori_col = out_ori_col[:, :, :, L_S:].contiguous()
        
        out_q_ori_row = out_q_ori_row.view(B, C, -1)
        out_q_ori_col = out_q_ori_col.view(B, C, -1)
        
        # inverse ordering
        out_s_inv_row = out_inv_row[:, :, :L_S].contiguous()  # (b, d, 16*16)
        out_s_inv_col = out_inv_col[:, :, :L_S].contiguous()
        
        out_inv_row = out_inv_row.view(B, C, ratio ** 2, -1)  # (b, d, 4*4, 16*16*2)
        out_inv_col = out_inv_col.view(B, C, ratio ** 2, -1)
        
        out_q_inv_row = out_inv_row[:, :, :, L_S:].contiguous()  # (b, d, 4*4, 16*16)
        out_q_inv_col = out_inv_col[:, :, :, L_S:].contiguous()
        
        out_q_inv_row = out_q_inv_row.view(B, C, -1)
        out_q_inv_col = out_q_inv_col.view(B, C, -1)
        
        # reverse inverse sequences into original ordering
        out_s_inv_row = torch.flip(out_s_inv_row, dims=[-1])  # (b, d, 16*16)
        out_s_inv_col = torch.flip(out_s_inv_col, dims=[-1])
        out_q_inv_row = torch.flip(out_q_inv_row, dims=[-1])  # (b, d, 4*4*16*16)
        out_q_inv_col = torch.flip(out_q_inv_col, dims=[-1])
        
        # recover query sequence back to patches
        out_q_ori_row = out_q_ori_row.permute(0, 2, 1).contiguous()  # (b, l, d)
        out_q_ori_col = out_q_ori_col.permute(0, 2, 1).contiguous()
        
        out_q_ori_row = out_q_ori_row.view(B, ratio ** 2, size ** 2, C)  # (b, ratio^2, size^2, d)
        out_q_ori_col = out_q_ori_col.view(B, ratio ** 2, size ** 2, C)
        
        out_q_ori_row = self.window_reverse(out_q_ori_row, ratio=ratio, H=H, W=W, mode="row")  # (b, h, w, d)
        out_q_ori_col = self.window_reverse(out_q_ori_col, ratio=ratio, H=H, W=W, mode="col")
        
        out_q_inv_row = out_q_inv_row.permute(0, 2, 1).contiguous()  # (b, l, d)
        out_q_inv_col = out_q_inv_col.permute(0, 2, 1).contiguous()
        
        out_q_inv_row = out_q_inv_row.view(B, ratio ** 2, size ** 2, C)  # (b, ratio^2, size^2, d)
        out_q_inv_col = out_q_inv_col.view(B, ratio ** 2, size ** 2, C)
        
        out_q_inv_row = self.window_reverse(out_q_inv_row, ratio=ratio, H=H, W=W, mode="row")  # (b, h, w, d)
        out_q_inv_col = self.window_reverse(out_q_inv_col, ratio=ratio, H=H, W=W, mode="col")
        
        # query feature reshape
        out_q_ori_row = out_q_ori_row.view(B, -1, C)  # (b, l, d)
        out_q_ori_col = out_q_ori_col.view(B, -1, C)
        
        out_q_ori_row = out_q_ori_row.permute(0, 2, 1).contiguous()  # (b, d, l)
        out_q_ori_col = out_q_ori_col.permute(0, 2, 1).contiguous()
        
        out_q_inv_row = out_q_inv_row.view(B, -1, C)  # (b, l, d)
        out_q_inv_col = out_q_inv_col.view(B, -1, C)
        
        out_q_inv_row = out_q_inv_row.permute(0, 2, 1).contiguous()  # (b, d, l)
        out_q_inv_col = out_q_inv_col.permute(0, 2, 1).contiguous()
        
        # recover support sequence back to patches
        out_s_ori_row = out_s_ori_row  # no change required
        out_s_inv_row = out_s_inv_row
        
        out_s_ori_col = out_s_ori_col.view(B, C, size, size)  # (b, d, w, h)
        out_s_inv_col = out_s_inv_col.view(B, C, size, size)
        
        out_s_ori_col = out_s_ori_col.permute(0, 1, 3, 2).contiguous()  # (b, d, h, w)
        out_s_inv_col = out_s_inv_col.permute(0, 1, 3, 2).contiguous()
        
        out_s_ori_col = out_s_ori_col.view(B, C, -1)  # (b, d, l)
        out_s_inv_col = out_s_inv_col.view(B, C, -1)
        
        # ########################################
        # 3rd scan - cross mamba
        # ########################################
        h_s = h_s.view(B, 4, C, -1).mean(1)  # (b, c, d_state)
        xs = x_q_dwn

        L = size ** 2
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, 1, -1, L), self.x_proj_weight[0:1].contiguous())
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, 1, -1, L), self.dt_projs_weight[0:1].contiguous())

        xs = xs.float().view(B, -1, L) # (b, d, l)
        dts = dts.contiguous().float().view(B, -1, L) # (b, d, l)
        Bs = Bs.float().view(B, 1, -1, L) # (b, 1, d_state, l)
        Cs = Cs.float().view(B, 1, -1, L) # (b, 1, d_state, l)
        Ds = self.Ds.float()[0].contiguous().view(-1) # (d)
        As = -torch.exp(self.A_logs.float().view(K, C, -1)[0].contiguous())  # (d, d_state)
        dt_projs_bias = self.dt_projs_bias.float()[0].contiguous().view(-1) # (d)

        out_y = self.selective_scan_cross(
            h_s, xs, dts, 
            As, Bs, Cs, Ds,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
        ).view(B, 1, -1, L)  # (b, 1, c, l)
        assert out_y.dtype == torch.float
        out_y_q = out_y[:, 0]
        
        return (out_q_ori_row, out_q_ori_col, out_q_inv_row, out_q_inv_col, out_y_q), (out_s_ori_row, out_s_ori_col, out_s_inv_row, out_s_inv_col)
    
    def selective_scan_cross(self, h, u, delta, A, B, C, D, delta_bias=None, delta_softplus=False):
        """
        h: (bs, k*d, d_state) => last hidden state from support
        u: (bs, k*d, l) => input query feature
        delta: (bs, k*d, l) => delta for discrete variables
        A: (k*d, d_state)  => variable A
        B: (bs, k, d_state, l) => variable B
        C: (bs, k, d_state, l) => variable C
        D: (k*d) => variable D
        delta_bias: (k*d) => delta bias
        """
        dtype_in = u.dtype
        h = h.float()
        u = u.float()
        
        # ========================================
        # Process inputs for discretization
        # ========================================
        delta = delta.float()
        if delta_bias is not None:
            delta = delta + delta_bias[..., None].float()
        if delta_softplus:
            delta = F.softplus(delta)
        batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]
        is_variable_B = B.dim() >= 3  # True
        is_variable_C = C.dim() >= 3  # True
        if A.is_complex():  # False
            if is_variable_B:
                B = torch.view_as_complex(rearrange(B.float(), "... (L two) -> ... L two", two=2))
            if is_variable_C:
                C = torch.view_as_complex(rearrange(C.float(), "... (L two) -> ... L two", two=2))
        else:
            B = B.float()
            C = C.float()
        
        # ========================================
        # \bar{A}
        # ========================================
        deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))  # (bs, k*d, l, d_state)
        
        # ========================================
        # \bar{B}u
        # ========================================
        if not is_variable_B:  # False
            deltaB_u = torch.einsum('bdl,dn,bdl->bdln', delta, B, u)
        else:
            if B.dim() == 3:
                deltaB_u = torch.einsum('bdl,bnl,bdl->bdln', delta, B, u)
            else:
                B = repeat(B, "B G N L -> B (G H) N L", H=dim // B.shape[1])  # (bs, k*d, d_state, l)
                deltaB_u = torch.einsum('bdl,bdnl,bdl->bdln', delta, B, u)  # (bs, k*d, l, d_state)
        
        # ========================================
        # \bar{C}
        # ========================================
        if is_variable_C and C.dim() == 4:
            C = repeat(C, "B G N L -> B (G H) N L", H=dim // C.shape[1])  # (bs, k*d, d_state, l)
            
        # ========================================
        # Scan over l (sequence length) dimension
        # ========================================
        # without for loop
        C_deltaA_h = torch.einsum("bdnl,bdln,bdn->bdl", C, deltaA, h)
        C_deltaB_u = torch.einsum("bdnl,bdln->bdl", C, deltaB_u)
        y = C_deltaA_h + C_deltaB_u
        
        # ys = []
        # for i in range(u.shape[2]):
        #     # ==============================
        #     # x = \bar{A}x + \bar{B}u
        #     # ==============================
        #     x = deltaA[:, :, i] * h + deltaB_u[:, :, i]  # (bs, k*d, d_state)
        #     if not is_variable_C:  # False
        #         y = torch.einsum('bdn,dn->bd', x, C)
        #     else:
        #         if C.dim() == 3:  # False
        #             y = torch.einsum('bdn,bn->bd', x, C[:, :, i])
        #         else:
        #             # ==============================
        #             # y = \bar{C}x
        #             # ==============================
        #             y = torch.einsum('bdn,bdn->bd', x, C[:, :, :, i])  # (bs, k*d)
        #     if y.is_complex():
        #         y = y.real * 2
        #     ys.append(y)
        # y = torch.stack(ys, dim=2) # (bs, k*d, l)
        # ==============================
        # y = \bar{C}x + \bar{D}u
        # ==============================
        out = y + u * rearrange(D, "d -> d 1")
        out = out.to(dtype=dtype_in)
        return out
    
    def forward_core(self, x_q, x_s):
        # x_q: (b, c, 64, 64)
        # x_s: (b, c, 64, 64)
        B, C, H, W = x_q.size()
        
        if self.layer_idx % 2 == 0:
            # ========================================
            # Self Mamba for Query and Support
            # ========================================
            (q1, q2, q3, q4) = self.self_mamba(x_q)
            (s1, s2, s3, s4) = self.self_mamba(x_s)
            y_q = q1 + q2 + q3 + q4
            y_s = s1 + s2 + s3 + s4
        else:
            # ========================================
            # Mix Mamba for Query
            # ========================================
            h, w = H // self.ratio, W // self.ratio
            
            x_q_idt = x_q
            x_q_dwn = F.interpolate(x_q, size=(h, w), mode='bilinear', align_corners=True)
            x_s_dwn = F.interpolate(x_s, size=(h, w), mode='bilinear', align_corners=True)
            
            (q1, q2, q3, q4, q5), (s1, s2, s3, s4) = self.mix_mamba(x_q_idt, x_q_dwn, x_s_dwn)  # (b, c, 64*64)
            y_q_idt = q1 + q2 + q3 + q4
            y_s_dwn = s1 + s2 + s3 + s4
            y_q_dwn = q5
            
            # interpolation
            y_q_dwn = rearrange(y_q_dwn, 'b c (h w) -> b c h w', h=h, w=w)
            y_s_dwn = rearrange(y_s_dwn, 'b c (h w) -> b c h w', h=h, w=w)
            
            y_q_dwn = F.interpolate(y_q_dwn, size=(H, W), mode='bilinear', align_corners=True)
            y_s_dwn = F.interpolate(y_s_dwn, size=(H, W), mode='bilinear', align_corners=True)
            
            y_q_dwn = rearrange(y_q_dwn, 'b c h w -> b c (h w)')
            y_s_dwn = rearrange(y_s_dwn, 'b c h w -> b c (h w)')
            
            # fuse
            y_q = y_q_idt + y_q_dwn
            y_s = y_s_dwn
        
        return y_q, y_s
        
    def forward(self, q_feat, s_feat):
        # q_feat: (b, 64, 64, c)
        # s_feat: (b, 64, 64, c)
        B, H, W, C = q_feat.shape

        # input projection
        xz_q = self.in_proj_q(q_feat)
        xz_s = self.in_proj_s(s_feat)
        
        x_q, z_q = xz_q.chunk(2, dim=-1)  # (b, 64, 64, d=2c)
        x_s, z_s = xz_s.chunk(2, dim=-1)  # (b, 64, 64, d=2c)

        # depth-wise convolution
        x_q = x_q.permute(0, 3, 1, 2).contiguous()
        x_s = x_s.permute(0, 3, 1, 2).contiguous()
        
        x_q = self.act(self.conv2d_q(x_q))  # (b, d, 64, 64)
        x_s = self.act(self.conv2d_s(x_s))  # (b, d, 64, 64)
        
        # mamba core
        y_q, y_s = self.forward_core(x_q, x_s)
        
        y_q = torch.transpose(y_q, dim0=1, dim1=2).contiguous().view(B, H, W, -1)  # (b, 64, 64, c)
        y_s = torch.transpose(y_s, dim0=1, dim1=2).contiguous().view(B, H, W, -1)  # (b, 64, 64, c)
        
        # layer norm
        y_q = self.out_norm(y_q)
        y_s = self.out_norm(y_s)
        
        # silu activation
        y_q = y_q * F.silu(z_q)
        y_s = y_s * F.silu(z_s)
        
        # output projection
        out_q = self.out_proj_q(y_q)
        out_s = self.out_proj_s(y_s)
        if self.dropout is not None:
            out_q = self.dropout(out_q)
            out_s = self.dropout(out_s)
        return out_q, out_s


class VSSBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0,
        d_state: int = 16,
        ratio: int = 4,
        layer_idx: int = 0,
        mlp_ratio: int = 1,
        **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS2D(d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state, ratio=ratio, layer_idx=layer_idx, **kwargs)
        
        self.drop_path = DropPath(drop_path)
        self.ln_2 = norm_layer(hidden_dim)
        
        self.mlp_q = Mlp(in_features=hidden_dim, hidden_features=hidden_dim * mlp_ratio)
        self.mlp_s = Mlp(in_features=hidden_dim, hidden_features=hidden_dim * mlp_ratio)

    def forward(self, q_feat, s_feat):
        q_skip = q_feat  # (b, 64, 64, c)
        s_skip = s_feat  # (b, 64, 64, c)
        
        q_feat, s_feat = self.self_attention(self.ln_1(q_feat), self.ln_1(s_feat))
        
        q_feat = q_skip + self.drop_path(q_feat)  # (b, 64, 64, c)
        s_feat = s_skip + self.drop_path(s_feat)  # (b, 64, 64, c)
        
        q_feat = q_feat + self.drop_path(self.mlp_q(self.ln_2(q_feat)))
        s_feat = s_feat + self.drop_path(self.mlp_s(self.ln_2(s_feat)))
        return q_feat, s_feat


class VSSLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
    """

    def __init__(
        self, 
        dim, 
        depth, 
        attn_drop=0.,
        drop_path=0., 
        norm_layer=nn.LayerNorm, 
        d_state=16,
        mlp_ratio=1,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        
        self.ratio = 4
        self.blocks = nn.ModuleList([
            VSSBlock(
                hidden_dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                attn_drop_rate=attn_drop,
                d_state=d_state,
                ratio=self.ratio,
                layer_idx=i,
                mlp_ratio=mlp_ratio
            )
            for i in range(depth)])
        
        if True: # is this really applied? Yes, but been overriden later in VSSM!
            def _init_weights(module: nn.Module):
                for name, p in module.named_parameters():
                    if name in ["out_proj.weight"]:
                        p = p.clone().detach_() # fake init, just to keep the seed ....
                        nn.init.kaiming_uniform_(p, a=math.sqrt(5))
            self.apply(_init_weights)

    def forward(self, q_feat, s_feat):
        # q_feat: (b, h, w, c)
        # s_feat: (b, h, w, c)
        # mamba block
        for blk in self.blocks:
            q_feat, s_feat = blk(q_feat, s_feat)
        return q_feat, s_feat


class VSSM(nn.Module):
    def __init__(self,
                 depths=[8], 
                 dims=[256],
                 mlp_ratio=1,
                 d_state=16,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=nn.LayerNorm, **kwargs):
        super().__init__()
        self.num_layers = len(depths)
        if isinstance(dims, int):
            dims = [int(dims * 2 ** i_layer) for i_layer in range(self.num_layers)]
        self.embed_dim = dims[0]
        self.num_features = dims[-1]
        self.dims = dims

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = VSSLayer(
                dim=dims[i_layer],
                depth=depths[i_layer],
                d_state=math.ceil(dims[0] / 6) if d_state is None else d_state, # 20240109
                mlp_ratio=mlp_ratio,
                drop=drop_rate, 
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer
            )
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        """
        out_proj.weight which is previously initilized in VSSBlock, would be cleared in nn.Linear
        no fc.weight found in the any of the model parameters
        no nn.Embedding found in the any of the model parameters
        so the thing is, VSSBlock initialization is useless
        
        Conv2D is not intialized !!!
        """
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward(self, q_feat, s_feat):
        q_feat = rearrange(q_feat, 'b c h w -> b h w c')
        s_feat = rearrange(s_feat, 'b c h w -> b h w c')
        q_feat = self.pos_drop(q_feat)
        s_feat = self.pos_drop(s_feat)

        for layer in self.layers:
            q_feat, s_feat = layer(q_feat, s_feat)
            
        q_feat = self.norm(q_feat)
        q_feat = rearrange(q_feat, 'b h w c -> b c h w')

        return q_feat
