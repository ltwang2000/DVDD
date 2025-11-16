# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional

import torch
import torch.nn as nn
from fairseq import utils
from fairseq.modules import LayerNorm, MultiheadAttention
from fairseq.modules.fairseq_dropout import FairseqDropout
from fairseq.modules.quant_noise import quant_noise
from torch import Tensor
from fairseq.modules import TransformerEncoderLayer
import torch.nn.functional as F
from .router import TopkRouter

class DyTransformerEncoderLayer(TransformerEncoderLayer):
    """Encoder layer block.

    In the original paper each operation (multi-head attention or FFN) is
    postprocessed with: `dropout -> add residual -> layernorm`. In the
    tensor2tensor code they suggest that learning is more robust when
    preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.encoder_normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
    """

    # MyTransformerEncoderLayer 的 __init__ 方法
    def __init__(self, args):
        super().__init__(args)
        self.embed_dim = args.encoder_embed_dim
        self.quant_noise = getattr(args, "quant_noise_pq", 0)
        self.quant_noise_block_size = getattr(args, "quant_noise_pq_block_size", 8)
        self.self_attn = self.build_self_attention(self.embed_dim, args)
        self.self_attn_layer_norm = LayerNorm(self.embed_dim)
        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )
        self.activation_fn = utils.get_activation_fn(
            activation=getattr(args, "activation_fn", "relu")
        )
        activation_dropout_p = getattr(args, "activation_dropout", 0)
        if activation_dropout_p == 0:
            # for backwards compatibility with models that use args.relu_dropout
            activation_dropout_p = getattr(args, "relu_dropout", 0)
        self.activation_dropout_module = FairseqDropout(
            float(activation_dropout_p), module_name=self.__class__.__name__
        )
        self.normalize_before = args.encoder_normalize_before
        self.fc1 = self.build_fc1(
            self.embed_dim,
            args.encoder_ffn_embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )
        self.fc2 = self.build_fc2(
            args.encoder_ffn_embed_dim,
            self.embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )

        self.final_layer_norm = LayerNorm(self.embed_dim)

    def build_fc1(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(
            nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size
        )

    def build_fc2(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(
            nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size
        )

    def build_self_attention(self, embed_dim, args):
        return MultiheadAttention(
            embed_dim,
            args.encoder_attention_heads,
            dropout=args.attention_dropout,
            self_attention=True,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
        )

    def residual_connection(self, x, residual):
        return residual + x

    def upgrade_state_dict_named(self, state_dict, name):
        """
        Rename layer norm states from `...layer_norms.0.weight` to
        `...self_attn_layer_norm.weight` and `...layer_norms.1.weight` to
        `...final_layer_norm.weight`
        """
        layer_norm_map = {"0": "self_attn_layer_norm", "1": "final_layer_norm"}
        for old, new in layer_norm_map.items():
            for m in ("weight", "bias"):
                k = "{}.layer_norms.{}.{}".format(name, old, m)
                if k in state_dict:
                    state_dict["{}.{}.{}".format(name, new, m)] = state_dict[k]
                    del state_dict[k]

    def forward(self, x, encoder_padding_mask, attn_mask: Optional[Tensor] = None):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, seq_len)` where padding elements are indicated by ``1``.
            attn_mask (ByteTensor): binary tensor of shape `(tgt_len, src_len)`,
                where `tgt_len` is the length of output and `src_len` is the
                length of input, though here both are equal to `seq_len`.
                `attn_mask[tgt_i, src_j] = 1` means that when calculating the
                embedding for `tgt_i`, we exclude (mask out) `src_j`. This is
                useful for strided self-attention.

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        # anything in original attn_mask = 1, becomes -1e8
        # anything in original attn_mask = 0, becomes 0
        # Note that we cannot use -inf here, because at some edge cases,
        # the attention weight (before softmax) for some padded element in query
        # will become -inf, which results in NaN in model parameters
        if attn_mask is not None:
            attn_mask = attn_mask.masked_fill(attn_mask.to(torch.bool), -1e8)

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        x, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=encoder_padding_mask,
            attn_mask=attn_mask,
        )
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)

        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        return x


class DyTransformerDecoderLayer(nn.Module):
    """Decoder layer block.

    In the original paper each operation (multi-head attention, encoder
    attention or FFN) is postprocessed with: `dropout -> add residual ->
    layernorm`. In the tensor2tensor code they suggest that learning is more
    robust when preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.decoder_normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(
        self, args, no_encoder_attn=False, add_bias_kv=False, add_zero_attn=False
    ):
        super().__init__()
        self.embed_dim = args.decoder_embed_dim
        self.d_model = 512
        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )
        self.quant_noise = getattr(args, "quant_noise_pq", 0)
        self.quant_noise_block_size = getattr(args, "quant_noise_pq_block_size", 8)

        self.cross_self_attention = getattr(args, "cross_self_attention", False)

        self.self_attn = self.build_self_attention(
            self.embed_dim,
            args,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
        )

        self.activation_fn = utils.get_activation_fn(
            activation=str(args.activation_fn)
            if getattr(args, "activation_fn", None) is not None
            else "relu"
        )
        activation_dropout_p = getattr(args, "activation_dropout", 0)
        if activation_dropout_p == 0:
            # for backwards compatibility with models that use args.relu_dropout
            activation_dropout_p = getattr(args, "relu_dropout", 0)
        self.activation_dropout_module = FairseqDropout(
            float(activation_dropout_p), module_name=self.__class__.__name__
        )
        self.normalize_before = args.decoder_normalize_before

        # use layerNorm rather than FusedLayerNorm for exporting.
        # char_inputs can be used to determint this.
        # TODO  remove this once we update apex with the fix
        export = getattr(args, "char_inputs", False)
        self.self_attn_layer_norm = LayerNorm(self.embed_dim, export=export)

        if no_encoder_attn:
            self.encoder_attn = None
            self.encoder_attn_layer_norm = None
        else:
            self.encoder_attn = self.build_encoder_attention(self.embed_dim, args)
            self.encoder_attn_layer_norm = LayerNorm(self.embed_dim, export=export)

        self.fc1 = self.build_fc1(
            self.embed_dim,
            args.decoder_ffn_embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )
        self.fc2 = self.build_fc2(
            args.decoder_ffn_embed_dim,
            self.embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )

        self.final_layer_norm = LayerNorm(self.embed_dim, export=export)
        self.need_attn = True

        self.onnx_trace = False

        #extra code
        self.cross_modal_layer_norm = LayerNorm(self.embed_dim)

        # 实例化两个独立的路由器（新增可调的 warmup/k_max，默认更保守且更快打开）
        self.region_router = TopkRouter(
            input_dim=self.embed_dim,
            k=getattr(args, "region_router_k", 4),
            temperature=getattr(args, "router_temp", 0.07),  # 0.07 -> 0.06
            k_warmup_steps=getattr(args, "router_k_warmup", 4000),  # 新增：默认 2000
            k_max=getattr(args, "region_router_k_max", None)
        )
        self.grid_router = TopkRouter(
            input_dim=self.embed_dim,
            k=getattr(args, "grid_router_k", 4),
            temperature=getattr(args, "router_temp", 0.07),
            k_warmup_steps=getattr(args, "router_k_warmup", 4000),
            k_max=getattr(args, "grid_router_k_max", None)
        )
        self.topk_router = TopkRouter(
            input_dim=self.embed_dim,
            k=getattr(args, "router_k_base", 4),
            temperature=getattr(args, "router_temp", 0.07),
            k_warmup_steps=getattr(args, "router_k_warmup", 4000),
            k_max=getattr(args, "router_k_max", None)
        )

        # 两层门控 + 视觉预 LN
        self.cross_modal_mix_gate = nn.Linear(self.embed_dim, 1)
        self.cross_modal_gate = nn.Linear(self.embed_dim, 1)
        self.vis_pre_ln = nn.LayerNorm(self.embed_dim)

        # 小残差缩放（保持可学习，但把初值加大到 0.6 更“有存在感”）
        self.vis_res_scale = nn.Parameter(torch.tensor(0.6))  # 0.1 -> 0.6

        # 跨模态 dropout
        self.cross_modal_dropout = nn.Dropout(0.1)
        self.attention_dropout = 0.1

        # （可选）三路权重门控占位，不在 forward 里强制使用
        self.visual_importance = nn.Linear(self.embed_dim, 3)

        # 额外：FFN 残差缩放系数（forward 会读取）
        self.ffn_alpha = 0.5

        self.reset_parameters()


    def reset_parameters(self):
        # 让模型一开始“少用图像”，训练学会何时打开
        if hasattr(self.cross_modal_gate, "bias") and self.cross_modal_gate.bias is not None:
            nn.init.constant_(self.cross_modal_gate.bias, -1.5)  # -3/-2 -> -1.5 更温和
        if hasattr(self.cross_modal_mix_gate, "bias") and self.cross_modal_mix_gate.bias is not None:
            nn.init.constant_(self.cross_modal_mix_gate.bias, 0.0)  # 融合门保持中性

        # （保留/复用你已有的 LN 等，不再次重复定义）
        self.vis_grid_ln = nn.LayerNorm(self.embed_dim)
        self.vis_region_ln = nn.LayerNorm(self.embed_dim)
        self.cross_modal_attn_layer_norm = nn.LayerNorm(self.embed_dim)

        # 如需 cross_modal_attn，可保留；当前 forward 未直接用它
        self.cross_modal_attn = MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.embed_dim,  # 维持你原设（虽不常见，但不动结构）
            dropout=self.attention_dropout,
            encoder_decoder_attention=True
        )
        self.layer_drop = nn.Dropout(0.1)
        #消融1
        self.vis_pre_ln = LayerNorm(self.embed_dim)  # 保留，用于稳定视觉特征
        #双门控消融
        #self.cross_modal_fuse = nn.Linear(2 * self.embed_dim, self.embed_dim)
        #跨模态门消融

    def build_attention_module(self, embed_dim, args, is_cross_attention=False):
        """一个辅助函数，用于构建各种注意力模块"""
        return MultiheadAttention(
            embed_dim,
            args.decoder_attention_heads,
            dropout=args.attention_dropout,
            encoder_decoder_attention=is_cross_attention,
        )


    def build_fc1(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)

    def build_fc2(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)

    def build_self_attention(
        self, embed_dim, args, add_bias_kv=False, add_zero_attn=False
    ):
        return MultiheadAttention(
            embed_dim,
            args.decoder_attention_heads,
            dropout=args.attention_dropout,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            self_attention=not getattr(args, "cross_self_attention", False),
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
        )

    def build_cross_attention(self, embed_dim, args):
        return MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=args.decoder_attention_heads,
            dropout=args.attention_dropout,
            encoder_decoder_attention=True
            # 重点：不要在这里设置 kdim 和 vdim，因为我们的Q, K, V维度相同
        )

    def build_encoder_attention(self, embed_dim, args):
        return MultiheadAttention(
            embed_dim,
            args.decoder_attention_heads,
            kdim=getattr(args, "encoder_embed_dim", None),
            vdim=getattr(args, "encoder_embed_dim", None),
            dropout=args.attention_dropout,
            encoder_decoder_attention=True,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
        )

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def residual_connection(self, x, residual):
        return residual + x

    def _cast_for_ln(self, x, ln):
        # 使输入与 LayerNorm 参数 dtype 对齐（通常是 float32）
        target_dtype = ln.weight.dtype
        return x.to(target_dtype) if x.dtype != target_dtype else x

    def forward(
            self,
            x,
            encoder_out: Optional[torch.Tensor] = None,
            encoder_padding_mask: Optional[torch.Tensor] = None,
            incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
            grid_feats: Optional[torch.Tensor] = None,
            region_feats: Optional[torch.Tensor] = None,
            grid_img_mask: Optional[torch.Tensor] = None,
            region_img_mask: Optional[torch.Tensor] = None,
            prev_self_attn_state: Optional[List[torch.Tensor]] = None,
            prev_attn_state: Optional[List[torch.Tensor]] = None,
            self_attn_mask: Optional[torch.Tensor] = None,
            self_attn_padding_mask: Optional[torch.Tensor] = None,
            need_attn: bool = False,
            need_head_weights: bool = False,
            return_vis: bool = False,
    ):
        if need_head_weights:
            need_attn = True

        # 用于可视化的容器
        vis_extra: Dict[str, torch.Tensor] = {}

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)

        # === self attention ===
        if prev_self_attn_state is not None:
            prev_key, prev_value = prev_self_attn_state[:2]
            saved_state = {"prev_key": prev_key, "prev_value": prev_value}
            if len(prev_self_attn_state) >= 3:
                saved_state["prev_key_padding_mask"] = prev_self_attn_state[2]
            assert incremental_state is not None
            self.self_attn._set_input_buffer(incremental_state, saved_state)

        y = x if not (self.cross_self_attention and encoder_out is not None) else torch.cat((encoder_out, x), dim=0)

        x, attn = self.self_attn(
            query=x,
            key=y,
            value=y,
            key_padding_mask=self_attn_padding_mask,
            incremental_state=incremental_state,
            need_weights=False,
            attn_mask=self_attn_mask,
        )
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        # === encoder attention ===
        if self.encoder_attn is not None and encoder_out is not None:
            residual = x
            if self.normalize_before:
                x = self.encoder_attn_layer_norm(x)
            if prev_attn_state is not None:
                prev_key, prev_value = prev_attn_state[:2]
                saved_state = {"prev_key": prev_key, "prev_value": prev_value}
                if len(prev_attn_state) >= 3:
                    saved_state["prev_key_padding_mask"] = prev_attn_state[2]
                assert incremental_state is not None
                self.encoder_attn._set_input_buffer(incremental_state, saved_state)
            x, attn = self.encoder_attn(
                query=x,
                key=encoder_out,
                value=encoder_out,
                key_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
                static_kv=True,
                need_weights=need_attn or (not self.training and self.need_attn),
                need_head_weights=need_head_weights,
            )
            x = self.dropout_module(x)
            x = self.residual_connection(x, residual)
            if not self.normalize_before:
                x = self.encoder_attn_layer_norm(x)

            # --- 跨模态注意力子层优化 ---
        # 修改后的视觉处理部分
        # --- 跨模态注意力子层（带 router） ---
        if (grid_feats is not None) and (region_feats is not None):
            residual = x
            x_norm = self.cross_modal_attn_layer_norm(x)  # [T,B,D]

            grid = self.vis_pre_ln(grid_feats)  # [B,Ng,D]
            region = self.vis_pre_ln(region_feats)  # [B,Nr,D]

            # 注意：对应 TopkRouter 的返回顺序
            # pooled, topk_vals, topk_idx, weights
            grid_pooled, grid_topk_vals, grid_topk_idx, grid_topk_w = self.grid_router(
                query=x_norm, values=grid, mask=grid_img_mask, return_candidates=True
            )
            region_pooled, region_topk_vals, region_topk_idx, region_topk_w = self.region_router(
                query=x_norm, values=region, mask=region_img_mask, return_candidates=True
            )

            mix_gate = torch.sigmoid(self.cross_modal_mix_gate(x_norm))  # [T,B,1]
            x_vis = mix_gate * region_pooled + (1.0 - mix_gate) * grid_pooled

            gate = torch.sigmoid(self.cross_modal_gate(x_norm))  # [T,B,1]
            x = residual + self.cross_modal_dropout(self.vis_res_scale * (gate * x_vis))

            if return_vis:
                vis_extra["mix_gate"] = mix_gate.detach().cpu()
                vis_extra["final_gate"] = gate.detach().cpu()
                # 真正的索引和权重要存的是下面这两个
                vis_extra["grid_cands_idx"] = grid_topk_idx.detach().cpu()  # [T,B,K], int
                vis_extra["grid_cands_score"] = grid_topk_w.detach().cpu()  # [T,B,K], float
                vis_extra["region_cands_idx"] = region_topk_idx.detach().cpu()
                vis_extra["region_cands_score"] = region_topk_w.detach().cpu()

            # ===== FFN 子层，保持原逻辑 =====
        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)
        x_ffn = self.activation_fn(self.fc1(x))
        x_ffn = self.activation_dropout_module(x_ffn)
        x_ffn = self.fc2(x_ffn)
        x = residual + self.dropout_module(x_ffn)
        if not self.normalize_before:
            x = self.final_layer_norm(x)

        # onnx trace 分支保持原样（三返回值）
        if self.onnx_trace and incremental_state is not None:
            saved_state = self.self_attn._get_input_buffer(incremental_state)
            if self_attn_padding_mask is not None:
                self_attn_state = [
                    saved_state["prev_key"],
                    saved_state["prev_value"],
                    saved_state["prev_key_padding_mask"],
                ]
            else:
                self_attn_state = [saved_state["prev_key"], saved_state["prev_value"]]
            return x, attn, self_attn_state

        # 普通分支：根据 return_vis 决定返回 3 个还是 4 个
        if return_vis:
            return x, attn, None, vis_extra
        else:
            return x, attn, None

def make_generation_fast_(self, need_attn: bool = False, **kwargs):
    self.need_attn = need_attn


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m
