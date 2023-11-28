"""Memsizer layer."""

from typing import Dict, List, Optional
import torch
import torch.nn as nn
from torch import Tensor
from fairseq import utils
from fairseq.modules import LayerNorm
from fairseq.modules.fairseq_dropout import FairseqDropout
from fairseq.modules.quant_noise import quant_noise
from src.attention import CausalAttention, CrossAttention
import math
import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--use-memsizer', action='store_true', help='use memsizer in both encoder and decoder.')
    parser.add_argument('--encoder-use-rfa', action='store_true', help='use memsizer in encoder.')
    parser.add_argument('--decoder-use-rfa', action='store_true', help='use memsizer in decoder.')
    parser.add_argument('--causal-proj-dim', type=int, default=4, help='the number of memory slots in causal attention.')
    parser.add_argument('--cross-proj-dim', type=int, default=32, help='the number of memory slots in non-causal attention.')

    parser.add_argument('--q-init-scale', type=float, default=8.0, help='init scale for \Phi.')
    parser.add_argument('--kv-init-scale', type=float, default=8.0, help='init scale for W_l and W_r.')
    parser.add_argument(
        "-f", action="store_true", help="None", default=False
    )
    parser.add_argument("--encoder_embed_dim", default=512, type=int, help="embedding_dim")
    parser.add_argument("--encoder_ffn_embed_dim", default=2048, type=int)
    parser.add_argument("--encoder_layers", default=4, type=int)
    parser.add_argument("--encoder_attention_heads", default=8, type=int, help="attention_heads")

    parser.add_argument("--decoder_embed_dim", default=512, type=int)
    parser.add_argument("--decoder_ffn_embed_dim", default=2048, type=int)
    parser.add_argument("--decoder_attention_heads", default=8, type=int, help="attention heads")
    parser.add_argument("--decoder_layers", default=4)

    parser.add_argument("--dropout", default=0.33, type=float, help="dropout")
    parser.add_argument('--attention-dropout', type=float, metavar='D', default=0,
                            help='dropout probability for attention weights')
    parser.add_argument('--encoder_normalize_before', action='store_false', default=True, help='apply layernorm before each encoder block')
    parser.add_argument('--decoder_normalize_before', action='store_false', default=True, help='apply layernorm before each encoder block')
    return parser.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
########
# Taken from:
# https://pytorch.org/tutorials/beginner/transformer_tutorial.html
# or also here:
# https://github.com/pytorch/examples/blob/master/word_language_model/model.py
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.0, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.max_len = max_len

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float()
                             * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # shape (max_len, 1, dim)
        self.register_buffer('pe', pe)  # Will not be trained.

    def forward(self, x):
        """Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        """
        assert x.size(0) < self.max_len, (
            f"Too long sequence length: increase `max_len` of pos encoding")
        # shape of x (len, B, dim)
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class MemsizerEncoderLayer(nn.Module):

    def __init__(
        self, args
    ):
        super().__init__()
        self.embed_dim = args.encoder_embed_dim
        self.num_heads = args.encoder_attention_heads
        self.head_dim = self.embed_dim // self.num_heads

        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )
        self.quant_noise = getattr(args, "quant_noise_pq", 0)
        self.quant_noise_block_size = getattr(args, "quant_noise_pq_block_size", 8)

        self.self_attn = self.build_self_attention(
            self.embed_dim,
            args
        )

        self.activation_fn = utils.get_activation_fn(
            activation=str(args.activation_fn)
            if getattr(args, "activation_fn", None) is not None
            else "relu"
        )
        activation_dropout_p = getattr(args, "activation_dropout", 0) or 0
        if activation_dropout_p == 0:
            # for backwards compatibility with models that use args.relu_dropout
            activation_dropout_p = getattr(args, "relu_dropout", 0) or 0
        self.activation_dropout_module = FairseqDropout(
            float(activation_dropout_p), module_name=self.__class__.__name__
        )
        self.normalize_before = args.encoder_normalize_before

        # use layerNorm rather than FusedLayerNorm for exporting.
        # char_inputs can be used to determint this.
        # TODO  remove this once we update apex with the fix
        export = getattr(args, "char_inputs", False)
        self.self_attn_layer_norm = LayerNorm(self.embed_dim, export=export)

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

        self.final_layer_norm = LayerNorm(self.embed_dim, export=export)
        self.need_attn = True

        self.onnx_trace = False

    def build_fc1(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)

    def build_fc2(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)

    def build_self_attention(
        self, embed_dim, args
    ):
        return CrossAttention(
            args=args,
            embed_dim=embed_dim,
            num_heads=self.num_heads,
            k_dim=args.cross_proj_dim,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
        )
    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def residual_connection(self, x, residual):
        return residual + x

    def forward(
        self,
        x,
        encoder_padding_mask,
        attn_mask: Optional[Tensor] = None,
    ):

        assert attn_mask is None

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)

        x = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=encoder_padding_mask,
            attn_mask=attn_mask
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

class MemsizerDecoderLayer(nn.Module):
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
        self, args, no_encoder_attn=False
    ):
        super().__init__()
        self.embed_dim = args.decoder_embed_dim
        self.num_heads = args.decoder_attention_heads
        self.head_dim = self.embed_dim // self.num_heads

        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )
        self.quant_noise = getattr(args, "quant_noise_pq", 0)
        self.quant_noise_block_size = getattr(args, "quant_noise_pq_block_size", 8)

        self.self_attn = self.build_self_attention(
            self.embed_dim,
            args
        )

        self.activation_fn = utils.get_activation_fn(
            activation=str(args.activation_fn)
            if getattr(args, "activation_fn", None) is not None
            else "relu"
        )
        activation_dropout_p = getattr(args, "activation_dropout", 0) or 0
        if activation_dropout_p == 0:
            # for backwards compatibility with models that use args.relu_dropout
            activation_dropout_p = getattr(args, "relu_dropout", 0) or 0
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

    def build_fc1(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)

    def build_fc2(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)

    def build_fc3(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)

    def build_self_attention(
        self, embed_dim, args
    ):
        return CausalAttention(
            args=args,
            embed_dim=embed_dim,
            num_heads=self.num_heads,
            k_dim=args.causal_proj_dim,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size
        )

    def build_encoder_attention(self, embed_dim, args):
        return CrossAttention(
            args=args,
            embed_dim=embed_dim,
            num_heads=self.num_heads,
            k_dim=args.cross_proj_dim,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
        )

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def residual_connection(self, x, residual):
        return residual + x

    def forward(
        self,
        x,
        encoder_out: Optional[Tensor] = None,
        encoder_padding_mask: Optional[Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        prev_self_attn_state: Optional[List[Tensor]] = None,
        prev_attn_state: Optional[List[Tensor]] = None,
        self_attn_mask: Optional[Tensor] = None,
        self_attn_padding_mask: Optional[Tensor] = None,
        need_attn = False,
        need_head_weights = False,
    ):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_state: s, z, random_matrices
            encoder_padding_mask (ByteTensor, optional): binary
                ByteTensor of shape `(batch, src_len)` where padding
                elements are indicated by ``1``.

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        if prev_self_attn_state is not None:
            prev_key, prev_value = prev_self_attn_state[:2]
            saved_state: Dict[str, Optional[Tensor]] = {
                "prev_key": prev_key,
                "prev_value": prev_value,
            }
            if len(prev_self_attn_state) >= 3:
                saved_state["prev_key_padding_mask"] = prev_self_attn_state[2]
            assert incremental_state is not None
            self.self_attn._set_input_buffer(incremental_state, saved_state)
        x = self.self_attn(
            x=x,
            key_padding_mask=self_attn_padding_mask,
            attn_mask=self_attn_mask,
            incremental_state=incremental_state
        )
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)
        if self.encoder_attn is not None:
            residual = x
            if self.normalize_before:
                x = self.encoder_attn_layer_norm(x)
            if prev_attn_state is not None:
                prev_key, prev_value = prev_attn_state[:2]
                saved_state: Dict[str, Optional[Tensor]] = {
                    "prev_key": prev_key,
                    "prev_value": prev_value,
                }
                if len(prev_attn_state) >= 3:
                    saved_state["prev_key_padding_mask"] = prev_attn_state[2]
                assert incremental_state is not None
                self.encoder_attn._set_input_buffer(incremental_state, saved_state)

            x = self.encoder_attn(
                query=x,
                key=encoder_out,
                value=encoder_out,
                key_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
            )

            x = self.dropout_module(x)
            x = self.residual_connection(x, residual)
            if not self.normalize_before:
                x = self.encoder_attn_layer_norm(x)

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
        return x, None, None


class Memsizer(nn.Module):
    def __init__(self, source_vocabulary_size, target_vocabulary_size,
                 d_model=512, pad_id=0, encoder_layers=4, decoder_layers=4,
                 dim_feedforward=2048, num_heads=8):
        # all arguments are (int)
        super().__init__()
        self.pad_id = pad_id

        self.embedding_src = nn.Embedding(source_vocabulary_size, d_model, padding_idx = pad_id)
        self.embedding_tgt = nn.Embedding(target_vocabulary_size, d_model, padding_idx = pad_id)

        self.pos_encoder = PositionalEncoding(d_model)
        args = parse_args()
        encoder_layer = MemsizerEncoderLayer(args)
        decoder_layer = MemsizerDecoderLayer(args)
        self.encoder = encoder_layer
        self.decoder = decoder_layer
        self.linear = nn.Linear(d_model, target_vocabulary_size)

    def create_src_padding_mask(self, src):
        # input src of shape ()
        src_padding_mask = src.transpose(0, 1) == 0
        return src_padding_mask

    def create_tgt_padding_mask(self, tgt):
        # input tgt of shape ()
        tgt_padding_mask = tgt.transpose(0, 1) == 0
        return tgt_padding_mask

    # Implement me!
    def greedy_decode(self, target, max_len, memory, memory_key_padding_mask):

      ys = torch.ones(1, 1).fill_(3).type_as(target.data).to(DEVICE)
      for i in range(max_len-1):
          tgt_key_padding_mask = self.create_tgt_padding_mask(ys).to(DEVICE)
          tgt_mask = (nn.Transformer.generate_square_subsequent_mask(ys.size(0))
                      .type(torch.bool)).to(DEVICE)
          tgt = self.embedding_tgt(ys)
          tgt = self.pos_encoder(tgt)

          out, _, _ = self.decoder(tgt, memory, self_attn_mask = tgt_mask, self_attn_padding_mask = tgt_key_padding_mask, encoder_padding_mask = memory_key_padding_mask)
          # shift the target by one
          out = out.transpose(0, 1)
          prob = self.linear(out[:, -1])

          _, next_word = torch.max(prob, dim=1)
          next_word = next_word.item()

          ys = torch.cat([ys, torch.ones(1, 1).type_as(target.data).fill_(next_word)], dim=0)

          # stop crieria 1
          if next_word == 2:
              break

      if ys.shape[0] < max_len:
        new_seq = ys.data.new(max_len, 1).fill_(0)
        new_seq[:ys.shape[0],:] = ys
        ys = new_seq
      return ys


    def greedy_search(self, src, tgt):
        src_key_padding_mask = self.create_src_padding_mask(src).to(DEVICE)
        out = self.embedding_src(src)
        out = self.pos_encoder(out)
        encoder_out = self.encoder(out, encoder_padding_mask = src_key_padding_mask)
        results = torch.ones(tgt.shape[1], tgt.shape[0]).type(torch.long).to(DEVICE)
        for i in range(encoder_out.shape[1]):
          memory = encoder_out[:,i,:].unsqueeze(dim = 1)
          memory_key_padding_mask = src_key_padding_mask[i,:].unsqueeze(dim = 0)
          result = self.greedy_decode(tgt[:,i], tgt.size()[0] + 1, memory, memory_key_padding_mask)
          result = result.permute(1,0)
          results[i,:] = result[:,1:]
        return results

    # Implement me!
    def forward(self, src, tgt):
        src_key_padding_mask = self.create_src_padding_mask(src).to(DEVICE)
        tgt_key_padding_mask = self.create_tgt_padding_mask(tgt).to(DEVICE)
        memory_key_padding_mask = src_key_padding_mask
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(
            tgt.shape[0]).to(DEVICE)

        tgt = self.embedding_tgt(tgt)
        tgt = self.pos_encoder(tgt)
        out = self.embedding_src(src)
        out = self.pos_encoder(out)

        encoder_out = self.encoder(out, encoder_padding_mask = src_key_padding_mask)
        decoder_out, _, _ = self.decoder(tgt, encoder_out, self_attn_mask = tgt_mask, self_attn_padding_mask = tgt_key_padding_mask, encoder_padding_mask = memory_key_padding_mask)

        out = self.linear(decoder_out)
        return out
    

class TransformerModel(nn.Module):
    def __init__(self, source_vocabulary_size, target_vocabulary_size,
                 d_model=512, pad_id=0, encoder_layers=4, decoder_layers=4,
                 dim_feedforward=2048, num_heads=8):
        # all arguments are (int)
        super().__init__()
        self.pad_id = pad_id

        self.embedding_src = nn.Embedding(source_vocabulary_size, d_model, padding_idx = pad_id)
        self.embedding_tgt = nn.Embedding(target_vocabulary_size, d_model, padding_idx = pad_id)

        self.pos_encoder = PositionalEncoding(d_model)
        self.transformer = nn.Transformer(
            d_model, num_heads, encoder_layers, decoder_layers, dim_feedforward)
        self.encoder = self.transformer.encoder
        self.decoder = self.transformer.decoder
        self.linear = nn.Linear(d_model, target_vocabulary_size)

    def create_src_padding_mask(self, src):
        # input src of shape ()
        src_padding_mask = src.transpose(0, 1) == 0
        return src_padding_mask

    def create_tgt_padding_mask(self, tgt):
        # input tgt of shape ()
        tgt_padding_mask = tgt.transpose(0, 1) == 0
        return tgt_padding_mask

    def greedy_decode(self, target, max_len, memory, memory_key_padding_mask):

      ys = torch.ones(1, 1).fill_(3).type_as(target.data).to(DEVICE)
      for i in range(max_len-1):
          tgt_key_padding_mask = self.create_tgt_padding_mask(ys).to(DEVICE)
          tgt_mask = (nn.Transformer.generate_square_subsequent_mask(ys.size(0))
                      .type(torch.bool)).to(DEVICE)
          tgt = self.embedding_tgt(ys)
          tgt = self.pos_encoder(tgt)

          out = self.decoder(tgt, memory, tgt_mask, tgt_key_padding_mask = tgt_key_padding_mask, memory_key_padding_mask = memory_key_padding_mask)
          # shift the target by one
          out = out.transpose(0, 1)
          prob = self.linear(out[:, -1])

          _, next_word = torch.max(prob, dim=1)
          next_word = next_word.item()

          ys = torch.cat([ys, torch.ones(1, 1).type_as(target.data).fill_(next_word)], dim=0)

          # stop crieria 1
          if next_word == 2:
              break

      if ys.shape[0] < max_len:
        new_seq = ys.data.new(max_len, 1).fill_(0)
        new_seq[:ys.shape[0],:] = ys
        ys = new_seq
      return ys


    def greedy_search(self, src, tgt):
        src_key_padding_mask = self.create_src_padding_mask(src).to(DEVICE)
        out = self.embedding_src(src)
        out = self.pos_encoder(out)
        encoder_out = self.encoder(out, src_key_padding_mask = src_key_padding_mask)
        results = torch.ones(tgt.shape[1], tgt.shape[0]).type(torch.long).to(DEVICE)
        for i in range(encoder_out.shape[1]):
          memory = encoder_out[:,i,:].unsqueeze(dim = 1)
          memory_key_padding_mask = src_key_padding_mask[i,:].unsqueeze(dim = 0)
          result = self.greedy_decode(tgt[:,i], tgt.size()[0] + 1, memory, memory_key_padding_mask)
          result = result.permute(1,0)
          results[i,:] = result[:,1:]
        return results

    def forward_separate(self, src, tgt):
        src_key_padding_mask = self.create_src_padding_mask(src).to(DEVICE)
        tgt_key_padding_mask = self.create_tgt_padding_mask(tgt).to(DEVICE)
        memory_key_padding_mask = src_key_padding_mask
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(
            tgt.shape[0]).to(DEVICE)

        tgt = self.embedding_tgt(tgt)
        tgt = self.pos_encoder(tgt)
        out = self.embedding_src(src)
        out = self.pos_encoder(out)

        encoder_out = self.encoder(out, src_key_padding_mask = src_key_padding_mask)
        decoder_out = self.decoder(tgt, encoder_out, tgt_mask = tgt_mask, tgt_key_padding_mask = tgt_key_padding_mask, memory_key_padding_mask = memory_key_padding_mask)

        out = self.linear(decoder_out)
        return out

    def forward(self, src, tgt):
        """Forward function.

        Parameters:
          src: tensor of shape (sequence_length, batch, data dim)
          tgt: tensor of shape (sequence_length, batch, data dim)
        Returns:
          tensor of shape (sequence_length, batch, data dim)
        """
        src_key_padding_mask = self.create_src_padding_mask(src).to(DEVICE)
        tgt_key_padding_mask = self.create_tgt_padding_mask(tgt).to(DEVICE)
        memory_key_padding_mask = src_key_padding_mask
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(
            tgt.shape[0]).to(DEVICE)

        tgt = self.embedding_tgt(tgt)
        tgt = self.pos_encoder(tgt)
        out = self.embedding_src(src)
        out = self.pos_encoder(out)
        out = self.transformer(
            out, tgt, src_key_padding_mask=src_key_padding_mask,
            tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask)
        out = self.linear(out)

        return out