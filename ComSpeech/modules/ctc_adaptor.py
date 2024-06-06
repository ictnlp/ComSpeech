import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torch_scatter

from fairseq import utils
from fairseq.data.data_utils import lengths_to_mask, lengths_to_padding_mask
from fairseq.models.speech_to_speech.modules.transformer_encoder import (
    TransformerEncoderNoEmb,
)
from fairseq.models.transformer import Embedding
from fairseq.modules import FairseqDropout
from fairseq.modules import PositionalEmbedding


def _uniform_assignment(src_lens, tgt_lens):
    # torch.set_printoptions(profile='full')
    tgt_indices = torch.arange(torch.max(tgt_lens)).expand(len(tgt_lens), -1).to(tgt_lens.device)
    ratio = tgt_lens / src_lens
    index_t = (tgt_indices / ratio.view(-1, 1)).long()
    return index_t


def build_embedding(dictionary, embed_dim):
    num_embeddings = len(dictionary)
    padding_idx = dictionary.pad()
    return Embedding(num_embeddings, embed_dim, padding_idx)


class CtcAdaptor(nn.Module):

    def __init__(self, args, src_dict, tgt_dict):
        super().__init__()
        self.tgt_dict = tgt_dict
        if getattr(args, "adaptor_encoder_layers", 0) > 0:
            self.adaptor_encoder = self.build_adaptor_encoder(args)
        else:
            self.adaptor_encoder = None 
        if getattr(args, "adaptor_decoder_layers", 0) > 0:
            self.adaptor_decoder = self.build_adaptor_decoder(args)
        else:
            self.adaptor_decoder = None
        self.input_proj = nn.Linear(args.decoder_embed_dim, args.decoder_embed_dim)
        self.embed_positions = PositionalEmbedding(
            args.max_target_positions, args.decoder_embed_dim, tgt_dict.pad()
        )
        self.src_embed_tokens = build_embedding(src_dict, args.decoder_embed_dim)
        self.tgt_embed_tokens = build_embedding(tgt_dict, args.decoder_embed_dim)
        self.embed_scale = math.sqrt(args.decoder_embed_dim)
        self.ctc_proj = nn.Linear(
                self.tgt_embed_tokens.weight.shape[1],
                self.tgt_embed_tokens.weight.shape[0],
                bias=False,
            )
        self.ctc_proj.weight = self.tgt_embed_tokens.weight
        self.upsample_ratio = args.ctc_upsample
        self.dropout_module = FairseqDropout(args.dropout)
        self.ctc_merge_type = args.ctc_merge_type

    def build_adaptor_encoder(self, args):
        _args = copy.deepcopy(args)
        _args.encoder_layers = args.adaptor_encoder_layers
        _args.encoder_embed_dim = args.decoder_embed_dim
        _args.encoder_ffn_embed_dim = args.decoder_ffn_embed_dim
        _args.encoder_attention_heads = args.decoder_attention_heads
        _args.encoder_normalize_before = True
        return TransformerEncoderNoEmb(_args)
    
    def build_adaptor_decoder(self, args):
        _args = copy.deepcopy(args)
        _args.encoder_layers = args.adaptor_decoder_layers
        _args.encoder_embed_dim = args.decoder_embed_dim
        _args.encoder_ffn_embed_dim = args.decoder_ffn_embed_dim
        _args.encoder_attention_heads = args.decoder_attention_heads
        _args.encoder_normalize_before = True
        return TransformerEncoderNoEmb(_args)

    def upsample(self, x, enc_padding_mask, tgt_phoneme):
        src_lens = (~enc_padding_mask).sum(dim=-1)
        up_lens = src_lens * self.upsample_ratio
        if tgt_phoneme is not None:
            tgt_lens = tgt_phoneme.ne(1).long().sum(dim=-1) 
            consecutive_equals = (tgt_phoneme[:, :-1] == tgt_phoneme[:, 1:]).long()
            consecutive_equals = consecutive_equals.cumsum(dim=-1)
            num_repeats = torch.gather(consecutive_equals, dim=1, index=(tgt_lens - 2).unsqueeze(-1)).squeeze(1)
            up_lens = torch.max(up_lens, tgt_lens + num_repeats)
        x = x.transpose(0, 1)  # B x T x C
        enc_padding_mask = lengths_to_padding_mask(up_lens)
        mapped_inputs = _uniform_assignment(src_lens, up_lens).masked_fill(
            enc_padding_mask, 0
        )
        copied_embedding = torch.gather(
            x,
            1,
            mapped_inputs.unsqueeze(-1).expand(
                *mapped_inputs.size(), x.size(-1)
            ),
        )
        return copied_embedding, enc_padding_mask
    
    def ctc_merge_features(self, x, index, probs, ctc_tgt_lens):
        # x: T x B x C, index: B x T, probs: B x T, ctc_tgt_lens: B x 1
        if self.ctc_merge_type == "softmax":
            probs = torch_scatter.composite.scatter_softmax(probs, index=index, dim=-1).type_as(x)
            weight_x = x.transpose(0, 1) * probs.unsqueeze(-1)
            merge_x = x.new_zeros((x.size(1), torch.max(ctc_tgt_lens) + 1, x.size(-1)))
            merge_x = merge_x.scatter_add(
                dim=1, 
                index=index.unsqueeze(-1).expand(index.size(0), index.size(1), x.size(-1)), 
                src=weight_x,
            )[:, 1:, :]
        elif self.ctc_merge_type == "meanpool":
            merge_x = x.new_zeros((x.size(1), torch.max(ctc_tgt_lens) + 1, x.size(-1)))
            merge_x = torch_scatter.scatter_mean(
                src=x.transpose(0, 1),
                dim=1, 
                index=index.unsqueeze(-1).expand(index.size(0), index.size(1), x.size(-1)), 
            )[:, 1:, :]
        elif self.ctc_merge_type == "maxpool":
            merge_x = x.new_zeros((x.size(1), torch.max(ctc_tgt_lens) + 1, x.size(-1)))
            merge_x = torch_scatter.scatter_max(
                src=x.transpose(0, 1),
                dim=1, 
                index=index.unsqueeze(-1).expand(index.size(0), index.size(1), x.size(-1)), 
            )[0][:, 1:, :]
        elif self.ctc_merge_type == "weighted":
            merge_x = x.new_zeros((x.size(1), torch.max(ctc_tgt_lens) + 1, x.size(-1)))
            merge_x = torch_scatter.scatter_mean(
                src=x.transpose(0, 1) * probs.unsqueeze(-1).type_as(x),
                dim=1, 
                index=index.unsqueeze(-1).expand(index.size(0), index.size(1), x.size(-1)), 
            )[:, 1:, :]
        return merge_x

    def compute_ctc_loss_and_align(self, x, logits, enc_padding_mask, ctc_tgt):
        # ctc loss
        ctc_lprobs = utils.log_softmax(logits.float(), dim=-1)
        ctc_lens = ctc_lprobs.new_full((ctc_lprobs.shape[1],), ctc_lprobs.shape[0]).long()
        ctc_lens -= enc_padding_mask.sum(dim=-1)
        ctc_tgt_lens = ctc_tgt.ne(1).long().sum(dim=-1)
        ctc_tgt_mask = lengths_to_mask(ctc_tgt_lens)
        ctc_tgt_flat = ctc_tgt.masked_select(ctc_tgt_mask)
        ctc_loss = F.ctc_loss(
            ctc_lprobs,
            ctc_tgt_flat,
            ctc_lens,
            ctc_tgt_lens,
            reduction="sum",
            zero_infinity=True,
        )
        ctc_lprobs = ctc_lprobs.transpose(0, 1)  # B x T
        all_alignments = ctc_lprobs.new_zeros((ctc_lprobs.size(0), ctc_lprobs.size(1))).long()
        for i in range(len(ctc_lprobs)):
            alignments, scores = torchaudio.functional.forced_align(
                ctc_lprobs[i, :ctc_lens[i]].unsqueeze(0).contiguous(),
                ctc_tgt[i, :ctc_tgt_lens[i]].unsqueeze(0),
            )
            all_alignments[i, :ctc_lens[i]] = alignments[0]
        all_alignments_mask = all_alignments.eq(0)
        diff = torch.diff(
            all_alignments,
            prepend=torch.zeros((len(ctc_lprobs), 1), device=all_alignments.device).fill_(-1),
        )
        diff = (diff != 0).masked_fill_(all_alignments_mask, 0)
        index = diff.cumsum(dim=1).masked_fill_(all_alignments_mask, 0)
        probs = torch.gather(ctc_lprobs, dim=-1, index=all_alignments.unsqueeze(-1)).squeeze(-1).exp()
        merge_x = self.ctc_merge_features(x, index, probs, ctc_tgt_lens)
        return ctc_loss, merge_x, ctc_tgt_lens

    def get_ctc_prediction(self, x, logits, enc_padding_mask):
        ctc_lprobs = utils.log_softmax(logits.float(), dim=-1)
        ctc_lens = ctc_lprobs.new_full((ctc_lprobs.shape[1],), ctc_lprobs.shape[0]).long()
        ctc_lens -= enc_padding_mask.sum(dim=-1)
        ctc_lprobs = ctc_lprobs.transpose(0, 1)  # B x T
        ctc_pred = ctc_lprobs.argmax(dim=-1).masked_fill_(enc_padding_mask, 0)  # B x T
        ctc_pred_mask = ctc_pred.eq(0)
        diff = torch.diff(
            ctc_pred,
            prepend=torch.zeros((len(ctc_lprobs), 1), device=ctc_pred.device).fill_(-1),
        )
        diff = (diff != 0).masked_fill_(ctc_pred_mask, 0)
        index = diff.cumsum(dim=1).masked_fill_(ctc_pred_mask, 0)
        ctc_tgt_lens = torch.max(index, dim=-1)[0]
        probs = torch.gather(ctc_lprobs, dim=-1, index=ctc_pred.unsqueeze(-1)).squeeze(-1).exp()
        merge_x = self.ctc_merge_features(x, index, probs, ctc_tgt_lens)
        return merge_x, ctc_tgt_lens, ctc_pred

    def get_ctc_input(self, x, enc_padding_mask, tgt_phoneme):
        x, enc_padding_mask = self.upsample(x, enc_padding_mask, tgt_phoneme)
        x = self.input_proj(x)
        x += self.embed_positions(enc_padding_mask)
        x = self.dropout_module(x)
        x = x.transpose(0, 1)  # T' x B x C

        return x, enc_padding_mask

    def forward_tgt_embedding(self, src_tokens):
        x = self.tgt_embed_tokens(src_tokens) * self.embed_scale
        x += self.embed_positions(src_tokens)
        x = self.dropout_module(x)
        return x

    def forward(self, x, enc_padding_mask, tgt_phoneme):
        if self.adaptor_encoder is not None:
            x = self.adaptor_encoder(
                x,
                enc_padding_mask,
            )
            x = x["encoder_out"][0]
        x, enc_padding_mask = self.get_ctc_input(x, enc_padding_mask, tgt_phoneme)
        if self.adaptor_decoder is not None:
            x = self.adaptor_decoder(
                x,
                enc_padding_mask,
            )
            x = x["encoder_out"][0]  # T' x B x C
        logits = self.ctc_proj(x)

        dp_loss, ctc_pred = None, None
        if tgt_phoneme is not None:
            dp_loss, x, out_lens = self.compute_ctc_loss_and_align(x, logits, enc_padding_mask, tgt_phoneme)
        else:
            x, out_lens, ctc_pred = self.get_ctc_prediction(x, logits, enc_padding_mask)
        new_enc_padding_mask = lengths_to_padding_mask(out_lens)
        x = x.transpose(0, 1)  # T * B * C
        
        out = {
            "x": x,
            "padding_mask": new_enc_padding_mask,
            "dp_loss": dp_loss,
            "ctc_pred": ctc_pred,
        }

        return out