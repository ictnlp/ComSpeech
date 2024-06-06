import torch
import logging
from ast import literal_eval
from collections import OrderedDict

from fairseq import checkpoint_utils
from fairseq.models import register_model, register_model_architecture
from fairseq.models.transformer import Embedding, TransformerDecoder
from fairseq.models.speech_to_text import (
    S2TConformerModel,
    conformer_base_architecture,
)

from ComSpeech.models.fastspeech2_style import FastSpeech2StyleEncoder
from ComSpeech.modules.ctc_adaptor import CtcAdaptor

logger = logging.getLogger(__name__)


def load_pretrained_component(component, state_dict, component_name):
    component_state_dict = OrderedDict()
    for key in state_dict.keys():
        if key.startswith(component_name):
            # encoder.input_layers.0.0.weight --> input_layers.0.0.weight
            component_subkey = key[len(component_name) + 1 :]
            component_state_dict[component_subkey] = state_dict[key]
    component.load_state_dict(component_state_dict, strict=True)
    logger.info(f"Successfully loaded pretrained weights for {component_name}")


@register_model("s2s_conformer_fastspeech2")
class S2SConformerFastSpeech2Model(S2TConformerModel):

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)

        def build_embedding(dictionary, embed_dim):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            return Embedding(num_embeddings, embed_dim, padding_idx)
        
        decoder_embed_tokens = build_embedding(
            task.target_dictionary, args.decoder_embed_dim
        )
        args.tgt_dict_size = len(task.target_dictionary)
        encoder = cls.build_encoder(args)
        decoder = cls.build_decoder(args, task, decoder_embed_tokens)

        base_model = cls(encoder, decoder)
        if getattr(args, "load_pretrained_s2t_from", None):
            state_dict = checkpoint_utils.load_checkpoint_to_cpu(args.load_pretrained_s2t_from)["model"]
            load_pretrained_component(base_model.encoder, state_dict, "encoder")
            load_pretrained_component(base_model.decoder, state_dict, "decoder")
        base_model.st_freezing_updates = getattr(args, "st_freezing_updates", 0)
        base_model.num_updates = 0
        
        base_model.tts = FastSpeech2StyleEncoder(
            args, 
            task.target_dictionary_tts, 
            args.max_target_positions, 
            embed_tokens=None,
        )
        if getattr(args, "load_pretrained_fastspeech_from", None):
            state_dict = checkpoint_utils.load_checkpoint_to_cpu(args.load_pretrained_fastspeech_from)["model"]
            load_pretrained_component(base_model.tts, state_dict, "tts")
        
        base_model.multiscale_modeling = args.multiscale_modeling
        if args.multiscale_modeling:
            base_model.w2p_adaptor = CtcAdaptor(args, task.target_dictionary, task.target_dictionary_tts)
            if getattr(args, "load_pretrained_ctc_from", None):
                state_dict = checkpoint_utils.load_checkpoint_to_cpu(args.load_pretrained_ctc_from)["model"]
                load_pretrained_component(base_model.w2p_adaptor, state_dict, "w2p_adaptor")
            logger.info(f"Build word-to-phoneme adaptor for multiscale modeling.")

        return base_model

    @classmethod
    def build_decoder(cls, args, task, embed_tokens):
        return TransformerDecoder(args, task.target_dictionary, embed_tokens)
    
    @staticmethod
    def add_args(parser):
        S2TConformerModel.add_args(parser)
        # Load pretrained models
        parser.add_argument(
            "--load-pretrained-s2t-from",
            type=str,
            help="path to pretrained s2t model",
        )
        parser.add_argument(
            "--load-pretrained-ctc-from",
            type=str,
            help="path to pretrained ctc model",
        )
        parser.add_argument(
            "--load-pretrained-fastspeech-from",
            type=str,
            help="path to pretrained fastspeech2 model",
        )
        # Word-to-Phone adaptor
        parser.add_argument("--adaptor-encoder-layers", type=int)
        parser.add_argument("--adaptor-decoder-layers", type=int)
        parser.add_argument("--ctc-upsample", type=int, default=1)
        parser.add_argument("--ctc-merge-type", type=str, default="softmax", choices=["softmax", "meanpool", "maxpool", "weighted"])
        # FastSpeech 2
        parser.add_argument("--output-frame-dim", type=int)
        # FFT blocks
        parser.add_argument("--fft-hidden-dim", type=int)
        parser.add_argument("--fft-kernel-size", type=int)
        parser.add_argument("--tts-input-embed-dim", type=int)
        parser.add_argument("--tts-encoder-layers", type=int)
        parser.add_argument("--tts-encoder-embed-dim", type=int)
        parser.add_argument("--tts-encoder-attention-heads", type=int)
        parser.add_argument("--tts-decoder-layers", type=int)
        parser.add_argument("--tts-decoder-embed-dim", type=int)
        parser.add_argument("--tts-decoder-attention-heads", type=int)
        # variance predictor
        parser.add_argument("--var-pred-n-bins", type=int)
        parser.add_argument("--var-pred-hidden-dim", type=int)
        parser.add_argument("--var-pred-kernel-size", type=int)
        parser.add_argument("--var-pred-dropout", type=float)
        # postnet
        parser.add_argument("--add-postnet", action="store_true")
        parser.add_argument("--postnet-dropout", type=float)
        parser.add_argument("--postnet-layers", type=int)
        parser.add_argument("--postnet-conv-dim", type=int)
        parser.add_argument("--postnet-conv-kernel-size", type=int)

    def forward_st(self, src_tokens, src_lengths, prev_output_tokens):
        st_decoder_out = super().forward(src_tokens, src_lengths, prev_output_tokens)
        x = st_decoder_out[1]["inner_states"][-1]
        if self.decoder.layer_norm is not None:
            x = self.decoder.layer_norm(x)
        x = x.transpose(0, 1)
        st_decoder_padding_mask = prev_output_tokens.eq(self.decoder.padding_idx)
        return x, st_decoder_padding_mask, st_decoder_out
    
    def forward_w2p_adaptor(self, x, st_decoder_padding_mask, tgt_phoneme=None):
        x = x.transpose(0, 1)
        adaptor_out = self.w2p_adaptor(x, st_decoder_padding_mask, tgt_phoneme)
        x = adaptor_out["x"]
        st_decoder_padding_mask = adaptor_out["padding_mask"]
        x = x.transpose(0, 1)
        return x, st_decoder_padding_mask, adaptor_out
    
    def forward(self, src_tokens, src_lengths, prev_output_tokens, tgt_phoneme=None):
        # 1. ST encoder and decoder
        x, st_decoder_padding_mask, st_decoder_out = self.forward_st(src_tokens, src_lengths, prev_output_tokens)
        
        # 2. Word-to-phone adaptor
        if self.multiscale_modeling:
            x, st_decoder_padding_mask, adaptor_out = self.forward_w2p_adaptor(x, st_decoder_padding_mask, tgt_phoneme)

        # 3. TTS encoder and decoder
        x, _ = self.tts.forward_encoder(x, st_decoder_padding_mask)
        tts_out = self.tts.forward_variance_adaptor_and_decoder(x, st_decoder_padding_mask)

        decoder_out = {
            "st_decoder_out": st_decoder_out,
            "adaptor_out": adaptor_out,
            "tts_out": tts_out,
        }

        return decoder_out
    
    def set_num_updates(self, num_updates):
        super().set_num_updates(num_updates)
        self.num_updates = num_updates


@register_model_architecture("s2s_conformer_fastspeech2", "s2s_conformer_fastspeech2")
def base_architecture(args):
    conformer_base_architecture(args)
    # Word-to-Phone adaptor
    args.adaptor_encoder_layers = getattr(args, "adaptor_encoder_layers", 0)
    args.adaptor_decoder_layers = getattr(args, "adaptor_decoder_layers", 0)
    # FastSpeech 2
    args.output_frame_dim = getattr(args, "output_frame_dim", 80)
    # FFT blocks
    args.fft_hidden_dim = getattr(args, "fft_hidden_dim", 1024)
    args.fft_kernel_size = getattr(args, "fft_kernel_size", 9)
    args.tts_input_embed_dim = getattr(args, "tts_input_embed_dim", 512)
    args.tts_encoder_layers = getattr(args, "tts_encoder_layers", 4)
    args.tts_encoder_embed_dim = getattr(args, "tts_encoder_embed_dim", 256)
    args.tts_encoder_attention_heads = getattr(args, "tts_encoder_attention_heads", 2)
    args.tts_decoder_layers = getattr(args, "tts_decoder_layers", 4)
    args.tts_decoder_embed_dim = getattr(args, "tts_decoder_embed_dim", 256)
    args.tts_decoder_attention_heads = getattr(args, "tts_decoder_attention_heads", 2)
    # variance predictor
    args.var_pred_n_bins = getattr(args, "var_pred_n_bins", 256)
    args.var_pred_hidden_dim = getattr(args, "var_pred_hidden_dim", 256)
    args.var_pred_kernel_size = getattr(args, "var_pred_kernel_size", 3)
    args.var_pred_dropout = getattr(args, "var_pred_dropout", 0.5)
    # postnet
    args.add_postnet = getattr(args, "add_postnet", False)
    args.postnet_dropout = getattr(args, "postnet_dropout", 0.5)
    args.postnet_layers = getattr(args, "postnet_layers", 5)
    args.postnet_conv_dim = getattr(args, "postnet_conv_dim", 512)
    args.postnet_conv_kernel_size = getattr(args, "postnet_conv_kernel_size", 5)