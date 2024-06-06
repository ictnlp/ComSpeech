import numpy as np
import torch

from fairseq.data.data_utils import lengths_to_padding_mask


class ComSpeechGenerator(object):
    def __init__(
        self,
        models,
        args,
        data_cfg,
        tgt_dict_st,
        tgt_dict_tts=None,
        eos=None,
        symbols_to_strip_from_output=None,
    ):
        self.model = models[0]
        self.model.eval()
        stats_npz_path = data_cfg.global_cmvn_stats_npz
        self.gcmvn_stats = None
        if stats_npz_path is not None:
            self.gcmvn_stats = np.load(stats_npz_path)

        self.tgt_dict_st = tgt_dict_st
        self.tgt_dict_tts = tgt_dict_tts

        from ComSpeech.generator.sequence_generator import SequenceGenerator
        from fairseq import search

        self.text_generator = SequenceGenerator(
            models,
            tgt_dict_st,
            beam_size=max(1, getattr(args, "beam", 5)),
            max_len_a=getattr(args, "max_len_a", 0),
            max_len_b=getattr(args, "max_len_b", 200),
            min_len=getattr(args, "min_len", 1),
            normalize_scores=(not getattr(args, "unnormalized", False)),
            len_penalty=getattr(args, "lenpen", 1),
            unk_penalty=getattr(args, "unkpen", 0),
            temperature=getattr(args, "temperature", 1.0),
            match_source_len=getattr(args, "match_source_len", False),
            no_repeat_ngram_size=getattr(args, "no_repeat_ngram_size", 0),
            search_strategy=search.BeamSearch(tgt_dict_st),
            eos=eos,
            symbols_to_strip_from_output=symbols_to_strip_from_output,
        )

        self.eos = self.text_generator.eos

    @torch.no_grad()
    def generate(self, model, sample, has_targ=False, **kwargs):
        src_tokens = sample["net_input"]["src_tokens"]
        src_lengths = sample["net_input"]["src_lengths"]
        bsz, src_len = src_tokens.size()[:2]
        n_frames_per_step = self.model.tts.n_frames_per_step
        out_dim = self.model.tts.out_dim
        raw_dim = out_dim // n_frames_per_step

        mode = sample["net_input"]["mode"]
        if mode in ("st", "s2st"):
            # initialize
            encoder_out = self.model.encoder(src_tokens, src_lengths)

            prefix_tokens = None
            constraints = None
            bos_token = None

            st_decoder = self.model.decoder

            # 1. ST decoder
            finalized_st = self.text_generator.generate_decoder(
                [encoder_out],
                src_tokens,
                src_lengths,
                sample,
                prefix_tokens,
                constraints,
                bos_token,
                decoder=st_decoder,
            )

            # extract decoder output corresponding to the best hypothesis
            max_tgt_len = max([len(hypo[0]["tokens"]) for hypo in finalized_st])
            prev_output_tokens_st = (
                src_tokens.new_zeros(src_tokens.shape[0], max_tgt_len)
                .fill_(st_decoder.padding_idx)
                .int()
            )  # B x T
            for i, hypo in enumerate(finalized_st):
                i_beam = 0
                tmp = hypo[i_beam]["tokens"].int()  # hyp + eos
                prev_output_tokens_st[i, 0] = self.text_generator.eos
                if tmp[-1] == self.text_generator.eos:
                    tmp = tmp[:-1]
                prev_output_tokens_st[i, 1 : len(tmp) + 1] = tmp

                text = "".join([self.tgt_dict_st[c] for c in tmp])
                text = text.replace("_", " ")
                text = text.replace("▁", " ")
                text = text.replace("<unk>", " ")
                text = text.replace("<s>", "")
                text = text.replace("</s>", "")
                if len(text) > 0 and text[0] == " ":
                    text = text[1:]
                sample_id = sample["id"].tolist()[i]
                print("{} (None-{})".format(text, sample_id))

            st_decoder_out = st_decoder(
                prev_output_tokens_st,
                encoder_out=encoder_out,
                features_only=True,
            )
            x = st_decoder_out[0].transpose(0, 1)

            st_decoder_padding_mask = prev_output_tokens_st.eq(st_decoder.padding_idx)

            # 2. Word-to-phone adaptor
            if self.model.multiscale_modeling:
                x = x.transpose(0, 1)
                x, st_decoder_padding_mask, adaptor_out = self.model.forward_w2p_adaptor(x, st_decoder_padding_mask)
                ctc_pred = adaptor_out["ctc_pred"]
                for i in range(len(ctc_pred)):
                    pred = ctc_pred[i]
                    pred = [c for c in pred if self.tgt_dict_tts[c] != "</s>"]
                    text = " ".join([self.tgt_dict_tts[c] for c in pred])
                    sample_id = sample["id"].tolist()[i]
                    print("Phoneme-{}\t{}".format(sample_id, text))

            if mode == "st":
                return finalized_st

            # 3. TTS encoder and decoder
            x, _ = self.model.tts.forward_encoder(x, st_decoder_padding_mask)
            tts_out = self.model.tts.forward_variance_adaptor_and_decoder(
                x,
                st_decoder_padding_mask,
            )

        if mode == "tts":
            tts_out = self.model.tts(
                sample["net_input"]["src_tokens"],
                sample["net_input"]["src_lengths"],
            )    

        feat, feat_post, out_lens = tts_out["x"], tts_out["x_post"], tts_out["out_lens"]
        if feat_post is not None:
            feat = feat_post

        feat = feat.view(bsz, -1, raw_dim)
        feat = self.gcmvn_denormalize(feat)

        out_lens = out_lens * n_frames_per_step
        tts_finalized = [
            {
                "feature": feat[b, :l] if l > 0 else feat.new_zeros([1, raw_dim]),
                "waveform": None,
            }
            for b, l in zip(range(bsz), out_lens)
        ]

        return tts_finalized
    
    def gcmvn_denormalize(self, x):
        # x: B x T x C
        if self.gcmvn_stats is None:
            return x
        mean = torch.from_numpy(self.gcmvn_stats["mean"]).to(x)
        std = torch.from_numpy(self.gcmvn_stats["std"]).to(x)
        assert len(x.shape) == 3 and mean.shape[0] == std.shape[0] == x.shape[2]
        x = x * std.view(1, 1, -1).expand_as(x)
        return x + mean.view(1, 1, -1).expand_as(x)