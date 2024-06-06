import copy
import torch
import torch.nn.functional as F

from dataclasses import dataclass, field

from fairseq import metrics, utils
from fairseq.data.data_utils import lengths_to_mask, lengths_to_padding_mask
from fairseq.criterions import register_criterion
from fairseq.criterions.label_smoothed_cross_entropy import (
    LabelSmoothedCrossEntropyCriterion,
    LabelSmoothedCrossEntropyCriterionConfig,
)


@dataclass
class ComSpeechCriterionConfig(LabelSmoothedCrossEntropyCriterionConfig):
    st_loss_weight: float = field(
        default=0.0,
        metadata={"help": "st loss weight"}
    )
    tts_loss_weight: float = field(
        default=0.0,
        metadata={"help": "tts loss weight"}
    )
    w2p_dp_loss_weight: float = field(
        default=0.0,
        metadata={"help": "word-to-phone dp loss weight"}
    )
    match_loss_weight: float = field(
        default=0.0,
        metadata={"help": "match loss weight"}
    )
    mse_loss_weight: float = field(
        default=1.0,
        metadata={"help": "mse loss weight"}
    )
    tctr_loss_weight: float = field(
        default=0.0,
        metadata={"help": "token-level contrastive loss weight"}
    )
    tctr_temp: float = field(
        default=1.0,
        metadata={"help": "temperature of token-level contrastive learning"}
    )
    metric_ctr: str = field(
        default="cosine",
        metadata={"help": "metric used in contrastive learning"}
    )


@register_criterion(
    "comspeech_criterion", dataclass=ComSpeechCriterionConfig
)
class ComSpeechCriterion(LabelSmoothedCrossEntropyCriterion):

    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        ignore_prefix_size=0,
        report_accuracy=False,
        st_loss_weight=0.0,
        tts_loss_weight=0.0,
        w2p_dp_loss_weight=0.0,
        match_loss_weight=0.0,
        mse_loss_weight=1.0,
        tctr_loss_weight=0.0,
        tctr_temp=1.0,
        metric_ctr="cosine",
    ):
        super().__init__(task, sentence_avg, label_smoothing, ignore_prefix_size, report_accuracy)
        self.st_loss_weight = st_loss_weight
        self.tts_loss_weight = tts_loss_weight
        self.w2p_dp_loss_weight = w2p_dp_loss_weight
        self.match_loss_weight = match_loss_weight
        self.tctr_loss_weight = tctr_loss_weight
        self.mse_loss_weight = mse_loss_weight
        self.tctr_temp = tctr_temp
        self.metric_ctr = metric_ctr

    def compute_st_loss(self, model, sample):
        x, st_decoder_padding_mask, st_decoder_out = model.forward_st(
            sample["net_input"]["src_tokens"],
            sample["net_input"]["src_lengths"],
            sample["net_input"]["prev_output_tokens"],
        )
        loss, _ = self.compute_loss(model, st_decoder_out, sample)
        loss /= sample["ntokens"]
        return loss, x, st_decoder_padding_mask, st_decoder_out

    def compute_w2p_loss(self, model, sample, x, st_decoder_padding_mask):
        x, st_decoder_padding_mask, adaptor_out = model.forward_w2p_adaptor(
            x, 
            st_decoder_padding_mask, 
            sample["multitask"]["target_phoneme"]["target"],
        )
        dp_loss = adaptor_out["dp_loss"] / sample["multitask"]["target_phoneme"]["ntokens"]
        return dp_loss, x, st_decoder_padding_mask, adaptor_out

    def get_logits(self, x1, x2):
        if self.metric_ctr == "cosine":
            logits = F.cosine_similarity(x1, x2, dim=-1)
        elif self.metric_ctr == "l1":
            logits = -F.l1_loss(x1, x2, reduction="none").sum(dim=-1)
        elif self.metric_ctr == "l2":
            logits = -F.mse_loss(x1, x2, reduction="none").sum(dim=-1)
        elif self.metric_ctr == "smoothl1":
            logits = -F.smooth_l1_loss(x1, x2, reduction="none").sum(dim=-1)
        elif self.metric_ctr == "dot":
            logits = (x1 * x2).sum(dim=-1)
        return logits

    def compute_mse_loss(self, x1, x2, st_decoder_padding_mask):
        mse_loss = F.mse_loss(x1, x2, reduction="none").sum(dim=-1)
        mse_loss = mse_loss[~st_decoder_padding_mask].mean()
        return mse_loss

    def compute_ctr_loss(self, x1, x2, st_decoder_padding_mask):
        _x1 = x1[~st_decoder_padding_mask]
        _x2 = x2[~st_decoder_padding_mask]
        bsz, dim = _x1.size()
        logits = self.get_logits(
            _x1.expand((bsz, bsz, dim)),
            _x2.expand((bsz, bsz, dim)).transpose(0, 1),
        )
        logits /= self.tctr_temp
        tctr_loss = -0.5 * (
            torch.nn.LogSoftmax(0)(logits) + torch.nn.LogSoftmax(1)(logits)
        ).diag().mean()
        return tctr_loss
    
    def compute_match_loss(self, model, sample, x, st_decoder_padding_mask):
        # st out
        x1, out1 = model.tts.forward_encoder(x, st_decoder_padding_mask, return_all_hiddens=True)
        # phoneme
        x2, phoneme_padding_mask = model.tts.forward_embedding(
            sample["multitask"]["target_phoneme"]["target"],
            sample["multitask"]["target_phoneme"]["target_lengths"]
        )
        x2, out2 = model.tts.forward_encoder(x2, phoneme_padding_mask, return_all_hiddens=True)
        assert torch.all(st_decoder_padding_mask == phoneme_padding_mask)
        # compute match loss
        mse_loss = self.compute_mse_loss(x1, x2, st_decoder_padding_mask)
        tctr_loss = self.compute_ctr_loss(x1, x2, st_decoder_padding_mask)
        match_loss = tctr_loss * self.tctr_loss_weight + mse_loss * self.mse_loss_weight
        return match_loss
    
    def compute_tts_loss(self, sample, tts_out):
        _feat_out, _feat_out_post = tts_out["x"], tts_out["x_post"]
        log_dur_out, pitch_out, energy_out = tts_out["log_dur_out"], tts_out["pitch_out"], tts_out["energy_out"]

        src_mask = lengths_to_mask(sample["net_input"]["src_lengths"])
        tgt_mask = lengths_to_mask(sample["target_lengths"])

        pitches, energies = sample["pitches"], sample["energies"]
        pitch_out, pitches = pitch_out[src_mask], pitches[src_mask]
        energy_out, energies = energy_out[src_mask], energies[src_mask]

        reduction="mean"
        feat_out, feat = _feat_out[tgt_mask], sample["target"][tgt_mask]
        l1_loss = F.l1_loss(feat_out, feat, reduction=reduction)
        if _feat_out_post is not None:
            l1_loss += F.l1_loss(_feat_out_post[tgt_mask], feat, reduction=reduction)

        pitch_loss = F.mse_loss(pitch_out, pitches, reduction=reduction)
        energy_loss = F.mse_loss(energy_out, energies, reduction=reduction)

        log_dur_out = log_dur_out[src_mask]
        dur = sample["durations"].float()
        dur = dur.half() if log_dur_out.type().endswith(".HalfTensor") else dur
        log_dur = torch.log(dur + 1)[src_mask]
        dur_loss = F.mse_loss(log_dur_out, log_dur, reduction=reduction)
        
        tts_loss = l1_loss + dur_loss + pitch_loss + energy_loss
        return tts_loss, l1_loss, dur_loss, pitch_loss, energy_loss

    def forward(self, model, sample, reduce=True):
        loss, st_loss, tts_loss, l1_loss, dur_loss, pitch_loss, energy_loss, w2p_dp_loss, match_loss = torch.Tensor([0]).cuda(), torch.Tensor([0]).cuda(), torch.Tensor([0]).cuda(), torch.Tensor([0]).cuda(), torch.Tensor([0]).cuda(), torch.Tensor([0]).cuda(), torch.Tensor([0]).cuda(), torch.Tensor([0]).cuda(), torch.Tensor([0]).cuda()

        mode = sample["net_input"]["mode"]
        if mode == "st":
            st_loss, x, st_decoder_padding_mask, st_decoder_out = self.compute_st_loss(model, sample)
            loss += st_loss * self.st_loss_weight
            if model.multiscale_modeling and self.w2p_dp_loss_weight > 0: 
                w2p_dp_loss, x, st_decoder_padding_mask, adaptor_out = self.compute_w2p_loss(model, sample, x, st_decoder_padding_mask)
                loss += w2p_dp_loss * self.w2p_dp_loss_weight
            if self.match_loss_weight > 0:
                match_loss = self.compute_match_loss(model, sample, x, st_decoder_padding_mask)
                loss += match_loss * self.match_loss_weight
        elif mode == "tts":
            tts_out = model.tts(
                sample["net_input"]["src_tokens"],
                sample["net_input"]["src_lengths"],
                durations=sample["durations"],
                pitches=sample["pitches"],
                energies=sample["energies"],
            )
            tts_loss, l1_loss, dur_loss, pitch_loss, energy_loss = self.compute_tts_loss(sample, tts_out)
            loss += tts_loss * self.tts_loss_weight
        elif mode == "s2st":
            # st loss
            _sample = copy.deepcopy(sample)
            key = "target_unigram" if model.multiscale_modeling else "target_phoneme"
            assert key in sample["multitask"]
            _sample["net_input"]["prev_output_tokens"] = sample["multitask"][key]["net_input"]["prev_output_tokens"]
            _sample["target"] = sample["multitask"][key]["target"]
            _sample["target_lengths"] = sample["multitask"][key]["target_lengths"]
            _sample["ntokens"] = sample["multitask"][key]["ntokens"]
            st_loss, x, st_decoder_padding_mask, st_decoder_out = self.compute_st_loss(model, _sample)
            loss += st_loss * self.st_loss_weight
            # w2p
            if model.multiscale_modeling:
                w2p_dp_loss, x, st_decoder_padding_mask, adaptor_out = self.compute_w2p_loss(model, _sample, x, st_decoder_padding_mask)
                loss += w2p_dp_loss * self.w2p_dp_loss_weight
            # tts
            x, _ = model.tts.forward_encoder(x, st_decoder_padding_mask)
            tts_out = model.tts.forward_variance_adaptor_and_decoder(
                x, 
                st_decoder_padding_mask,
                durations=sample["durations"],
                pitches=sample["pitches"],
                energies=sample["energies"],
            )
            _sample = copy.deepcopy(sample)
            _sample["net_input"]["src_tokens"] = sample["multitask"]["target_phoneme"]["target"]
            _sample["net_input"]["src_lengths"] = sample["multitask"]["target_phoneme"]["target_lengths"]
            tts_loss, l1_loss, dur_loss, pitch_loss, energy_loss = self.compute_tts_loss(_sample, tts_out)
            loss += tts_loss * self.tts_loss_weight

        sample_size = 1
        logging_output = {
            "loss": loss.data,
            "st_loss": st_loss.data,
            "tts_loss": tts_loss.data,
            "l1_loss": l1_loss.data,
            "dur_loss": dur_loss.data,
            "pitch_loss": pitch_loss.data,
            "energy_loss": energy_loss.data,
            "w2p_dp_loss": w2p_dp_loss.data,
            "match_loss": match_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }

        return loss, sample_size, logging_output
    
    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size, sample_size, round=3
        )
        for key in logging_outputs[0]:
            if key[-5:] == "_loss":
                val = utils.item(sum(log.get(key, 0) for log in logging_outputs))
                metrics.log_scalar(
                    key[:-5],
                    val / sample_size if sample_size > 0 else 0.0,
                    sample_size,
                    round=3,
                )