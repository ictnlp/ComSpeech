import logging
import torch
import numpy as np

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from pathlib import Path

from fairseq.data import ConcatDataset, Dictionary, ResamplingDataset
from fairseq.data import data_utils as fairseq_data_utils
from fairseq.data.audio.speech_to_text_dataset import S2TDataConfig
from fairseq.data.audio.feature_transforms import CompositeAudioFeatureTransform
from fairseq.data.audio.waveform_transforms import CompositeAudioWaveformTransform
from fairseq.data.audio.dataset_transforms import CompositeAudioDatasetTransform
from fairseq.data.audio.speech_to_text_dataset import _collate_frames
from fairseq.data.audio.text_to_speech_dataset import (
    TextToSpeechDataset,
    TextToSpeechDatasetCreator,
)

logger = logging.getLogger(__name__)


@dataclass
class TextToSpeechDatasetModifiedItem(object):
    index: int
    source: torch.Tensor
    target: Optional[torch.Tensor] = None
    speaker_id: Optional[int] = None
    duration: Optional[torch.Tensor] = None
    pitch: Optional[torch.Tensor] = None
    energy: Optional[torch.Tensor] = None


class TextToSpeechDatasetModified(TextToSpeechDataset):

    def __init__(
        self,
        split: str,
        is_train_split: bool,
        cfg: S2TDataConfig,
        audio_paths: List[str],
        n_frames: List[int],
        src_texts: Optional[List[str]] = None,
        tgt_texts: Optional[List[str]] = None,
        speakers: Optional[List[str]] = None,
        src_langs: Optional[List[str]] = None,
        tgt_langs: Optional[List[str]] = None,
        ids: Optional[List[str]] = None,
        tgt_dict: Optional[Dictionary] = None,
        pre_tokenizer=None,
        bpe_tokenizer=None,
        n_frames_per_step=1,
        speaker_to_id=None,
        durations: Optional[List[List[int]]] = None,
        pitches: Optional[List[str]] = None,
        energies: Optional[List[str]] = None,
    ):
        self.split, self.is_train_split = split, is_train_split
        self.cfg = cfg
        self.audio_paths, self.n_frames = audio_paths, n_frames
        self.n_samples = len(audio_paths)
        assert len(n_frames) == self.n_samples > 0
        assert src_texts is None or len(src_texts) == self.n_samples
        assert tgt_texts is None or len(tgt_texts) == self.n_samples
        assert speakers is None or len(speakers) == self.n_samples
        assert src_langs is None or len(src_langs) == self.n_samples
        assert tgt_langs is None or len(tgt_langs) == self.n_samples
        assert ids is None or len(ids) == self.n_samples
        assert (tgt_dict is None and tgt_texts is None) or (
            tgt_dict is not None and tgt_texts is not None
        )
        self.src_texts, self.tgt_texts = src_texts, tgt_texts
        self.src_langs, self.tgt_langs = src_langs, tgt_langs
        self.speakers = speakers
        self.tgt_dict = tgt_dict
        self.check_tgt_lang_tag()
        self.ids = ids
        self.shuffle = cfg.shuffle if is_train_split else False

        self.feature_transforms = CompositeAudioFeatureTransform.from_config_dict(
            self.cfg.get_target_feature_transforms(split, is_train_split)
        )
        self.waveform_transforms = CompositeAudioWaveformTransform.from_config_dict(
            self.cfg.get_target_waveform_transforms(split, is_train_split)
        )
        # TODO: add these to data_cfg.py
        self.dataset_transforms = CompositeAudioDatasetTransform.from_config_dict(
            self.cfg.get_dataset_transforms(split, is_train_split)
        )

        # check proper usage of transforms
        if self.feature_transforms and self.cfg.use_audio_input:
            logger.warning(
                "Feature transforms will not be applied. To use feature transforms, "
                "set use_audio_input as False in config."
            )

        self.pre_tokenizer = pre_tokenizer
        self.bpe_tokenizer = bpe_tokenizer
        self.n_frames_per_step = n_frames_per_step
        self.speaker_to_id = speaker_to_id

        self.tgt_lens = self.get_tgt_lens_and_check_oov()
        self.append_eos = True

        self.durations = durations
        self.pitches = pitches
        self.energies = energies

        logger.info(self.__repr__())

    def __getitem__(self, index: int) -> TextToSpeechDatasetModifiedItem:
        tts_item = super().__getitem__(index)

        return TextToSpeechDatasetModifiedItem(
            index=index,
            source=tts_item.source,
            target=tts_item.target,
            speaker_id=tts_item.speaker_id,
            duration=tts_item.duration,
            pitch=tts_item.pitch,
            energy=tts_item.energy,
        )

    def collater(self, samples: List[TextToSpeechDatasetModifiedItem]) -> Dict[str, Any]:
        if len(samples) == 0:
            return {}

        src_lengths, order = torch.tensor(
            [s.target.shape[0] for s in samples], dtype=torch.long
        ).sort(descending=True)
        id_ = torch.tensor([s.index for s in samples], dtype=torch.long).index_select(
            0, order
        )
        feat = _collate_frames(
            [s.source for s in samples], self.cfg.use_audio_input
        ).index_select(0, order)
        target_lengths = torch.tensor(
            [s.source.shape[0] for s in samples], dtype=torch.long
        ).index_select(0, order)

        src_tokens = fairseq_data_utils.collate_tokens(
            [s.target for s in samples],
            self.tgt_dict.pad(),
            self.tgt_dict.eos(),
            left_pad=False,
            move_eos_to_beginning=False,
        ).index_select(0, order)

        speaker = None
        if self.speaker_to_id is not None:
            speaker = (
                torch.tensor([s.speaker_id for s in samples], dtype=torch.long)
                .index_select(0, order)
                .view(-1, 1)
            )

        bsz, _, d = feat.size()
        prev_output_tokens = torch.cat(
            (feat.new_zeros((bsz, 1, d)), feat[:, :-1, :]), dim=1
        )

        durations, pitches, energies = None, None, None
        if self.durations is not None:
            durations = fairseq_data_utils.collate_tokens(
                [s.duration for s in samples], 0
            ).index_select(0, order)
            assert src_tokens.shape[1] == durations.shape[1]
        if self.pitches is not None:
            pitches = _collate_frames([s.pitch for s in samples], True)
            pitches = pitches.index_select(0, order)
            assert src_tokens.shape[1] == pitches.shape[1]
        if self.energies is not None:
            energies = _collate_frames([s.energy for s in samples], True)
            energies = energies.index_select(0, order)
            assert src_tokens.shape[1] == energies.shape[1]
        src_texts = [self.tgt_dict.string(samples[i].target) for i in order]

        return {
            "id": id_,
            "net_input": {
                "src_tokens": src_tokens,
                "src_lengths": src_lengths,
                "prev_output_tokens": prev_output_tokens,
            },
            "speaker": speaker,
            "target": feat,
            "durations": durations,
            "pitches": pitches,
            "energies": energies,
            "target_lengths": target_lengths,
            "ntokens": sum(target_lengths).item(),
            "nsentences": len(samples),
            "src_texts": src_texts,
        }


class TextToSpeechDatasetModifiedCreator(TextToSpeechDatasetCreator):

    @classmethod
    def _from_list(
        cls,
        split_name: str,
        is_train_split,
        samples: List[Dict],
        cfg: S2TDataConfig,
        tgt_dict,
        pre_tokenizer,
        bpe_tokenizer,
        n_frames_per_step,
        speaker_to_id,
        multitask=None,
    ) -> TextToSpeechDatasetModified:
        audio_root = Path(cfg.audio_root)
        ids = [s[cls.KEY_ID] for s in samples]
        audio_paths = [(audio_root / s[cls.KEY_AUDIO]).as_posix() for s in samples]
        n_frames = [int(s[cls.KEY_N_FRAMES]) for s in samples]
        tgt_texts = [s[cls.KEY_TGT_TEXT] for s in samples]
        src_texts = [s.get(cls.KEY_SRC_TEXT, cls.DEFAULT_SRC_TEXT) for s in samples]
        speakers = [s.get(cls.KEY_SPEAKER, cls.DEFAULT_SPEAKER) for s in samples]
        src_langs = [s.get(cls.KEY_SRC_LANG, cls.DEFAULT_LANG) for s in samples]
        tgt_langs = [s.get(cls.KEY_TGT_LANG, cls.DEFAULT_LANG) for s in samples]

        durations = [s.get(cls.KEY_DURATION, None) for s in samples]
        durations = [
            None if dd is None else [int(d) for d in dd.split(" ")] for dd in durations
        ]
        durations = None if any(dd is None for dd in durations) else durations

        pitches = [s.get(cls.KEY_PITCH, None) for s in samples]
        pitches = [
            None if pp is None else (audio_root / pp).as_posix() for pp in pitches
        ]
        pitches = None if any(pp is None for pp in pitches) else pitches

        energies = [s.get(cls.KEY_ENERGY, None) for s in samples]
        energies = [
            None if ee is None else (audio_root / ee).as_posix() for ee in energies
        ]
        energies = None if any(ee is None for ee in energies) else energies

        return TextToSpeechDatasetModified(
            split_name,
            is_train_split,
            cfg,
            audio_paths,
            n_frames,
            src_texts,
            tgt_texts,
            speakers,
            src_langs,
            tgt_langs,
            ids,
            tgt_dict,
            pre_tokenizer,
            bpe_tokenizer,
            n_frames_per_step,
            speaker_to_id,
            durations,
            pitches,
            energies,
        )
    
    @classmethod
    def _from_tsv(
        cls,
        root: str,
        cfg: S2TDataConfig,
        split: str,
        tgt_dict,
        is_train_split: bool,
        pre_tokenizer,
        bpe_tokenizer,
        n_frames_per_step,
        speaker_to_id,
        multitask: Optional[Dict] = None,
    ) -> TextToSpeechDatasetModified:
        samples = cls._load_samples_from_tsv(root, split)
        return cls._from_list(
            split,
            is_train_split,
            samples,
            cfg,
            tgt_dict,
            pre_tokenizer,
            bpe_tokenizer,
            n_frames_per_step,
            speaker_to_id,
            multitask,
        )

    @classmethod
    def from_tsv(
        cls,
        root: str,
        cfg: S2TDataConfig,
        splits: str,
        tgt_dict,
        pre_tokenizer,
        bpe_tokenizer,
        is_train_split: bool,
        epoch: int,
        seed: int,
        n_frames_per_step: int = 1,
        speaker_to_id=None,
        multitask: Optional[Dict] = None,
    ) -> TextToSpeechDatasetModified:
        datasets = [
            cls._from_tsv(
                root=root,
                cfg=cfg,
                split=split,
                tgt_dict=tgt_dict,
                is_train_split=is_train_split,
                pre_tokenizer=pre_tokenizer,
                bpe_tokenizer=bpe_tokenizer,
                n_frames_per_step=n_frames_per_step,
                speaker_to_id=speaker_to_id,
                multitask=multitask,
            )
            for split in splits.split(",")
        ]

        if is_train_split and len(datasets) > 1 and cfg.sampling_alpha != 1.0:
            # temperature-based sampling
            size_ratios = cls.get_size_ratios(datasets, alpha=cfg.sampling_alpha)
            datasets = [
                ResamplingDataset(
                    d, size_ratio=r, seed=seed, epoch=epoch, replace=(r >= 1.0)
                )
                for r, d in zip(size_ratios, datasets)
            ]

        return ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]
