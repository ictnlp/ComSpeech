import torch
import logging
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from fairseq.data import data_utils as fairseq_data_utils
from fairseq.data import ConcatDataset, FairseqDataset, Dictionary
from fairseq.data.audio.speech_to_text_dataset import (
    _collate_frames,
    SpeechToTextDatasetCreator,
)
from fairseq.data.audio.audio_utils import get_features_or_waveform
from fairseq.data.audio.feature_transforms import CompositeAudioFeatureTransform
from fairseq.data.audio.waveform_transforms import CompositeAudioWaveformTransform
from fairseq.data.audio.speech_to_text_dataset import TextTargetMultitaskData
from fairseq.data.audio.speech_to_speech_dataset import (
    SpeechToSpeechDataset,
    SpeechToSpeechMultitaskDataset,
)
from ComSpeech.datasets.speech_to_speech_data_cfg_modified import S2SDataConfigModified

logger = logging.getLogger(__name__)


@dataclass
class SpeechToSpeechDatasetModifiedItem(object):
    index: int
    source: torch.Tensor
    target: Optional[torch.Tensor] = None
    target_speaker: Optional[torch.Tensor] = None
    tgt_lang_tag: Optional[int] = None
    duration: Optional[torch.Tensor] = None
    pitch: Optional[torch.Tensor] = None
    energy: Optional[torch.Tensor] = None


class SpeechToSpeechDatasetModified(FairseqDataset):
    def __init__(
        self,
        split: str,
        is_train_split: bool,
        cfg: S2SDataConfigModified,
        src_audio_paths: List[str],
        src_n_frames: List[int],
        tgt_audio_paths: Optional[List[str]] = None,
        tgt_n_frames: Optional[List[int]] = None,
        ids: Optional[List[str]] = None,
        n_frames_per_step: int = 1,
        durations: Optional[List[List[int]]] = None,
        pitches: Optional[List[str]] = None,
        energies: Optional[List[str]] = None,
    ):
        self.split, self.is_train_split = split, is_train_split
        self.cfg = cfg
        self.src_audio_paths, self.src_n_frames = src_audio_paths, src_n_frames
        self.tgt_audio_paths, self.tgt_n_frames = tgt_audio_paths, tgt_n_frames
        self.ids = ids
        
        self.n_samples = len(src_n_frames)
        self.shuffle = cfg.shuffle if is_train_split else False
        
        self.source_feature_transforms = CompositeAudioFeatureTransform.from_config_dict(
            self.cfg.get_source_feature_transforms(split, is_train_split)
        )
        self.source_waveform_transforms = CompositeAudioWaveformTransform.from_config_dict(
            self.cfg.get_source_waveform_transforms(split, is_train_split)
        )
        self.target_feature_transforms = CompositeAudioFeatureTransform.from_config_dict(
            self.cfg.get_target_feature_transforms(split, is_train_split)
        )
        self.target_waveform_transforms = CompositeAudioWaveformTransform.from_config_dict(
            self.cfg.get_target_waveform_transforms(split, is_train_split)
        )

        assert not self.cfg.use_audio_input

        # NOTE: n_frames_per_step is used for target audio rather than source audio
        self.n_frames_per_step = n_frames_per_step  

        self.durations = durations
        self.pitches = pitches
        self.energies = energies

        logger.info(self.__repr__())


    def __repr__(self):
        return (
            self.__class__.__name__
            + f'(split="{self.split}", n_samples={self.n_samples:_}, '
            f"prepend_tgt_lang_tag={self.cfg.prepend_tgt_lang_tag}, "
            f"n_frames_per_step={self.n_frames_per_step}, "
            f"shuffle={self.shuffle}, "
            f"source_feature_transforms={self.source_feature_transforms}, "
            f"source_waveform_transforms={self.source_waveform_transforms}, "
            f"target_feature_transforms={self.target_feature_transforms}, "
            f"target_waveform_transforms={self.target_waveform_transforms}, "
        )

    def pack_frames(self, feature: torch.Tensor):
        if self.n_frames_per_step == 1:
            return feature
        n_packed_frames = feature.shape[0] // self.n_frames_per_step
        feature = feature[: self.n_frames_per_step * n_packed_frames]
        return feature.reshape(n_packed_frames, -1)

    def _get_source_audio(self, index: int) -> torch.Tensor:
        """
        Gives source audio for given index with any relevant transforms applied.
        """
        source = get_features_or_waveform(
            self.src_audio_paths[index],
            waveform_transforms=self.source_waveform_transforms,
        )
        if self.source_feature_transforms is not None:
            source = self.source_feature_transforms(source)
        source = torch.from_numpy(source).float()
        return source

    def _get_target_audio(self, index: int) -> torch.Tensor:
        """
        Gives target audio for given index with any relevant transforms applied.
        """
        target = get_features_or_waveform(
            self.tgt_audio_paths[index],
            waveform_transforms=self.target_waveform_transforms,
        )
        if self.target_feature_transforms is not None:
            target = self.target_feature_transforms(target)
        target = torch.from_numpy(target).float()
        return target

    def __getitem__(self, index: int) -> SpeechToSpeechDatasetModifiedItem:
        # source audio
        source = self._get_source_audio(index)

        # target audio
        target = None
        if self.tgt_audio_paths is not None:
            target = self._get_target_audio(index)
            target = self.pack_frames(target)

        # variations
        duration, pitch, energy = None, None, None
        if self.durations is not None:
            duration = torch.tensor(
                self.durations[index] + [0], dtype=torch.long  # pad 0 for EOS
            )
        if self.pitches is not None:
            pitch = get_features_or_waveform(self.pitches[index])
            pitch = torch.from_numpy(
                np.concatenate((pitch, [0]))  # pad 0 for EOS
            ).float()
        if self.energies is not None:
            energy = get_features_or_waveform(self.energies[index])
            energy = torch.from_numpy(
                np.concatenate((energy, [0]))  # pad 0 for EOS
            ).float()

        return SpeechToSpeechDatasetModifiedItem(
            index=index, source=source, target=target,
            target_speaker=torch.FloatTensor([]), tgt_lang_tag=None,
            duration=duration, pitch=pitch, energy=energy,
        )

    def collater(
        self, samples: List[SpeechToSpeechDatasetModifiedItem], return_order: bool = False
) -> Dict:
        if len(samples) == 0:
            return {}
        indices = torch.tensor([x.index for x in samples], dtype=torch.long)
        # source audio
        sources = [x.source for x in samples]
        frames = _collate_frames(sources, self.cfg.use_audio_input)
        # sort samples by descending number of frames
        n_frames = torch.tensor([x.size(0) for x in sources], dtype=torch.long)
        n_frames, order = n_frames.sort(descending=True)
        indices = indices.index_select(0, order)
        frames = frames.index_select(0, order)

        # target audio
        target, target_lengths, prev_output_tokens, ntokens = None, None, None, None
        if self.tgt_audio_paths is not None:
            target = _collate_frames(
                [x.target for x in samples], is_audio_input=False
            )
            bsz, _, d = target.size()
            prev_output_tokens = torch.cat(
                (target.new_full((bsz, 1, d), 0.0), target[:, :-1, :]), dim=1
            )
            target_lengths = torch.tensor(
                [x.target.size(0) for x in samples], dtype=torch.long
            )
            target = target.index_select(0, order)
            target_lengths = target_lengths.index_select(0, order)
            prev_output_tokens = prev_output_tokens.index_select(0, order)
            ntokens = sum(x.target.size(0) for x in samples)
        
        # variations
        durations, pitches, energies = None, None, None
        if self.durations is not None:
            durations = fairseq_data_utils.collate_tokens(
                [x.duration for x in samples], 0
            ).index_select(0, order)
        if self.pitches is not None:
            pitches = _collate_frames([x.pitch for x in samples], True)
            pitches = pitches.index_select(0, order)
        if self.energies is not None:
            energies = _collate_frames([x.energy for x in samples], True)
            energies = energies.index_select(0, order)

        net_input = {
            "src_tokens": frames,
            "src_lengths": n_frames,
            "prev_output_tokens": prev_output_tokens,
            "tgt_speaker": None,
        }
        out = {
            "id": indices,
            "net_input": net_input,
            "speaker": None,
            "target": target,
            "target_lengths": target_lengths,
            "durations": durations,
            "pitches": pitches,
            "energies": energies,
            "ntokens": ntokens,
            "nsentences": len(samples),
        }
        if return_order:
            out["order"] = order
        return out

    def __len__(self):
        return self.n_samples

    def num_tokens(self, index):
        return self.src_n_frames[index]
    
    def size(self, index):
        return self.src_n_frames[index], self.tgt_n_frames[index]
    
    @property
    def sizes(self):
        return np.array(self.src_n_frames)
    
    @property
    def can_reuse_epoch_itr_across_epochs(self):
        return True

    def ordered_indices(self):
        if self.shuffle:
            order = [np.random.permutation(len(self))]
        else:
            order = [np.arange(len(self))]
        # first by descending order of # of frames then by original/random order
        order.append([-n for n in self.src_n_frames])
        return np.lexsort(order)

    def prefetch(self, indices):
        raise False


class SpeechToSpeechMultitaskDatasetModified(SpeechToSpeechDatasetModified):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.multitask_data = {}

    def add_multitask_dataset(self, task_name, task_data):
        self.multitask_data[task_name] = task_data

    def __getitem__(
        self, index: int
    ) -> Tuple[SpeechToSpeechDatasetModifiedItem, Dict[str, torch.Tensor]]:
        s2s_data = super().__getitem__(index)

        multitask_target = {}
        sample_id = self.ids[index]
        for task_name, task_dataset in self.multitask_data.items():
            multitask_target[task_name] = task_dataset.get(sample_id, None)

        return s2s_data, multitask_target

    def collater(
        self, samples: List[Tuple[SpeechToSpeechDatasetModifiedItem, Dict[str, torch.Tensor]]]
    ) -> Dict:
        if len(samples) == 0:
            return {}

        out = super().collater([s for s, _ in samples], return_order=True)
        order = out["order"]
        del out["order"]

        for task_name, task_dataset in self.multitask_data.items():
            if "multitask" not in out:
                out["multitask"] = {}
            d = [s[task_name] for _, s in samples]
            task_target = task_dataset.collater(d)
            out["multitask"][task_name] = {
                "target": task_target["target"].index_select(0, order),
                "target_lengths": task_target["target_lengths"].index_select(0, order),
                "ntokens": task_target["ntokens"],
            }
            out["multitask"][task_name]["net_input"] = {
                "prev_output_tokens": task_target["prev_output_tokens"].index_select(
                    0, order
                ),
            }

        return out


class SpeechToSpeechDatasetModifiedCreator(object):
    # mandatory columns
    KEY_ID, KEY_SRC_AUDIO, KEY_SRC_N_FRAMES = "id", "src_audio", "src_n_frames"
    # optional columns
    KEY_TGT_AUDIO, KEY_TGT_N_FRAMES = "tgt_audio", "tgt_n_frames"
    KEY_DURATION, KEY_PITCH, KEY_ENERGY = "duration", "pitch", "energy"

    @classmethod
    def _from_list(
        cls,
        split_name: str,
        is_train_split,
        samples: List[Dict],
        cfg: S2SDataConfigModified,
        target_is_code: bool,
        tgt_dict: Dictionary = None,
        n_frames_per_step: int = 1,
        multitask: Optional[Dict] = None,
    ) -> SpeechToSpeechDatasetModified:
        ids = [s[cls.KEY_ID] for s in samples]
        src_audio_paths = [s[cls.KEY_SRC_AUDIO] for s in samples]
        src_n_frames = [int(s[cls.KEY_SRC_N_FRAMES]) for s in samples]
        tgt_audio_paths = [s.get(cls.KEY_TGT_AUDIO, None) for s in samples]
        tgt_n_frames = [int(s.get(cls.KEY_TGT_N_FRAMES, 0)) for s in samples]

        tgt_audio_paths = None if any(tgt is None for tgt in tgt_audio_paths) else tgt_audio_paths
        durations = [s.get(cls.KEY_DURATION, None) for s in samples]
        durations = [
            None if dd is None else [int(d) for d in dd.split(" ")] for dd in durations
        ]
        durations = None if any(dd is None for dd in durations) else durations

        pitches = [s.get(cls.KEY_PITCH, None) for s in samples]
        pitches = None if any(pp is None for pp in pitches) else pitches

        energies = [s.get(cls.KEY_ENERGY, None) for s in samples]
        energies = None if any(ee is None for ee in energies) else energies

        if target_is_code:
            has_multitask = multitask is not None and len(multitask.keys()) > 0
            dataset_cls = (
                SpeechToSpeechMultitaskDataset if has_multitask else SpeechToSpeechDataset
            )
            tgt_audio_paths = ["" for s in samples]
            ds = dataset_cls(
                split=split_name,
                is_train_split=is_train_split,
                data_cfg=cfg,
                src_audio_paths=src_audio_paths,
                src_n_frames=src_n_frames,
                tgt_audio_paths=tgt_audio_paths,
                tgt_n_frames=tgt_n_frames,
                src_langs=None,
                tgt_langs=None,
                ids=ids,
                target_is_code=target_is_code,
                tgt_dict=tgt_dict,
                n_frames_per_step=n_frames_per_step,
                durations=durations,
                pitches=pitches,
                energies=energies,
            )
        else:
            has_multitask = multitask is not None and len(multitask.keys()) > 0
            dataset_cls = (
                SpeechToSpeechMultitaskDatasetModified if has_multitask else SpeechToSpeechDatasetModified
            )
            ds = dataset_cls(
                split=split_name,
                is_train_split=is_train_split,
                cfg=cfg,
                src_audio_paths=src_audio_paths,
                src_n_frames=src_n_frames,
                tgt_audio_paths=tgt_audio_paths,
                tgt_n_frames=tgt_n_frames,
                ids=ids,
                n_frames_per_step=n_frames_per_step,
                durations=durations,
                pitches=pitches,
                energies=energies,
            )

        if has_multitask:
            for task_name, task_obj in multitask.items():
                task_data = TextTargetMultitaskData(
                    task_obj.args, split_name, task_obj.target_dictionary
                )
                ds.add_multitask_dataset(task_name, task_data)
        return ds

    @classmethod
    def from_tsv(
        cls,
        root: str,
        cfg: S2SDataConfigModified,
        splits: str,
        target_is_code: bool,
        is_train_split: bool,
        tgt_dict: Dictionary = None,
        n_frames_per_step: int = 1,
        multitask: Optional[Dict] = None,
    ) -> SpeechToSpeechDatasetModified:
        datasets = []
        for split in splits.split(","):
            samples = SpeechToTextDatasetCreator._load_samples_from_tsv(root, split)
            ds = cls._from_list(
                split_name=split,
                is_train_split=is_train_split,
                samples=samples,
                cfg=cfg,
                target_is_code=target_is_code,
                tgt_dict=tgt_dict,
                n_frames_per_step=n_frames_per_step,
                multitask=multitask,
            )
            datasets.append(ds)
        return ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]
