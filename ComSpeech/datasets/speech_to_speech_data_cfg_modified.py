import logging
from copy import deepcopy
from typing import Dict
from fairseq.data.audio.data_cfg import S2SDataConfig

logger = logging.getLogger(__name__)


class S2SDataConfigModified(S2SDataConfig):

    def get_source_feature_transforms(self, split, is_train):
        cfg = deepcopy(self.config)
        # TODO: deprecate transforms
        cur = self.get_transforms("source_", split, is_train)
        if cur is not None:
            logger.warning(
                "Auto converting source_transforms into source_feature_transforms, "
                "but transforms will be deprecated in the future. Please "
                "update this in the config."
            )
            ft_transforms = self.get_transforms("source_feature_", split, is_train)
            if ft_transforms:
                cur.extend(ft_transforms)
        else:
            cur = self.get_transforms("source_feature_", split, is_train)
        cfg["feature_transforms"] = cur
        return cfg

    def get_source_waveform_transforms(self, split, is_train):
        cfg = deepcopy(self.config)
        cfg["waveform_transforms"] = self.get_transforms("source_waveform_", split, is_train)
        return cfg

    def get_target_feature_transforms(self, split, is_train):
        cfg = deepcopy(self.config)
        # TODO: deprecate transforms
        cur = self.get_transforms("target_", split, is_train)
        if cur is not None:
            logger.warning(
                "Auto converting target_transforms into target_feature_transforms, "
                "but transforms will be deprecated in the future. Please "
                "update this in the config."
            )
            ft_transforms = self.get_transforms("target_feature_", split, is_train)
            if ft_transforms:
                cur.extend(ft_transforms)
        else:
            cur = self.get_transforms("target_feature_", split, is_train)
        cfg["feature_transforms"] = cur
        return cfg

    def get_target_waveform_transforms(self, split, is_train):
        cfg = deepcopy(self.config)
        cfg["waveform_transforms"] = self.get_transforms("target_waveform_", split, is_train)
        return cfg

    @property
    def vocab_filename_st(self):
        """fairseq vocabulary file under data root"""
        return self.config.get("vocab_filename_st", "dict.txt")
    
    @property
    def vocab_filename_tts(self):
        """fairseq vocabulary file under data root"""
        return self.config.get("vocab_filename_tts", "dict.txt")
    
    @property
    def pre_tokenizer(self) -> Dict:
        """Pre-tokenizer to apply before subword tokenization. Returning
        a dictionary with `tokenizer` providing the tokenizer name and
        the other items providing the tokenizer-specific arguments.
        Tokenizers are defined in `fairseq.data.encoders.*`"""
        tokenizer = self.config.get("pre_tokenizer", {"tokenizer": None})
        return self._auto_convert_to_abs_path(tokenizer)

    @property
    def bpe_tokenizer(self) -> Dict:
        """Subword tokenizer to apply after pre-tokenization. Returning
        a dictionary with `bpe` providing the tokenizer name and
        the other items providing the tokenizer-specific arguments.
        Tokenizers are defined in `fairseq.data.encoders.*`"""
        tokenizer = self.config.get("bpe_tokenizer", {"bpe": None})
        return self._auto_convert_to_abs_path(tokenizer)