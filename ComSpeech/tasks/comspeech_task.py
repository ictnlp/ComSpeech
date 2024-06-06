import torch
import logging

from pathlib import Path
from argparse import Namespace

from fairseq.data import Dictionary
from fairseq.data.iterators import GroupedEpochBatchIterator
from fairseq.data import encoders
from fairseq.data.audio.multi_modality_dataset import (
    MultiModalityDataset,
    ModalityDatasetItem,
)
from fairseq.data.audio.data_cfg import MultitaskConfig
from fairseq.tasks import LegacyFairseqTask, register_task
from fairseq.tasks.speech_to_text import DummyMultiTask

from ComSpeech.datasets.speech_to_speech_data_cfg_modified import S2SDataConfigModified
from ComSpeech.datasets.speech_to_text_dataset_modified import SpeechToTextDatasetModifiedCreator, SpeechToTextDatasetModified
from ComSpeech.datasets.text_to_speech_dataset_modified import TextToSpeechDatasetModifiedCreator
from ComSpeech.datasets.speech_to_speech_dataset_modified import SpeechToSpeechDatasetModifiedCreator

logger = logging.getLogger(__name__)


@register_task("comspeech_task")
class ComSpeechTask(LegacyFairseqTask):

    @classmethod
    def add_args(cls, parser):
        parser.add_argument("data", help="manifest root path")
        parser.add_argument(
            "--st-data",
            type=str,
            help="st data path",
        )
        parser.add_argument(
            "--tts-data",
            type=str,
            help="tts data path",
        )
        parser.add_argument(
            "--config-yaml",
            type=str,
            default="config.yaml",
            help="Configuration YAML filename (under manifest root)",
        )
        parser.add_argument(
            "--multitask-config-yaml",
            type=str,
            default=None,
            help="Configuration YAML filename for the multitasks (under manifest root)",
        )
        parser.add_argument(
            "--max-source-positions",
            default=6000,
            type=int,
            metavar="N",
            help="max number of tokens in the source sequence",
        )
        parser.add_argument(
            "--max-target-positions",
            default=1024,
            type=int,
            metavar="N",
            help="max number of tokens in the target sequence",
        )
        parser.add_argument(
            "--max-target-audio-positions",
            default=1200,
            type=int,
            metavar="N",
            help="max number of frames in the target audio",
        )
        parser.add_argument(
            "--n-frames-per-step",
            type=int,
            default=1,
            help="# stacked frames, use 0 for reduced discrete unit sequence",
        )
        parser.add_argument(
            "--max-tokens-st",
            type=int,
            metavar="N",
            help="maximum tokens for st batch",
        )
        parser.add_argument(
            "--max-tokens-tts",
            type=int,
            metavar="N",
            help="maximum tokens for tts batch",
        )
        parser.add_argument(
            "--batch-size-st",
            type=int,
            metavar="N",
            help="batch size for st dataset",
        )
        parser.add_argument(
            "--batch-size-tts",
            type=int,
            metavar="N",
            help="batch size for tts dataset",
        )
        parser.add_argument(
            "--st-sample-ratio",
            type=float,
            default=1.0,
            help="sample ratio of st dataset",
        )
        parser.add_argument(
            "--s2st-sample-ratio",
            type=float,
            default=1.0,
            help="sample ratio of s2st dataset",
        )
        parser.add_argument(
            "--tts-sample-ratio",
            type=float,
            default=1.0,
            help="sample ratio of tts dataset",
        )
        parser.add_argument(
            "--validate-task",
            type=str,
            default="s2st",
            choices=["st", "tts", "s2st"],
            help="validation task",
        )
        parser.add_argument(
            "--multiscale-modeling",
            action="store_true",
            default=False,
            help="use subword vocab for st instead of phoneme",
        )
        parser.add_argument(
            "--supervised-s2st-training",
            action="store_true",
            default=False,
            help="supervised training with s2st data",
        )
    
    def __init__(self, args, tgt_dict_st, tgt_dict_tts):
        super().__init__(args)
        self.tgt_dict_st = tgt_dict_st
        self.tgt_dict_tts = tgt_dict_tts
        self.data_cfg = S2SDataConfigModified(Path(args.data) / args.config_yaml)
        self.speaker_to_id = self._get_speaker_to_id()

        if args.multiscale_modeling:
            self.pre_tokenizer = self.build_tokenizer(self.args)
            self.bpe_tokenizer = self.build_bpe(self.args)
        else:
            self.pre_tokenizer = None
            self.bpe_tokenizer = None
        
        self.multitask_tasks = {}
        if getattr(args, "multitask_config_yaml", None) is not None:
            multitask_cfg = MultitaskConfig(
                Path(args.data) / args.multitask_config_yaml
            )
            for i, (task_name, task_config) in enumerate(
                multitask_cfg.get_all_tasks().items()
            ):
                task_obj = DummyMultiTask(
                    task_config,
                    task_config.tgt_dict,
                    first_pass=False,
                )
                self.multitask_tasks[task_name] = task_obj            

    def _get_speaker_to_id(self):
        speaker_to_id = None
        speaker_set_filename = self.data_cfg.config.get("speaker_set_filename")
        if speaker_set_filename is not None:
            speaker_set_path = Path(self.args.data) / speaker_set_filename
            with open(speaker_set_path) as f:
                speaker_to_id = {r.strip(): i for i, r in enumerate(f)}
        return speaker_to_id

    @classmethod
    def setup_task(cls, args, **kwargs):
        data_cfg = S2SDataConfigModified(Path(args.data) / args.config_yaml)
        st_dict_path = Path(args.data) / data_cfg.vocab_filename_st
        if not st_dict_path.is_file():
            raise FileNotFoundError(f"Dict not found: {st_dict_path.as_posix()}")
        tgt_dict_st = Dictionary.load(st_dict_path.as_posix())
        logger.info(
            f"ST dictionary size ({data_cfg.vocab_filename_st}): " f"{len(tgt_dict_st):,}"
        )
        tts_dict_path = Path(args.data) / data_cfg.vocab_filename_tts
        if not tts_dict_path.is_file():
            raise FileNotFoundError(f"Dict not found: {tts_dict_path.as_posix()}")
        tgt_dict_tts = Dictionary.load(tts_dict_path.as_posix())
        logger.info(
            f"TTS dictionary size ({data_cfg.vocab_filename_tts}): " f"{len(tgt_dict_tts):,}"
        )

        if getattr(args, "train_subset", None) is not None:
            if not all(s.startswith("train") for s in args.train_subset.split(",")):
                raise ValueError('Train splits should be named like "train*".')
        return cls(args, tgt_dict_st, tgt_dict_tts)

    def build_criterion(self, args):
        from fairseq import criterions

        if self.data_cfg.prepend_tgt_lang_tag and args.ignore_prefix_size != 1:
            raise ValueError(
                'Please set "--ignore-prefix-size 1" since '
                "target language ID token is prepended as BOS."
            )
        return criterions.build_criterion(args, self)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        is_train_split = split.startswith("train")
        concat_dataset = []
        if is_train_split:
            if self.args.st_data:
                st_dataset = self.load_st_dataset(split, epoch)
                concat_dataset.append(ModalityDatasetItem(
                    "st",
                    st_dataset,
                    [self.args.max_source_positions, self.args.max_target_positions],
                    self.args.max_tokens_st,
                    self.args.batch_size_st,
                ))
            if self.args.supervised_s2st_training:
                s2st_dataset = self.load_s2st_dataset(split, epoch)
                concat_dataset.append(ModalityDatasetItem(
                    "s2st",
                    s2st_dataset,
                    [self.args.max_source_positions, self.args.max_target_audio_positions],
                    self.args.max_tokens,
                    self.args.batch_size,
                ))
            if self.args.tts_data:
                tts_dataset = self.load_tts_dataset(split, epoch)
                concat_dataset.append(ModalityDatasetItem(
                    "tts",
                    tts_dataset,
                    [self.args.max_target_positions, self.args.max_target_audio_positions],
                    self.args.max_tokens_tts,
                    self.args.batch_size_tts,
                ))
        else:
            if self.args.validate_task == "st":
                st_dataset = self.load_st_dataset(split, epoch)
                concat_dataset.append(ModalityDatasetItem(
                    "st",
                    st_dataset,
                    [self.args.max_source_positions, self.args.max_target_positions],
                    self.args.max_tokens_st,
                    self.args.batch_size_st,
                ))
            elif self.args.validate_task == "s2st":
                s2st_dataset = self.load_s2st_dataset(split, epoch)
                concat_dataset.append(ModalityDatasetItem(
                    "s2st",
                    s2st_dataset,
                    [self.args.max_source_positions, self.args.max_target_audio_positions],
                    self.args.max_tokens,
                    self.args.batch_size,
                ))
            elif self.args.validate_task == "tts":
                tts_dataset = self.load_tts_dataset(split, epoch)
                concat_dataset.append(ModalityDatasetItem(
                    "tts",
                    tts_dataset,
                    [self.args.max_target_positions, self.args.max_target_audio_positions],
                    self.args.max_tokens_tts,
                    self.args.batch_size_tts,
                ))
        self.datasets[split] = MultiModalityDataset(concat_dataset)

    def load_st_dataset(self, split, epoch=1):
        is_train_split = split.startswith("train")
        st_dataset = SpeechToTextDatasetModifiedCreator.from_tsv(
            root=self.args.st_data,
            cfg=self.data_cfg,
            splits=split,
            tgt_dict=self.tgt_dict_st,
            pre_tokenizer=self.pre_tokenizer,
            bpe_tokenizer=self.bpe_tokenizer,
            is_train_split=is_train_split,
            epoch=epoch,
            seed=self.args.seed,
            speaker_to_id=None,
            multitask=None if split.startswith("test") else self.multitask_tasks,
        )
        return st_dataset

    def load_tts_dataset(self, split, epoch=1):
        is_train_split = split.startswith("train")
        tts_dataset = TextToSpeechDatasetModifiedCreator.from_tsv(
            root=self.args.tts_data,
            cfg=self.data_cfg,
            splits=split,
            tgt_dict=self.tgt_dict_tts,
            pre_tokenizer=None,
            bpe_tokenizer=None,
            is_train_split=is_train_split,
            epoch=epoch,
            seed=self.args.seed,
            n_frames_per_step=self.args.n_frames_per_step,
            speaker_to_id=self.speaker_to_id,
        )
        return tts_dataset
    
    def load_s2st_dataset(self, split, epoch=1):
        is_train_split = split.startswith("train")
        s2st_dataset = SpeechToSpeechDatasetModifiedCreator.from_tsv(
            root=self.args.data,
            cfg=self.data_cfg,
            splits=split,
            target_is_code=False,
            tgt_dict=None,
            is_train_split=is_train_split,
            n_frames_per_step=self.args.n_frames_per_step,
            multitask=None if split.startswith("test") else self.multitask_tasks,
        )
        return s2st_dataset

    @property
    def target_dictionary(self):
        return self.tgt_dict_st

    @property
    def target_dictionary_tts(self):
        return self.tgt_dict_tts

    @property
    def source_dictionary(self):
        return None
    
    def max_positions(self):
        return self.args.max_source_positions, self.args.max_target_positions, self.args.max_target_audio_positions

    def build_model(self, args, from_checkpoint=False):
        args.input_feat_per_channel = self.data_cfg.input_feat_per_channel
        args.input_channels = self.data_cfg.input_transformed_channels
        args.target_speaker_embed = self.data_cfg.target_speaker_embed is not None
        args.n_frames_per_step = self.args.n_frames_per_step
        args.pitch_min = self.data_cfg.config["features"].get("pitch_min", None)
        args.pitch_max = self.data_cfg.config["features"].get("pitch_max", None)
        args.energy_min = self.data_cfg.config["features"].get("energy_min", None)
        args.energy_max = self.data_cfg.config["features"].get("energy_max", None)
        model = super().build_model(args, from_checkpoint)
        return model

    def build_tokenizer(self, args):
        logger.info(f"pre-tokenizer: {self.data_cfg.pre_tokenizer}")
        return encoders.build_tokenizer(Namespace(**self.data_cfg.pre_tokenizer))

    def build_bpe(self, args):
        logger.info(f"tokenizer: {self.data_cfg.bpe_tokenizer}")
        return encoders.build_bpe(Namespace(**self.data_cfg.bpe_tokenizer))

    def train_step(
        self, sample, model, criterion, optimizer, update_num, ignore_grad=False
    ):
        sample["update_num"] = update_num
        return super().train_step(sample, model, criterion, optimizer, update_num, ignore_grad)

    def get_batch_iterator(
        self,
        dataset,
        max_tokens=None,
        max_sentences=None,
        max_positions=None,
        ignore_invalid_inputs=False,
        required_batch_size_multiple=1,
        seed=1,
        num_shards=1,
        shard_id=0,
        num_workers=0,
        epoch=0,
        data_buffer_size=0,
        disable_iterator_cache=False,
        skip_remainder_batch=False,
        grouped_shuffling=False,
        update_epoch_batch_itr=False,
    ):
        num_dataset = len(dataset.datasets)
        if num_dataset == 1:
            mult_ratio = [1.0]
        elif num_dataset == 2:
            mult_ratio = [
                self.args.st_sample_ratio, 
                self.args.tts_sample_ratio,
            ]
        elif num_dataset == 3:
            mult_ratio = [
                self.args.st_sample_ratio, 
                self.args.s2st_sample_ratio,
                self.args.tts_sample_ratio,
            ]

        # initialize the dataset with the correct starting epoch
        dataset.set_epoch(epoch)

        batch_samplers = dataset.get_batch_samplers(
            mult_ratio, required_batch_size_multiple, seed
        )

        # return a reusable, sharded iterator
        epoch_iter = GroupedEpochBatchIterator(
            dataset=dataset,
            collate_fn=dataset.collater,
            batch_samplers=batch_samplers,
            seed=seed,
            num_shards=num_shards,
            shard_id=shard_id,
            num_workers=num_workers,
            epoch=epoch,
            mult_rate=1,
            buffer_size=data_buffer_size,
            skip_remainder_batch=skip_remainder_batch,
        )
        self.dataset_to_epoch_iter[dataset] = {}  # refresh it every epoch
        return epoch_iter
    
    def build_generator(
        self,
        models,
        args,
        seq_gen_cls=None,
        extra_gen_cls_kwargs=None,
    ):  
        from ComSpeech.generator.comspeech_generator import ComSpeechGenerator

        lang_token_ids_aux = {
            i
            for s, i in self.tgt_dict_st.indices.items()
            if SpeechToTextDatasetModified.is_lang_tag(s)
        }

        extra_gen_cls_kwargs = {}
        extra_gen_cls_kwargs[
            "symbols_to_strip_from_output"
        ] = lang_token_ids_aux

        seq_generator = ComSpeechGenerator(
            models,
            args,
            self.data_cfg,
            self.target_dictionary,
            tgt_dict_tts=self.target_dictionary_tts,
            **extra_gen_cls_kwargs,
        )
        
        return seq_generator