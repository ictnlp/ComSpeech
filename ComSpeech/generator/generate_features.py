# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import ast
import numpy as np
from pathlib import Path
import sys
import torch

from fairseq import checkpoint_utils, options, tasks, utils
from fairseq.logging import progress_bar
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.data.audio.text_to_speech_dataset import TextToSpeechDataset


def make_parser():
    parser = options.get_speech_generation_parser()
    options.add_generation_args(parser)
    return parser

def build_generator(task, models, args):
    from ComSpeech.generator.comspeech_generator import ComSpeechGenerator

    return ComSpeechGenerator(
        models,
        args,
        task.data_cfg,
        task.target_dictionary,
        tgt_dict_tts=task.target_dictionary_tts,
    )

def postprocess_results(dataset: TextToSpeechDataset, sample, hypos):
    def to_np(x):
        return None if x is None else x.detach().cpu().numpy()

    sample_ids = sample["id"].tolist()
    feat_preds = [to_np(hypo["feature"]) for hypo in hypos]

    return zip(sample_ids, feat_preds)

def dump_result(args, sample_id, feat_pred):
    out_root = Path(args.results_path)
    feat_dir = out_root / "feat"
    feat_dir.mkdir(exist_ok=True, parents=True)
    np.save(feat_dir / f"{sample_id}.npy", feat_pred.transpose(1, 0))

def main(args):
    cfg = convert_namespace_to_omegaconf(args)
    utils.import_user_module(cfg.common)

    use_cuda = torch.cuda.is_available() and not args.cpu
    task = tasks.setup_task(args)
    models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task(
        [args.path],
        task=task,
        arg_overrides=ast.literal_eval(args.model_overrides),
    )
    model = models[0].cuda() if use_cuda else models[0]
    # use the original n_frames_per_step
    task.args.n_frames_per_step = saved_cfg.task.n_frames_per_step
    task.load_dataset(args.gen_subset, task_cfg=saved_cfg.task)

    generator = build_generator(task, [model], args)
    itr = task.get_batch_iterator(
        dataset=task.dataset(args.gen_subset),
        max_tokens=args.max_tokens,
        max_sentences=args.batch_size,
        max_positions=(sys.maxsize, sys.maxsize),
        ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=args.required_batch_size_multiple,
        num_shards=args.num_shards,
        shard_id=args.shard_id,
        num_workers=args.num_workers,
        data_buffer_size=args.data_buffer_size,
    ).next_epoch_itr(shuffle=False)

    Path(args.results_path).mkdir(exist_ok=True, parents=True)
    dataset = task.dataset(args.gen_subset)
    with progress_bar.build_progress_bar(args, itr) as t:
        for sample in t:
            sample = utils.move_to_cuda(sample) if use_cuda else sample
            hypos = generator.generate(model, sample)
            for result in postprocess_results(dataset, sample, hypos):
                dump_result(args, *result)

def cli_main():
    parser = make_parser()
    args = options.parse_args_and_arch(parser)
    main(args)

if __name__ == "__main__":
    cli_main()
