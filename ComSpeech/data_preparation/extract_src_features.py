#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
from pathlib import Path
import shutil
import requests
import tarfile
from typing import Optional, Tuple

import pandas as pd
import torchaudio
from torch import Tensor
from torch.utils.data import Dataset
from tqdm import tqdm

from fairseq.data.audio.audio_utils import convert_waveform
from examples.speech_to_text.data_utils import (
    create_zip,
    extract_fbank_features,
    get_zip_manifest,
    load_df_from_tsv,
    save_df_to_tsv,
)

log = logging.getLogger(__name__)


MANIFEST_COLUMNS = ["id", "src_audio", "src_n_frames"]


class CoVoST(Dataset):
    """Create a Dataset for CoVoST (https://github.com/facebookresearch/covost).

    Args:
        root (str): root path to the dataset and generated manifests/features
        source_language (str): source (audio) language
        target_language (str, optional): target (text) language,
        None for no translation (default: None)
        version (int, optional): CoVoST version. (default: 2)
        download (bool, optional): Whether to download the dataset if it is not
        found at root path. (default: ``False``).
    """

    COVOST_URL_TEMPLATE = (
        "https://dl.fbaipublicfiles.com/covost/"
        "covost_v2.{src_lang}_{tgt_lang}.tsv.tar.gz"
    )

    VERSIONS = {2}
    SPLITS = ["train", "dev", "test"]

    XX_EN_LANGUAGES = {
        1: ["fr", "de", "nl", "ru", "es", "it", "tr", "fa", "sv-SE", "mn", "zh-CN"],
        2: [
            "fr",
            "de",
            "es",
            "ca",
            "it",
            "ru",
            "zh-CN",
            "pt",
            "fa",
            "et",
            "mn",
            "nl",
            "tr",
            "ar",
            "sv-SE",
            "lv",
            "sl",
            "ta",
            "ja",
            "id",
            "cy",
        ],
    }
    EN_XX_LANGUAGES = {
        1: [],
        2: [
            "de",
            "tr",
            "fa",
            "sv-SE",
            "mn",
            "zh-CN",
            "cy",
            "ca",
            "sl",
            "et",
            "id",
            "ar",
            "ta",
            "lv",
            "ja",
        ],
    }

    def __init__(
        self,
        root: str,
        split: str,
        source_language: str,
        target_language: Optional[str] = None,
        version: int = 2,
    ) -> None:
        assert version in self.VERSIONS and split in self.SPLITS
        assert source_language is not None
        self.no_translation = target_language is None
        if not self.no_translation:
            assert "en" in {source_language, target_language}
            if source_language == "en":
                assert target_language in self.EN_XX_LANGUAGES[version]
            else:
                assert source_language in self.XX_EN_LANGUAGES[version]
        else:
            # Hack here so that we can get "split" column from CoVoST TSV.
            # Note that we use CoVoST train split for ASR which is an extension
            # to Common Voice train split.
            target_language = "de" if source_language == "en" else "en"

        self.root: Path = Path(root)

        cv_tsv_path = self.root / "validated.tsv"
        assert cv_tsv_path.is_file()

        covost_url = self.COVOST_URL_TEMPLATE.format(
            src_lang=source_language, tgt_lang=target_language
        )
        covost_archive = self.root / Path(covost_url).name
        if not covost_archive.is_file():
            response = requests.get(covost_url, stream=True)
            with open(covost_archive, 'wb') as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
        with tarfile.open(covost_archive, "r:gz") as tar:
            tar.extractall(path=self.root)

        cv_tsv = load_df_from_tsv(cv_tsv_path)
        covost_tsv = load_df_from_tsv(
            self.root / Path(covost_url).name.replace(".tar.gz", "")
        )
        df = pd.merge(
            left=cv_tsv[["path", "sentence", "client_id"]],
            right=covost_tsv[["path", "translation", "split"]],
            how="inner",
            on="path",
        )
        if split == "train":
            df = df[(df["split"] == split) | (df["split"] == f"{split}_covost")]
        else:
            df = df[df["split"] == split]
        data = df.to_dict(orient="index").items()
        data = [v for k, v in sorted(data, key=lambda x: x[0])]
        self.data = []
        for e in data:
            try:
                path = self.root / "clips" / e["path"]
                _ = torchaudio.info(path.as_posix())
                self.data.append(e)
            except RuntimeError:
                pass

    def __getitem__(
        self, n: int
    ) -> Tuple[Tensor, int, str, str, Optional[str], str, str]:
        """Load the n-th sample from the dataset.

        Args:
            n (int): The index of the sample to be loaded

        Returns:
            tuple: ``(waveform, sample_rate, sentence, translation, speaker_id,
            sample_id)``
        """
        data = self.data[n]
        path = self.root / "clips" / data["path"]
        waveform, sample_rate = torchaudio.load(path)
        sentence = data["sentence"]
        translation = None if self.no_translation else data["translation"]
        speaker_id = data["client_id"]
        _id = data["path"].replace(".mp3", "")
        return waveform, sample_rate, sentence, translation, speaker_id, _id

    def __len__(self) -> int:
        return len(self.data)


class CVSS_C(CoVoST):
    def __init__(
        self,
        cvss_root: str,
        covost_root: str,
        split: str,
        source_language: str,
        target_language: Optional[str] = None,
        version: int = 2,
    ) -> None:
        super().__init__(covost_root, split, source_language, target_language, version)

        self.cvss_root = cvss_root
        self.split = split
        with open(cvss_root / f"{split}.tsv", "r") as f:
            target_data = f.read().splitlines()
            target_data = [x.split("\t") for x in target_data]
            target_dict = {k: v for k, v in target_data}
        
        self.s2s_data = []
        for e in self.data:
            if e["path"] in target_dict:
                e["translation"] = target_dict[e["path"]]
                self.s2s_data.append(e)
    
    def __getitem__(
        self, n: int
    ) -> Tuple[Tensor, int, str, str, Optional[str], str, str]:
        """Load the n-th sample from the dataset.

        Args:
            n (int): The index of the sample to be loaded

        Returns:
            tuple: ``(waveform, sample_rate, sentence, translation, speaker_id,
            sample_id)``
        """
        data = self.s2s_data[n]
        src_path = self.root / "clips" / data["path"]
        src_waveform, src_sample_rate = torchaudio.load(src_path)
        tgt_path = self.cvss_root / self.split / f"{data['path']}.wav"
        tgt_waveform, tgt_sample_rate = torchaudio.load(tgt_path)
        sentence = data["sentence"]
        translation = data["translation"]
        speaker_id = data["client_id"]
        _id = data["path"].replace(".mp3", "")
        
        return src_waveform, src_sample_rate, tgt_waveform, tgt_sample_rate, sentence, translation, speaker_id, _id

    def __len__(self) -> int:
        return len(self.s2s_data)


def process(args):
    output_root = Path(args.output_root)
    output_root.mkdir(exist_ok=True)

    source_root = output_root / "src_fbank80"
    source_zip_path = output_root / f"{source_root.name}.zip"

    src_lang = args.src_lang
    covost_root = Path(args.covost_data_root) / src_lang
    cvss_root = Path(args.cvss_data_root) / f"{src_lang}-en"
    if not covost_root.is_dir():
        raise NotADirectoryError(f"{covost_root} does not exist")
    if not cvss_root.is_dir():
        raise NotADirectoryError(f"{cvss_root} does not exist")
    
    source_root.mkdir(exist_ok=True)
    print(f"Extracting source features for {src_lang}-en...")
    for split in CoVoST.SPLITS:
        dataset = CVSS_C(cvss_root, covost_root, split, src_lang, "en")
        for waveform, sample_rate, _, _, _, _, _, utt_id in tqdm(dataset):
            src_sample_rate = 16_000
            waveform, sample_rate = convert_waveform(
                waveform, sample_rate, to_mono=True,
                to_sample_rate=src_sample_rate
            )
            extract_fbank_features(
                waveform, sample_rate, source_root / f"{utt_id}.npy"
            )

    print("ZIPing source audios/features...")
    create_zip(source_root, source_zip_path)
    shutil.rmtree(source_root)
    
    print("Fetching ZIP manifest...")
    src_audio_paths, src_audio_lengths = get_zip_manifest(
        source_zip_path,
        is_audio=False
    )

    # Generate TSV manifest
    print("Generating manifest...")
    for split in CoVoST.SPLITS:
        manifest = {c: [] for c in MANIFEST_COLUMNS}
        dataset = CVSS_C(cvss_root, covost_root, split, src_lang, "en")
        for _, _, _, _, _, _, _, utt_id in tqdm(dataset):
            manifest["id"].append(utt_id)
            manifest["src_audio"].append(src_audio_paths[utt_id])
            manifest["src_n_frames"].append(src_audio_lengths[utt_id])
        df = pd.DataFrame.from_dict(manifest)
        save_df_to_tsv(df, output_root / f"{split}.tsv")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cvss-data-root", required=True, type=str,
        help="data root of cvss-c",
    )
    parser.add_argument(
        "--covost-data-root", required=True, type=str,
        help="data root of covost2",
    )
    parser.add_argument(
        "--output-root", required=True, type=str,
        help="output root",
    )
    parser.add_argument(
        "--src-lang", required=True, type=str,
        help="source languages",
    )
    args = parser.parse_args()

    process(args)


if __name__ == "__main__":
    main()
