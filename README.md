# ComSpeech

[![arXiv](https://img.shields.io/badge/arXiv-2406.07289-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2406.07289)
[![project](https://img.shields.io/badge/%F0%9F%8E%A7%20Demo-Listen%20to%20ComSpeech-orange.svg)](https://ictnlp.github.io/ComSpeech-Site/)
[![model](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-ComSpeech_Models-blue.svg)](https://huggingface.co/ICTNLP/ComSpeech_Models/tree/main)
[![code](https://img.shields.io/badge/Github-Codes-keygen.svg?logo=github)](https://github.com/ictnlp/ComSpeech)

> **Authors: [Qingkai Fang](https://fangqingkai.github.io/), [Shaolei Zhang](https://zhangshaolei1998.github.io/), [Zhengrui Ma](https://scholar.google.com.hk/citations?user=dUgq6tEAAAAJ), [Min Zhang](https://scholar.google.com.hk/citations?user=CncXH-YAAAAJ), [Yang Feng*](https://people.ucas.edu.cn/~yangfeng?language=en)**

Code for ACL 2024 paper "[Can We Achieve High-quality Direct Speech-to-Speech Translation without Parallel Speech Data?](https://arxiv.org/abs/2406.07289)".

![](assets/ComSpeech-ZS.png)

<p align="center">
  🎧 Listen to <a href="https://ictnlp.github.io/ComSpeech-Site/">ComSpeech's translated speech</a> 🎧 
</p>

## 💡 Highlights

1. ComSpeech is a general composite S2ST model architecture, which can **seamlessly integrate any pretrained S2TT and TTS models into a direct S2ST model**.
2. ComSpeech surpasses previous two-pass models like UnitY and Translatotron 2 **in both translation quality and decoding speed**.
3. With our proposed training strategy **ComSpeech-ZS**, we **achieve performance comparable to supervised training without using any parallel speech data**.

We also have some other projects on **speech-to-speech translation** that you might be interested in:

1. **StreamSpeech (ACL 2024)**: An "All in One" seamless model for offline and simultaneous speech recognition, speech translation and speech synthesis. [![arXiv](https://img.shields.io/badge/paper-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2406.03049) [![code](https://img.shields.io/badge/code-666666.svg?logo=github)](https://github.com/ictnlp/StreamSpeech)
2. **NAST-S2x (ACL 2024)**: A fast and end-to-end simultaneous speech-to-text/speech translation model. [![arXiv](https://img.shields.io/badge/paper-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2406.06937v1) [![code](https://img.shields.io/badge/code-666666.svg?logo=github)](https://github.com/ictnlp/NAST-S2x)
3. **DASpeech (NeurIPS 2023)**: An non-autoregressive two-pass direct speech-to-speech translation model with high-quality translations and fast decoding speed. [![arXiv](https://img.shields.io/badge/paper-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2310.07403) [![code](https://img.shields.io/badge/code-666666.svg?logo=github)](https://github.com/ictnlp/DASpeech)
4. **CTC-S2UT (ACL 2024 Findings)**: An non-autoregressive textless speech-to-speech translation model with up to 26.81× decoding speedup. [![arXiv](https://img.shields.io/badge/paper-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2406.07330) [![code](https://img.shields.io/badge/code-666666.svg?logo=github)](https://github.com/ictnlp/CTC-S2UT)

## 🔥 Quick Start

### Requirements

- python==3.8, torch==2.1.2
- Install fairseq:

  ```bash
  cd fairseq
  pip install -e .
  ```

### Data Preparation

1. Download [CoVoST 2](https://github.com/facebookresearch/covost) Fr/De/Es-En and [CVSS-C](https://github.com/google-research-datasets/cvss) X-En (21 languages in total) datasets and place them in the `data/` directory. 

2. Download our released data manifests from 🤗[Huggingface](https://huggingface.co/datasets/ICTNLP/ComSpeech_Datasets/tree/main), and also place them in the `data/` directory. The directory should be like the following:

```
data
├── comspeech
│   ├── cvss_de_en
│   ├── cvss_es_en
│   ├── cvss_fr_en
│   └── cvss_x_en
├── covost2
│   └── fr
│       ├── clips
│       ├── dev.tsv
│       ├── invalidated.tsv
│       ├── other.tsv
│       ├── test.tsv
│       ├── train.tsv
│       └── validated.tsv
└── cvss-c
    └── fr-en
        └── mfa.tar.gz
```

3. Extract fbank features for the source speech.

```bash
for src_lang in fr de es; do
    python ComSpeech/data_preparation/extract_src_features.py \
        --cvss-data-root data/cvss-c/ \
        --covost-data-root data/covost2/ \
        --output-root data/cvss-c/${src_lang}-en/src \
        --src-lang $src_lang
done
```

4. Extract mel-spectrogram, duration, pitch, and energy information for the target speech.

```bash
for src_lang in ar ca cy de es et fa fr id it ja lv mn nl pt ru sl sv-SE ta tr zh-CN; do
    mkdir -p data/cvss-c/${src_lang}-en/mfa_align
    tar -xzvf data/cvss-c/${src_lang}-en/mfa.tar.gz -C data/cvss-c/${src_lang}-en/mfa_align/
    python ComSpeech/data_preparation/extract_tgt_features.py \
        --audio-manifest-root data/cvss-c/${src_lang}-en/ \
        --output-root data/cvss-c/${src_lang}-en/tts \
        --textgrid-dir data/cvss-c/${src_lang}-en/mfa_align/speaker/
done
```

5. Replace the path in files in the `data/comspeech/` directory.

```bash
python ComSpeech/data_preparation/fill_data.py
```

### ComSpeech (Supervised Learning)

> [!Note] 
> The following scripts use 4 RTX 3090 GPUs by default. You can adjust `--update-freq`, `--max-tokens-st`, `--max-tokens`, and `--batch-size-tts` depending on your available GPUs.

In the **supervised learning** scenario, we first use the S2TT data and TTS data to pretrain the S2TT and TTS models respectively, and then finetune the entire model using the S2ST data. The following script is an example on the CVSS Fr-En dataset. For De-En and Es-En directions, you only need to change the source language in scripts.

1. Pretrain the S2TT model, and the best checkpoint will be saved at `ComSpeech/checkpoints/st.cvss.fr-en/checkpoint_best.pt`.

```bash
bash ComSpeech/train_scripts/st/train.st.cvss.fr-en.sh
```

2. Pretrain the TTS model, and the best checkpoint will be saved at `ComSpeech/checkpoints/tts.fastspeech2.cvss-fr-en/checkpoint_best.pt`.

```bash
bash ComSpeech/train_scripts/tts/train.tts.fastspeech2.cvss-fr-en.sh
```

3. Finetune the entire model using the S2ST data, and the chekpoints will be saved at `ComSpeech/checkpoints/s2st.fr-en.comspeech`.

```bash
bash ComSpeech/train_scripts/s2st/train.s2st.fr-en.comspeech.sh
```

4.  Average the 5 best checkpoints and test the results on the `test` set.

```bash
bash ComSpeech/test_scripts/generate.fr-en.comspeech.sh
```

> [!Note] 
> To run inference, you need to download the pretrained HiFi-GAN vocoder from [this link](https://drive.google.com/drive/folders/1vJlfkwR7Uyheq2U5HrPnfTm-tzwuNuey) and place it in the `hifi-gan/` directory. 

### ComSpeech-ZS (Zero-shot Learning)

In the **zero-shot learning** scenario, we first pretrain the S2TT model using CVSS Fr/De/Es-En S2TT data, and pretrain the TTS model using CVSS X-En TTS (X∉{Fr,De,Es}) data. Then, we finetune the entire model in two stages using these two parts of the data.

1. Pretrain the S2TT model, and the best checkpoint will be saved at `ComSpeech/checkpoints/st.cvss.fr-en/checkpoint_best.pt`.

```bash
bash ComSpeech/train_scripts/st/train.st.cvss.fr-en.sh
```

2. Pretrain the TTS model, and the best checkpoint will be saved at `ComSpeech/checkpoints/tts.fastspeech2.cvss-x-en/checkpoint_best.pt` (note: this checkpoint is used for experiments on all language pairs in the zero-shot learning scenario).

```bash
bash ComSpeech/train_scripts/tts/train.tts.fastspeech2.cvss-x-en.sh
```

3. Finetune the S2TT model and the vocabulary adaptor using S2TT data (stage 1), and the best checkpoint will be saved at `ComSpeech/checkpoints/st.cvss.fr-en.ctc/checkpoint_best.pt`.

```bash
bash ComSpeech/train_scripts/st/train.st.cvss.fr-en.ctc.sh
```

4. Finetune the entire model using both S2TT and TTS data (stage 2), and the checkpoints will be saved at `ComSpeech/checkpoints/s2st.fr-en.comspeech-zs`.

```bash
bash ComSpeech/train_scripts/s2st/train.s2st.fr-en.comspeech-zs.sh
```

5. Average the 5 best checkpoints and test the results on the `test` set.

```bash
bash ComSpeech/test_scripts/generate.fr-en.comspeech-zs.sh
```

### Checkpoints

We have released the checkpoints for each of the above steps. You can download them from 🤗[HuggingFace](https://huggingface.co/ICTNLP/ComSpeech_Models).

#### Supervised Learning

| Directions | S2TT Pretrain                                                | TTS Pretrain                                                 | ComSpeech                                                    |
| ---------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Fr-En      | [[download](https://huggingface.co/ICTNLP/ComSpeech_Models/resolve/main/checkpoints/st.cvss.fr-en/checkpoint_best.pt?download=true)] | [[download](https://huggingface.co/ICTNLP/ComSpeech_Models/resolve/main/checkpoints/tts.fastspeech2.cvss-fr-en/checkpoint_best.pt?download=true)] | [[download](https://huggingface.co/ICTNLP/ComSpeech_Models/resolve/main/checkpoints/s2st.fr-en.comspeech/average_best_checkpoint.pt?download=true)] |
| De-En      | [[download](https://huggingface.co/ICTNLP/ComSpeech_Models/resolve/main/checkpoints/st.cvss.de-en/checkpoint_best.pt?download=true)] | [[download](https://huggingface.co/ICTNLP/ComSpeech_Models/resolve/main/checkpoints/tts.fastspeech2.cvss-de-en/checkpoint_best.pt?download=true)] | [[download](https://huggingface.co/ICTNLP/ComSpeech_Models/resolve/main/checkpoints/s2st.de-en.comspeech/average_best_checkpoint.pt?download=true)] |
| Es-En      | [[download](https://huggingface.co/ICTNLP/ComSpeech_Models/resolve/main/checkpoints/st.cvss.es-en/checkpoint_best.pt?download=true)] | [[download](https://huggingface.co/ICTNLP/ComSpeech_Models/resolve/main/checkpoints/tts.fastspeech2.cvss-es-en/checkpoint_best.pt?download=true)] | [[download](https://huggingface.co/ICTNLP/ComSpeech_Models/resolve/main/checkpoints/s2st.es-en.comspeech/average_best_checkpoint.pt?download=true)] |

#### Zero-shot Learning

| Directions | S2TT Pretrain                                                | TTS Pretrain                                                 | 1-stage Finetune                                             | 2-stage Finetune                                             |
| ---------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Fr-En      | [[download](https://huggingface.co/ICTNLP/ComSpeech_Models/resolve/main/checkpoints/st.cvss.fr-en/checkpoint_best.pt?download=true)] | [[download](https://huggingface.co/ICTNLP/ComSpeech_Models/resolve/main/checkpoints/tts.fastspeech2.cvss-x-en/checkpoint_best.pt?download=true)] | [[download](https://huggingface.co/ICTNLP/ComSpeech_Models/resolve/main/checkpoints/st.cvss.fr-en.ctc/checkpoint_best.pt?download=true)] | [[download](https://huggingface.co/ICTNLP/ComSpeech_Models/resolve/main/checkpoints/s2st.fr-en.comspeech-zs/average_best_checkpoint.pt?download=true)] |
| De-En      | [[download](https://huggingface.co/ICTNLP/ComSpeech_Models/resolve/main/checkpoints/st.cvss.de-en/checkpoint_best.pt?download=true)] | [[download](https://huggingface.co/ICTNLP/ComSpeech_Models/resolve/main/checkpoints/tts.fastspeech2.cvss-x-en/checkpoint_best.pt?download=true)] | [[download](https://huggingface.co/ICTNLP/ComSpeech_Models/resolve/main/checkpoints/st.cvss.de-en.ctc/checkpoint_best.pt?download=true)] | [[download](https://huggingface.co/ICTNLP/ComSpeech_Models/resolve/main/checkpoints/s2st.de-en.comspeech-zs/average_best_checkpoint.pt?download=true)] |
| Es-En      | [[download](https://huggingface.co/ICTNLP/ComSpeech_Models/resolve/main/checkpoints/st.cvss.es-en/checkpoint_best.pt?download=true)] | [[download](https://huggingface.co/ICTNLP/ComSpeech_Models/resolve/main/checkpoints/tts.fastspeech2.cvss-x-en/checkpoint_best.pt?download=true)] | [[download](https://huggingface.co/ICTNLP/ComSpeech_Models/resolve/main/checkpoints/st.cvss.es-en.ctc/checkpoint_best.pt?download=true)] | [[download](https://huggingface.co/ICTNLP/ComSpeech_Models/resolve/main/checkpoints/s2st.es-en.comspeech-zs/average_best_checkpoint.pt?download=true)] |

## 🖋 Citation

If you have any questions, please feel free to submit an issue or contact `fangqingkai21b@ict.ac.cn`.

If our work is useful for you, please cite as:

```
@inproceedings{fang-etal-2024-can,
    title = {Can We Achieve High-quality Direct Speech-to-Speech Translation without Parallel Speech Data?},
    author = {Fang, Qingkai and Zhang, Shaolei and Ma, Zhengrui and Zhang, Min and Feng, Yang},
    booktitle = {Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics},
    year = {2024},
}
```
