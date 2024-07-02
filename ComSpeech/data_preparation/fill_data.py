import pandas as pd
from examples.speech_to_text.data_utils import (
    load_df_from_tsv,
    save_df_to_tsv
)

database = {}
for lang in ["fr", "de", "es"]:
    for split in ["train", "dev", "test"]:
        path = f"data/cvss-c/{lang}-en/src/{split}.tsv"
        df = load_df_from_tsv(path)
        data = (df.T.to_dict().values())
        for item in data:
            key = item['id'].replace('.mp3', '')
            database[key] = {
                'src_audio': item['src_audio'],
                'src_n_frames': item['src_n_frames'],
            }

for lang in ["ar", "ca", "cy", "de", "es", "et", "fa", "fr", "id", "it", "ja", "lv", "mn", "nl", "pt", "ru", "sl", "sv-SE", "ta", "tr", "zh-CN"]:
    for split in ["train", "dev", "test"]:
        path = f"data/cvss-c/{lang}-en/tts/{split}.tsv"
        df = load_df_from_tsv(path)
        data = (df.T.to_dict().values())
        for item in data:
            key = item['id'].replace('.mp3', '')
            if key in database:
                database[key].update({
                    'tgt_audio': item['audio'],
                    'tgt_n_frames': item['n_frames'],
                    'pitch': item['pitch'],
                    'energy': item['energy'],
                })
            else:
                database[key] = {
                    'tgt_audio': item['audio'],
                    'tgt_n_frames': item['n_frames'],
                    'pitch': item['pitch'],
                    'energy': item['energy'],
                }

for lang in ["fr", "de", "es"]:
    for split in ["train", "dev", "test", "dev.full", "test.full"]:
        path = f"data/comspeech/cvss_{lang}_en/s2s/{split}.tsv"
        df = load_df_from_tsv(path)
        data = (df.T.to_dict().values())
        for item in data:
            key = item['id'].replace('.mp3', '')
            item['src_audio'] = database[key]['src_audio']
            item['src_n_frames'] = database[key]['src_n_frames']
            if 'tgt_audio' in item:
                item['tgt_audio'] = database[key]['tgt_audio']
                item['tgt_n_frames'] = database[key]['tgt_n_frames']
                item['pitch'] = database[key]['pitch']
                item['energy'] = database[key]['energy']
        df = pd.DataFrame.from_dict(data)
        save_df_to_tsv(df, path)

    for split in ["train", "dev", "test"]:
        path = f"data/comspeech/cvss_{lang}_en/speech2unigram/{split}.tsv"
        df = load_df_from_tsv(path)
        data = (df.T.to_dict().values())
        for item in data:
            key = item['id'].replace('.mp3', '')
            item['audio'] = database[key]['src_audio']
            item['n_frames'] = database[key]['src_n_frames']
        df = pd.DataFrame.from_dict(data)
        save_df_to_tsv(df, path)

for split in ["train", "dev", "test"]:
    path = f"data/comspeech/cvss_x_en/tts/{split}.tsv"
    df = load_df_from_tsv(path)
    data = (df.T.to_dict().values())
    for item in data:
        key = item['id'].replace('.mp3', '')
        item['audio'] = database[key]['tgt_audio']
        item['n_frames'] = database[key]['tgt_n_frames']
        item['pitch'] = database[key]['pitch']
        item['energy'] = database[key]['energy']
    df = pd.DataFrame.from_dict(data)
    save_df_to_tsv(df, path)