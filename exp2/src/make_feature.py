#!/usr/bin/env python
# coding: utf-8

# # Analytics of Ambient
# 
# - 作業環境
#     - https://miro.com/app/board/o9J_l9jSAOo=/
#     - https://drive.google.com/drive/folders/1wnKK2SaN9DgPUa2PFSJbwizaDHOHitGc?usp=sharing
#  

# ## やりたいこと
# 
#  - ある音のラウドネスとシャープネスなどの指標でマッピングする
#  
# ## 課題
# 
#  - 音を分割するひつようがある. 
#      - N 秒
#  - 心理音響指標をpythonで出せるか
#      - [MoSQITo/tuto_signal_basic_operations.ipynb at master · Eomys/MoSQITo · GitHub](https://github.com/Eomys/MoSQITo/blob/master/tutorials/tuto_signal_basic_operations.ipynb)
# 
# ## 進め方
# 
#  - 1. 音を聴く.
#  - 2. 心理音響指標算出
#  
#  
#  ## 結果
#   - 音源が長くて心理音響指標が計算できない
#  
#  ## 変更
#   - 音源を5秒刻みに変更
#   - 中間3秒だけで心理音響指標を算出
#   - jsonで保存

# # 参考
# 
#  - 周波数重心
#      https://www.geidai.ac.jp/~marui/r_program/spectrum.html

# In[92]:


# # 出力形式
# 
#  - グラフ
#      - Audio
#      - Loudness
#      - Sharpness
#      - Roughness
#  - json
#      - filename
#      - length
#      - loudness.mean
#      - sharpness.mean
#      - roughness.mean


from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import japanize_matplotlib

from tqdm import tqdm

from mosqito.functions.shared.load import load
from mosqito.functions.loudness_zwicker.comp_loudness import comp_loudness
from mosqito.functions.loudness_zwicker.comp_loudness import comp_loudness
from mosqito.functions.sharpness.comp_sharpness import comp_sharpness
from mosqito.functions.roughness_danielweber.comp_roughness import comp_roughness




def output(wav_path):

    fig_path = Path(wav_path).with_suffix(".jpg")
    json_path = Path(wav_path).with_suffix(".json")
    sig_path = Path(wav_path).with_suffix(".pkl")

    print(wav_path)
    print(fig_path)
    print(json_path)

    # SIgnal
    signal, fs = load( None, wav_path)

    N = len(signal)
    y_signal = signal
    t_signal = np.linspace(0, N*(1/fs), N)


    # Loudness
    print("Loudness ..")
    loudness = comp_loudness(False, signal, fs, field_type = 'free')
    loudness_sone = loudness['values']
    loudness_time = np.linspace(0,0.002*(loudness_sone.size - 1), loudness_sone.size)

    # Sharpness
    print("Sharpness ..")
    sharpness = comp_sharpness(False, signal, fs, method="din", skip=0.2)

    sharp_acum = sharpness['values']
    sharp_time = np.linspace(0,0.002*(sharp_acum.size - 1), sharp_acum.size)


    # Roughness
    print("Roughness ..")
    roughness = comp_roughness(signal, fs, overlap=0)

    y_rough = roughness['values']
    t_rough = roughness['time']


    data_json = {
        "wav_path" : wav_path,
        "loudness" : np.mean(loudness_sone), 
        "sharpness" : np.mean(sharp_acum), 
        "roughness" : np.mean(y_rough), 
    }

    data_sig ={
        "y_loudness" : loudness_sone,
        "x_loudness" : loudness_time,
        "y_sharpness" : sharp_acum,
        "x_sharpness" : sharp_time,
        "y_roughness" : y_rough,
        "x_roughness" : t_rough,
    }

    import json
    with open(json_path, mode='wt', encoding='utf-8') as fp:
        json.dump(data_json, fp, ensure_ascii=False, indent=2)
    del data_json
    # ロード
    with open(json_path, mode='rt', encoding='utf-8') as fp:
        data = json.load(fp)
        print(data)

    import pickle
    with open(sig_path, mode="wb") as fp:
    	pickle.dump(data_sig, fp)
    del data_sig

    with open(sig_path, mode='rb') as fp:
        data_sig = pickle.load(fp)
        print(data_sig)


    plt.rcParams["font.size"] = 16

    fig, ax = plt.subplots(4, 1, figsize=(20, 10), sharex=True)

    # signal
    plt.sca(ax[0])

    fill_sig = np.abs(y_signal) ** 2
    plt.fill_between(t_signal, fill_sig, 0, color="b")
    plt.fill_between(t_signal, -1*fill_sig, 0, color="b")

    plt.ylabel("Signal")
    plt.grid()

    # Loudness
    plt.sca(ax[1])
    plt.plot(loudness_time, loudness_sone, color="r")
    plt.hlines(y=np.mean(loudness_sone), xmin=0, xmax=5, colors='gray', linestyle='dashed', linewidth=3)
    plt.ylabel("Loudness [Sone]")
    plt.grid()

    # Sharpness
    plt.sca(ax[2])
    plt.plot(sharp_time, sharp_acum, color="r")
    plt.hlines(y=np.mean(sharp_acum), xmin=0, xmax=5, colors='gray', linestyle='dashed', linewidth=3)
    plt.ylabel("Sharpness [acum]")
    plt.grid()

    # Roughness
    plt.sca(ax[3])
    plt.plot(t_rough, y_rough, color="r")
    plt.hlines(y=np.mean(y_rough), xmin=0, xmax=5, colors='gray', linestyle='dashed', linewidth=3)
    plt.ylabel("Roughness [asper]")
    plt.grid()

    plt.xlabel("Time [s]")

    plt.suptitle(wav_path)

    plt.tight_layout()
    plt.savefig(fig_path)

    plt.close()


# M1じゃ動かない. 実行途中でエラーが発生する
# from concurrent import futures
# with futures.ThreadPoolExecutor(max_workers=2) as executor:
# # with futures.ProcessPoolExecutor(max_workers=2) as executor:
#     results = list(tqdm(executor.map(output, wave_paths), total=len(wave_paths)))

# In[94]:
INTERIM_DIR = "../data/01_raw/1_boubakiki_mono"

wave_paths = list(Path(INTERIM_DIR).glob("*.wav"))
wave_paths = [str(p) for p in wave_paths]
wave_paths.sort()

for w in tqdm(wave_paths):
    output(w)



# INTERIM_DIR = "../data/01_raw/2_record_mono"

# wave_paths = list(Path(INTERIM_DIR).glob("*.wav"))
# wave_paths = [str(p) for p in wave_paths]
# wave_paths.sort()

# for w in tqdm(wave_paths):
    output(w)
