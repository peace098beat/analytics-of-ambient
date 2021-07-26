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
#  - 2. 

# # Utils

# In[1]:


get_ipython().system('git clone https://github.com/AllenDowney/ThinkDSP.git ')


# In[2]:


import sys
sys.path.insert(0, 'ThinkDSP/code/') 
import thinkdsp
import IPython

def play_sound(snd_path):
    wave = thinkdsp.read_wave(f"{snd_path}.wav") # Paste this into the previous examples
    return wave


# # mp4を見る

# In[3]:


from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import japanize_matplotlib
import seaborn

import librosa
import librosa.display


# In[4]:


DATA_DIR = "data/Noize"
assert Path(DATA_DIR).exists()


# In[5]:


m4a_paths = [str(p) for p in list(Path(DATA_DIR).glob("*.m4a")) ]
m4a_paths


# ### m4aをWavに変換

# In[6]:


def convert_wav(snd_path, mono=True):
    import subprocess
    
    if mono:
        wav_path = Path(snd_path).with_suffix(".mono.wav")
        subprocess.call(['ffmpeg', '-i', snd_path, '-ac', "1",
                    str(wav_path)])
    else:
        wav_path = Path(snd_path).with_suffix(".wav")
        subprocess.call(['ffmpeg', '-i', snd_path,
                    str(wav_path)])
    
    return wav_path


# In[7]:


for m4a in m4a_paths:
    wav_p = convert_wav(m4a, mono=True)
    print(wav_p)


# ## 聞いてみる

# In[8]:


from mosqito.classes.Audio import Audio
from mosqito import COLORS


# In[9]:


wave_paths = list(Path(DATA_DIR).glob("*.mono.wav"))
wave_paths = [str(p) for p in wave_paths]
wave_paths.sort()


# In[10]:


def plot_wave(wave_path):
    print(wave_path)
    woodpecker = Audio(wave_path)
    
    woodpecker.compute_loudness(field_type="free")
    woodpecker.compute_sharpness(method="din", skip=0.2)
    woodpecker.compute_roughness()


    woodpecker.signal.plot_2D_Data(
        "time",
        type_plot="curve",
        color_list=COLORS,
    )
    woodpecker.loudness_zwicker.plot_2D_Data(
    "time",
    type_plot="curve",
    color_list=COLORS,
    )
    woodpecker.sharpness["din"].plot_2D_Data(
    "time",
    type_plot="curve",
    color_list=COLORS,
    )
    woodpecker.roughness["Daniel Weber"].plot_2D_Data(
        "time",
        type_plot="curve",
        color_list=COLORS,
        y_min=0,
        y_max=1.4,
    )


    thinkdsp.read_wave(wave_path).play()
#     IPython.display.Audio('sound.wav')

    return woodpecker


# In[ ]:


get_ipython().run_cell_magic('time', '', "wp = plot_wave(wave_paths[0])\nIPython.display.Audio('sound.wav')")


# In[ ]:


plot_wave(wave_paths[1])
IPython.display.Audio('sound.wav')


# In[ ]:


plot_wave(wave_paths[2])
IPython.display.Audio('sound.wav')


# In[ ]:


plot_wave(wave_paths[3])
IPython.display.Audio('sound.wav')


# In[ ]:


plot_wave(wave_paths[4])
IPython.display.Audio('sound.wav')


# In[ ]:


plot_wave(wave_paths[5])
IPython.display.Audio('sound.wav')


# In[ ]:


plot_wave(wave_paths[6])
IPython.display.Audio('sound.wav')


# In[ ]:


plot_wave(wave_paths[7])
IPython.display.Audio('sound.wav')


# In[ ]:


plot_wave(wave_paths[8])
IPython.display.Audio('sound.wav')


# In[ ]:


plot_wave(wave_paths[9])
IPython.display.Audio('sound.wav')


# In[ ]:


plot_wave(wave_paths[10])
IPython.display.Audio('sound.wav')


# In[ ]:




