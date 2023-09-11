
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import pandas as pd
import ffmpeg
import io

# Streamlitのページ設定
st.set_page_config(page_title="音声解析Webアプリ", layout="wide")

# ヘッダー
st.title("音声解析Webアプリ")
st.caption("Created by Daiki Ito")
st.subheader("音声の多角的な可視化と特徴点の算出")

# 音声ファイルのアップロード
uploaded_file = st.file_uploader("音声ファイルをアップロードしてください", type=["wav", "mp3", "m4a"])

def convert_m4a_to_mp3(m4a_data):
    """Convert m4a data to mp3"""
    input_audio = ffmpeg.input('pipe:0', format='m4a')
    output_audio = ffmpeg.output(input_audio, 'pipe:1', format='mp3')
    _, mp3_data = ffmpeg.run(output_audio, input=m4a_data, capture_stdout=True, capture_stderr=True)
    return io.BytesIO(mp3_data)

if uploaded_file is not None:
    file_extension = uploaded_file.name.split('.')[-1]  # Get the file extension

    if file_extension == 'm4a':
        uploaded_file = convert_m4a_to_mp3(uploaded_file.read())

    y, sr = librosa.load(uploaded_file, sr=None)
    display_audio_insights(y, sr)
    visualize_audio(y, sr)
    analyze_audio_features(y, sr)

# ... Rest of the code ...

# Copyright
st.markdown('© 2022-2023 Daiki Ito. All Rights Reserved.')
