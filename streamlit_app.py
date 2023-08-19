# 必要なライブラリをインポート
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

# タイトルを設定
st.title("音声の多角的な可視化と特徴点の算出")

# 音声ファイルをアップロード
uploaded_file = st.file_uploader("音声ファイルをアップロードしてください", type=["wav", "mp3"])

if uploaded_file is not None:
    # 音声ファイルを読み込む
    y, sr = librosa.load(uploaded_file, sr=None)
    
    # 音声の波形を可視化
    st.subheader("音声の波形")
    st.write("このグラフは、時間に対する音声の振幅を示しています。")
    plt.figure(figsize=(10, 4))
    librosa.display.waveshow(y, sr=sr)
    st.pyplot(plt)
    
    # スペクトログラムを可視化
    st.subheader("スペクトログラム")
    st.write("このグラフは、時間に対する周波数の強度を示しています。")
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    st.pyplot(plt)
    
    # メルスペクトログラムを可視化
    st.subheader("メルスペクトログラム")
    st.write("このグラフは、時間に対するメル周波数の強度を示しています。")
    M = librosa.feature.melspectrogram(y=y, sr=sr)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(librosa.power_to_db(M, ref=np.max), sr=sr, y_axis='mel', x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    st.pyplot(plt)
    
    # クロマグラムを可視化
    st.subheader("クロマグラム")
    st.write("このグラフは、時間に対する12の異なるピッチクラスの強度を示しています。")
    C = librosa.feature.chroma_cqt(y=y, sr=sr)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(C, sr=sr, x_axis='time', y_axis='chroma')
    plt.colorbar()
    st.pyplot(plt)
    
    # トーンネットワークを可視化
    st.subheader("トーンネットワーク")
    st.write("このグラフは、時間に対する6つのトーンネットワークの強度を示しています。")
    T = librosa.feature.tonnetz(y=y, sr=sr)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(T, sr=sr, x_axis='time', y_axis='tonnetz')
    plt.colorbar()
    st.pyplot(plt)
    
    # 音声の特徴点を算出
    st.subheader("特徴点")
    st.write("以下は、音声から算出されたいくつかの主要な特徴点です。")
    
    # MFCCを算出
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    st.write("MFCCs (Mel-frequency cepstral coefficients): 音声のテクスチャやタイミングの情報をキャッチする特徴量です。")
    st.write(f"MFCCs: {mfccs.mean(axis=1)}")
    
    # クロマ特徴量を算出
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    st.write("Chroma: 12の異なるピッチクラスの強度を示す特徴量です。")
    st.write(f"Chroma: {chroma.mean(axis=1)}")
    
    # スペクトルのコントラストを算出
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    st.write("Spectral Contrast: スペクトルのピークとバレーの違いを示す特徴量です。")
    st.write(f"Spectral Contrast: {contrast.mean(axis=1)}")
    
    # Zero Crossing Rateを算出
    zcr = librosa.feature.zero_crossing_rate(y)
    st.write("Zero Crossing Rate: 音声信号がゼロを越える回数を示す特徴量です。")
    st.write(f"Zero Crossing Rate: {np.mean(zcr)}")
    
    # Spectral Roll-offを算出
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    st.write("Spectral Roll-off: この周波数以下にスペクトルの指定された割合が存在するという特徴量です。")
    st.write(f"Spectral Roll-off: {np.mean(rolloff)}")
