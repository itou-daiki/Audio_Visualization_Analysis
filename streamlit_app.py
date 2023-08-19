import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import pandas as pd

# Streamlitのページ設定
st.set_page_config(page_title="音声解析Webアプリ", layout="wide")

# ヘッダー
st.title("音声解析Webアプリ")
st.caption("Created by Daiki Ito")
st.subheader("音声の多角的な可視化と特徴点の算出")

# 音声ファイルのアップロード
uploaded_file = st.file_uploader("音声ファイルをアップロードしてください", type=["wav", "mp3"])

# 音声の特徴点をテーブル形式で表示
def display_features(df, title, description):
    df.index = df.index + 1
    st.subheader(title)
    st.write(description)
    st.table(df)

# 音声の可視化
def visualize_audio(y, sr):
    # 音声の波形
    st.subheader("音声の波形")
    plt.figure(figsize=(10, 4))
    librosa.display.waveshow(y, sr=sr)
    st.pyplot(plt)

    # スペクトログラム
    st.subheader("スペクトログラム")
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    st.pyplot(plt)

    # メルスペクトログラム
    st.subheader("メルスペクトログラム")
    M = librosa.feature.melspectrogram(y=y, sr=sr)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(librosa.power_to_db(M, ref=np.max), sr=sr, y_axis='mel', x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    st.pyplot(plt)

    # クロマグラム
    st.subheader("クロマグラム")
    C = librosa.feature.chroma_cqt(y=y, sr=sr)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(C, sr=sr, x_axis='time', y_axis='chroma')
    plt.colorbar()
    st.pyplot(plt)

    # トーンネットワーク
    st.subheader("トーンネットワーク")
    T = librosa.feature.tonnetz(y=y, sr=sr)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(T, sr=sr, x_axis='time', y_axis='tonnetz')
    plt.colorbar()
    st.pyplot(plt)

# 音声の特徴点の算出と表示
def analyze_audio_features(y, sr):
    # MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_df = pd.DataFrame({"MFCCs": [f"{val:.4f}" for val in mfccs.mean(axis=1)]})
    display_features(mfccs_df, "MFCCs (Mel-frequency cepstral coefficients)", "音声のテクスチャやタイミングの情報をキャッチする特徴量です。")

    # クロマ特徴量
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_df = pd.DataFrame({"Chroma": [f"{val:.4f}" for val in chroma.mean(axis=1)]})
    display_features(chroma_df, "Chroma", "12の異なるピッチクラスの強度を示す特徴量です。")

    # スペクトルのコントラスト
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    contrast_df = pd.DataFrame({"Spectral Contrast": [f"{val:.4f}" for val in contrast.mean(axis=1)]})
    display_features(contrast_df, "Spectral Contrast", "スペクトルのピークとバレーの違いを示す特徴量です。")

    # Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y)
    zcr_df = pd.DataFrame({"Zero Crossing Rate": [f"{np.mean(zcr):.4f}"]})
    display_features(zcr_df, "Zero Crossing Rate", "音声信号がゼロを越える回数を示す特徴量です。")

    # Spectral Roll-off
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    rolloff_df = pd.DataFrame({"Spectral Roll-off": [f"{np.mean(rolloff):.4f}"]})
    display_features(rolloff_df, "Spectral Roll-off", "この周波数以下にスペクトルの指定された割合が存在するという特徴量です。")

# 音声のインサイトを表示
def display_audio_insights(y, sr):
    st.subheader("音声のインサイト")

    # 音声の長さ
    duration = librosa.get_duration(y=y, sr=sr)
    st.write(f"音声の長さ: {duration:.2f} 秒")

    # 平均的な音量
    avg_volume = np.mean(np.abs(y))
    st.write(f"平均的な音量: {avg_volume:.4f}")

    # 最大音量
    max_volume = np.max(np.abs(y))
    st.write(f"最大音量: {max_volume:.4f}")

    # 平均的なピッチ
    pitches, magnitudes = librosa.core.piptrack(y=y, sr=sr)
    avg_pitch = np.mean(pitches[pitches > 0])
    st.write(f"平均的なピッチ: {avg_pitch:.4f} Hz")

if uploaded_file is not None:
    y, sr = librosa.load(uploaded_file, sr=None)
    visualize_audio(y, sr)
    analyze_audio_features(y, sr)
    display_audio_insights(y, sr)

# Copyright
st.markdown('© 2022-2023 Daiki Ito. All Rights Reserved.')
