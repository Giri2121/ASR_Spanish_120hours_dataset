import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import tensorflow as tf
import tensorflow_io as tfio
from tensorflow import keras
from tensorflow.keras.layers import StringLookup



# An integer scalar Tensor. The window length in samples.
frame_length = 256
# An integer scalar Tensor. The number of samples to step.
frame_step = 160
# An integer scalar Tensor. The size of the FFT to apply.
# If not provided, uses the smallest power of 2 enclosing frame_length.
fft_length = 384
epsilon = 1e-8

'''
f_df = pd.read_csv()
trans_list = f_df['transcript'].to_list()
unique_letters = set()
for trans in trans_list:
    for letter in trans:
        unique_letters.add(letter)

unique_letters = list(unique_letters)

'''
unique_letters = [x for x in "abcdefghijklmnopqrstuvwxyzáéíñóúü "]
char_to_num = StringLookup(vocabulary=unique_letters,oov_token="")
num_to_char = StringLookup(vocabulary=char_to_num.get_vocabulary(),oov_token="",invert=True)


def basic_eda(df,data_path=None):
    '''
    :param df: the original dataframe
    :return: dataframe whose transcript length is <= 30
    '''
    #print(df.shape)
    #print(df.head())
    #print(df.columns)
    df['transcript_len'] = df['transcript'].apply(lambda x:len(x))
    df['full_path'] = df['wav_filename'].apply(lambda x:os.path.normpath(data_path)+x)
    df = df[df['transcript_len'] <= 30]
    df = df.sample(frac=1,random_state=42)
    return df


def train_test_split(df):
    split_len = int(round(len(df)*0.80))
    train_df = df[:split_len]
    test_df = df[split_len:]

    print(f' Len of training Data {len(train_df)}')
    print(f'Len if testing data {len(test_df)}')

    return train_df,test_df


def mono_16k_out(filename):
    curr_wave = tf.io.read_file(filename)
    wav,sample_rate = tf.audio.decode_wav(curr_wave,desired_channels = 1)
    wav = tf.squeeze(wav,axis=-1)
    sample_rate = tf.cast(sample_rate,dtype = tf.int64)
    wav = tfio.audio.resample(wav,rate_in = sample_rate,rate_out = 16000)
    return wav


def random_audio_plot(audio_path):
    files = os.listdir(audio_path)
    sample_waves = random.sample(files,4)
    plt.figure(figsize=(12,10))
    for i,wave_ in enumerate(sample_waves):
        converted_wave = mono_16k_out(os.path.join(audio_path,wave_))
        plt.subplot(2,2,i+1)
        plt.plot(converted_wave)
        plt.title(wave_)

    plt.show()


def encode_audio_label(wav_file,label):
    audio = mono_16k_out(wav_file)
    spectrogram = tf.signal.stft(audio,frame_length=frame_length,frame_step=frame_step,fft_length=fft_length)
    spectrogram = tf.math.pow(tf.abs(spectrogram),0.5)
    #normalisation
    mean_ = tf.math.reduce_mean(spectrogram,1,keepdims=True)
    std_dev = tf.math.reduce_std(spectrogram,1,keepdims=True)
    spectrogram = (spectrogram-mean_)/(std_dev + epsilon)

    #processing the label
    label = tf.strings.lower(label)
    label = tf.strings.unicode_split(label,input_encoding='UTF-8')
    label = char_to_num(label)
    return spectrogram,label

def tfto_dataset(train_df,test_df):
    batch_size = 8
    padded_shape = (193,)
    #padded_shapes = ([None],())

    train_ds = tf.data.Dataset.from_tensor_slices(
        (list(train_df['full_path']), list(train_df['transcript']))
    )

    train_ds = (
        train_ds.map(encode_audio_label,num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

    #test dataset
    test_ds = tf.data.Dataset.from_tensor_slices(
        (list(test_df['full_path']),list(test_df['transcript']))
    )

    test_ds = (
        test_ds.map(encode_audio_label,num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

    return train_ds,test_ds

def decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    results = tf.keras.backend.ctc_decode(pred,input_len=input_len,greedy=True)[0][0]
    #iterating over text to get results
    output_text = []
    for result in results:
        result = tf.strings.reduce_join(num_to_char(result)).numpy().decode("UTF-8")
        output_text.append(result)
    return output_text


def CTC_LOSS(y_true,y_pred):
    #computing the CTC loss
    batch_len = tf.cast(tf.shape(y_true)[0],dtype="int64")
    input_len = tf.cast(tf.shape(y_pred)[1],dtype="int64")
    label_len = tf.cast(tf.shape(y_true)[1],dtype="int64")

    input_len = input_len * tf.ones(shape = (batch_len,1),dtype="int64")
    label_len = label_len * tf.ones(shape = (batch_len,1),dtype="int64")

    loss = keras.backend.ctc_batch_cost(y_true,y_pred,input_len,label_len)
    return loss

if __name__ == '__main__':
    path_to_file = r'C:\Users\vikassaigiridhar\Music\spanish_translation\New folder\asr-spanish-v1-carlfm01\asr-spanish-v1-carlfm01\files.csv'
    df = pd.read_csv(path_to_file)
    train_dfs,test_dfs = train_test_split(df)
