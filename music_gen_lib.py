# store the function/object used in the project

# import modules
from __future__ import print_function
import numpy as np
import librosa
import keras
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Dropout, Flatten, Input, Reshape, Add
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, concatenate, LSTM
from keras import backend as K
from keras.utils import np_utils
from keras import regularizers
import time
from keras.engine.topology import Layer
from hyperparams import HyperParams as hp
from keras.callbacks import ModelCheckpoint


def split_data(T, split_idxes):
    """
    give the indexes of training, validation, and testing data
    :param T: label of all data
    :param split_idxes: splitting points of the data
    :return:
    """
    genres = np.unique(T)
    training_idxes = []
    validation_idxes = []
    testing_idxes = []
    for idx, music_genre in enumerate(genres):
        tmp_logidx = music_genre == T
        tmp_idx = np.flatnonzero(tmp_logidx)
        tmp_shuffled_idx = np.random.permutation(tmp_idx)
        tmp_num_examles = len(tmp_shuffled_idx)
        tmp_split_idxes = np.asarray(split_idxes * tmp_num_examles, dtype=np.int)
        training_idxes.append(tmp_shuffled_idx[tmp_split_idxes[0]: tmp_split_idxes[1]])
        validation_idxes.append(tmp_shuffled_idx[tmp_split_idxes[1]: tmp_split_idxes[2]])
        testing_idxes.append(tmp_shuffled_idx[tmp_split_idxes[2]: tmp_split_idxes[3]])
    return np.concatenate(training_idxes), np.concatenate(validation_idxes), np.concatenate(testing_idxes)


def load_original_data():
    """
    load original audio files
    :return:
    """
    import os

    # genre_folders = [x[0] for x in os.walk(data_folder)]
    genre_folders = os.listdir(hp.data_folder)
    X = []
    T = []
    SR = []
    min_length = 0
    for sub_folder in genre_folders:
        genre_path = hp.data_folder + "/" + sub_folder
        audio_files = os.listdir(genre_path)
        for audio_name in audio_files:
            audio_path = genre_path + "/" + audio_name
            x, sr = librosa.core.load(audio_path)  # x = 661794
            if x.shape[0] < 30 * sr:
                x = np.append(x, np.zeros(30 * sr - x.shape[0]))  # insure all files are exactly the same length
                if min_length < x.shape[0]:
                    min_length = x.shape[0]  # report the duration of the minimum audio clip
                    print("This audio last %f seconds, zeros are padded at the end." % (x.shape[0] * 1.0 / sr))
            X.append(x[:30 * sr])
            SR.append(sr)
            T.append(sub_folder)
    return np.asarray(X), np.asarray(SR), np.asarray(T, dtype=str)


# calculate mel-spectrogram
def mel_spectrogram(ys, sr, n_mels=hp.n_mels, hop_size=hp.hop_size, fmax=hp.fmax, pre_emphasis=hp.pre_emphasis):
    """
    calculate the spectrogram in mel scale, refer to documentation of librosa and MFCC tutorial
    :param ys:
    :param sr:
    :param n_mels:
    :param hop_size:
    :param fmax:
    :param pre_emphasis:
    :return:
    """
    if pre_emphasis:
        ys = np.append(ys[0], ys[1:] - pre_emphasis * ys[:-1])
    return librosa.feature.melspectrogram(ys, sr,
                                          n_fft=hp.n_fft,
                                          hop_length=hop_size, n_mels=n_mels,
                                          fmax=fmax)


# batch convert waveform into spectrogram in mel-scale
def batch_mel_spectrogram(X, SR):
    """
    convert all waveforms in R into time * 64 spectrogram in mel scale
    :param X:
    :param SR:
    :return:
    """
    melspec_list = []
    for idx in range(X.shape[0]):
        tmp_melspec = mel_spectrogram(X[idx], SR[idx])
        melspec_list.append(tmp_melspec)
    return np.asarray(melspec_list)


# def segment_spectrogram(input_spectrogram, num_fft_windows=num_fft_windows):
#     # given a spectrogram of a music that's longer than 3 seconds, segment it into relatively independent pieces
#     length_in_fft = input_spectrogram.shape[1]
#     num_segments = int(length_in_fft / num_fft_windows)
#     pass


def baseline_model_32(num_genres=hp.num_genres, input_shape=hp.input_shape):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu', kernel_regularizer=regularizers.l2(0.01),
                     input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 4)))
    model.add(Conv2D(64, (3, 5), activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(MaxPooling2D(pool_size=(2, 4)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.02)))
    model.add(Dropout(0.2))
    model.add(Dense(num_genres, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(decay=1e-5),
                  metrics=['accuracy'])
    return model


def baseline_model_64(num_genres=hp.num_genres, input_shape=hp.input_shape):
    model = Sequential()
    model.add(Conv2D(64, kernel_size=(3, 3),
                     activation='relu', kernel_regularizer=regularizers.l2(0.01),
                     input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 4)))
    model.add(Conv2D(64, (3, 5), activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(MaxPooling2D(pool_size=(2, 4)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.02)))
    model.add(Dropout(0.2))
    model.add(Dense(num_genres, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(1e-4),
                  metrics=['accuracy'])
    return model


def baseline_model_96(num_genres=hp.num_genres, input_shape=hp.input_shape):
    model = Sequential()
    model.add(Conv2D(96, kernel_size=(3, 3),
                     activation='relu', kernel_regularizer=regularizers.l2(0.01),
                     input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 4)))
    model.add(Conv2D(64, (3, 5), activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(MaxPooling2D(pool_size=(2, 4)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.02)))
    model.add(Dropout(0.2))
    model.add(Dense(num_genres, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(decay=1e-5),
                  metrics=['accuracy'])
    return model


def baseline_model_128(num_genres=hp.num_genres, input_shape=hp.input_shape):
    model = Sequential()
    model.add(Conv2D(128, kernel_size=(3, 3),
                     activation='relu', kernel_regularizer=regularizers.l2(0.01),
                     input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 4)))
    model.add(Conv2D(64, (3, 5), activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(MaxPooling2D(pool_size=(2, 4)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.02)))
    model.add(Dropout(0.2))
    model.add(Dense(num_genres, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(decay=1e-5),
                  metrics=['accuracy'])
    return model


def model_nnet1(num_genres=hp.num_genres, input_shape=hp.input_shape):
    inputs = Input(shape=input_shape, name='model_inputs')  # (?, 256, 128, 1)
    # conv1
    conv1 = Conv2D(256, kernel_size=(256, 4), activation='relu',
                   input_shape=input_shape, kernel_regularizer=regularizers.l2(0.02))
    conv1_temp = conv1(inputs)
    drop1 = Dropout(rate=0.2)
    conv1_outputs = drop1(conv1_temp)  # (?, 1, 125, 256)
    # hidden_1
    mp = MaxPooling2D(pool_size=(1, 125))
    hidden_1 = mp(conv1_outputs)  # (?, 1, 1, 256)
    # conv2
    conv2 = Conv2D(256, kernel_size=(1, 4), activation='relu',
                   kernel_regularizer=regularizers.l2(0.01),
                   padding='same')
    conv2_outputs = conv2(conv1_outputs)  # (?, 1, 125, 256)
    # hidden_2
    ap = AveragePooling2D(pool_size=(1, 125))
    hidden_2 = ap(conv2_outputs)  # (?, 1, 1, 256)
    # hidden_3
    conv3 = Conv2D(128, kernel_size=(1, 4), activation='relu',
                   kernel_regularizer=regularizers.l2(0.01),
                   padding='same')
    hidden_3 = mp(conv3(conv2_outputs))  # (?, 1, 1, 128)
    # hidden_4
    hidden_4 = ap(conv3(conv2_outputs))  # (?, 1, 1, 128)
    # concatenate
    concat = concatenate([hidden_1, hidden_2, hidden_3, hidden_4], axis=-1)  # (?, 1, 1, 768)
    re = Reshape([768])
    concat = re(concat)
    # dense classifier
    d1 = Dense(256, activation='relu')
    d1_outputs = d1(concat)  # (?, 256)
    d2 = Dense(64, activation='relu')
    d2_outputs = d2(d1_outputs)  # (?, 64)
    d3 = Dense(num_genres, activation='softmax')
    drop2 = Dropout(rate=0.1)
    outputs = d3(drop2(d2_outputs))  # (?, 10)
    re2 = Reshape([10])
    outputs = re2(outputs)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    return model


def model_nnet2(num_genres=hp.num_genres, input_shape=hp.input_shape):
    inputs = Input(shape=input_shape, name='model_inputs')  # (?, 256, 128, 1)
    # conv1
    conv1 = Conv2D(256, kernel_size=(128, 4), activation='relu',
                   input_shape=input_shape, kernel_regularizer=regularizers.l2(0.02))
    conv1_temp = conv1(inputs)
    drop1 = Dropout(rate=0.2)
    conv1_outputs = drop1(conv1_temp)  # (?, 1, 125, 256)
    # hidden_2
    conv2 = Conv2D(256, kernel_size=(1, 4), activation='relu',
                   kernel_regularizer=regularizers.l2(0.01),
                   padding='same')
    conv2_outputs = conv2(conv1_outputs)  # (?, 1, 125, 256)
    # hidden_3
    conv3 = Conv2D(256, kernel_size=(1, 4), activation='relu',
                   kernel_regularizer=regularizers.l2(0.01),
                   padding='same')
    conv3_outputs = conv3(conv2_outputs)  # (?, 1, 125, 256)
    # Add
    add = Add()
    add_outputs = add([conv1_outputs, conv2_outputs, conv3_outputs]) # (?, 1, 125, 256)
    # equivalent to
    # added = keras.layers.add([conv1_outputs, conv2_outputs, conv3_outputs])
    # MaxPooling
    mp = MaxPooling2D(pool_size=(1, 125))
    mp_outputs = mp(add_outputs)  # (?, 1, 1, 256)
    # AveragePooling
    ap = AveragePooling2D(pool_size=(1, 125))
    ap_outputs = ap(add_outputs)  # (?, 1, 1, 256)
    # concat
    concat = concatenate([mp_outputs, ap_outputs])  # (?, 1, 1, 512)
    re = Reshape([512])
    concat_outputs = re(concat)  # (?, 512)
    # dense classifier
    d1 = Dense(128, activation='relu')
    d1_outputs = d1(concat_outputs)
    d2 = Dense(64, activation='relu')
    d2_outputs = d2(d1_outputs)
    d3 = Dense(num_genres, activation='softmax')
    drop2 = Dropout(rate=0.1)
    outputs = d3(drop2(d2_outputs))
    re2 = Reshape([10])
    outputs = re2(outputs)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    return model


def model_nnet1_128(num_genres=hp.num_genres, input_shape=hp.input_shape):
    inputs = Input(shape=input_shape, name='model_inputs')  # (?, 128, 128, 1)
    # conv1
    conv1 = Conv2D(256, kernel_size=(128, 4), activation='relu',
                   input_shape=input_shape, kernel_regularizer=regularizers.l2(0.02))
    conv1_temp = conv1(inputs)
    drop1 = Dropout(rate=0.2)
    conv1_outputs = drop1(conv1_temp)  # (?, 1, 125, 256)
    # hidden_1
    mp = MaxPooling2D(pool_size=(1, 125))
    hidden_1 = mp(conv1_outputs)  # (?, 1, 1, 256)
    # conv2
    conv2 = Conv2D(256, kernel_size=(1, 4), activation='relu',
                   kernel_regularizer=regularizers.l2(0.01),
                   padding='same')
    conv2_outputs = conv2(conv1_outputs)  # (?, 1, 125, 256)
    # hidden_2
    ap = AveragePooling2D(pool_size=(1, 125))
    hidden_2 = ap(conv2_outputs)  # (?, 1, 1, 256)
    # hidden_3
    conv3 = Conv2D(128, kernel_size=(1, 4), activation='relu',
                   kernel_regularizer=regularizers.l2(0.01),
                   padding='same')
    hidden_3 = mp(conv3(conv2_outputs))  # (?, 1, 1, 128)
    # hidden_4
    hidden_4 = ap(conv3(conv2_outputs))  # (?, 1, 1, 128)
    # concatenate
    concat = concatenate([hidden_1, hidden_2, hidden_3, hidden_4], axis=-1)  # (?, 1, 1, 768)
    re = Reshape([768])
    concat = re(concat)
    # dense classifier
    d1 = Dense(256, activation='relu')
    d1_outputs = d1(concat)  # (?, 256)
    d2 = Dense(64, activation='relu')
    d2_outputs = d2(d1_outputs)  # (?, 64)
    d3 = Dense(num_genres, activation='softmax')
    drop2 = Dropout(rate=0.1)
    outputs = d3(drop2(d2_outputs))  # (?, 10)
    re2 = Reshape([10])
    outputs = re2(outputs)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    return model


def model_nnet3_128(num_genres=hp.num_genres, input_shape=hp.input_shape):
    inputs = Input(shape=input_shape, name='model_inputs')  # (?, 128, 128, 1)
    # conv1
    conv1 = Conv2D(256, kernel_size=(128, 4), activation='relu',
                   input_shape=input_shape, kernel_regularizer=regularizers.l2(0.02))
    conv1_temp = conv1(inputs)
    drop1 = Dropout(rate=0.2)
    conv1_outputs = drop1(conv1_temp)  # (?, 1, 125, 256)
    # hidden_1
    mp = MaxPooling2D(pool_size=(1, 125))
    hidden_1 = mp(conv1_outputs)  # (?, 1, 1, 256)
    # conv2
    conv2 = Conv2D(256, kernel_size=(1, 4), activation='relu',
                   kernel_regularizer=regularizers.l2(0.01),
                   padding='same')
    conv2_outputs = conv2(conv1_outputs)  # (?, 1, 125, 256)
    # hidden_2
    ap = AveragePooling2D(pool_size=(1, 125))
    hidden_2 = ap(conv1_outputs)  # (?, 1, 1, 256)
    # hidden_3
    conv3 = Conv2D(128, kernel_size=(1, 4), activation='relu',
                   kernel_regularizer=regularizers.l2(0.01),
                   padding='same')
    hidden_3 = mp(conv3(conv2_outputs))  # (?, 1, 1, 128)
    # hidden_4
    hidden_4 = ap(conv3(conv2_outputs))  # (?, 1, 1, 128)
    # concatenate
    concat = concatenate([hidden_1, hidden_2, hidden_3, hidden_4], axis=-1)  # (?, 1, 1, 768)
    re = Reshape([768])
    concat = re(concat)
    # dense classifier
    d1 = Dense(256, activation='relu')
    d1_outputs = d1(concat)  # (?, 256)
    d2 = Dense(64, activation='relu')
    d2_outputs = d2(d1_outputs)  # (?, 64)
    d3 = Dense(num_genres, activation='softmax')
    drop2 = Dropout(rate=0.1)
    outputs = d3(drop2(d2_outputs))  # (?, 10)
    re2 = Reshape([10])
    outputs = re2(outputs)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])
    return model



def model_lstm(num_genres=hp.num_genres, input_shape=hp.input_shape):
    model = Sequential()
    model.add(Reshape((128, 128), input_shape=input_shape))
    model.add(LSTM(units=128, return_sequences=True))
    model.add(Reshape((128, 128, 1)))
    model.summary()
    model.add(MaxPooling2D(pool_size=(2, 4)))
    model.add(Conv2D(64, (3, 5), activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(MaxPooling2D(pool_size=(2, 4)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.02)))
    model.add(Dropout(0.2))
    model.add(Dense(num_genres, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adadelta(),
                      metrics=['accuracy'])
    return model

def net1_correct():
    inputs = Input(shape=(256, 128, 1))
    conv1 = Conv2D(256, (249, 4), padding='valid', activation='relu', kernel_regularizer=regularizers.l2(0.02))(inputs)
    conv1_drop = Dropout(0.2)(conv1)
    conv2 = Conv2D(256, (1, 4), padding='same', kernel_regularizer=regularizers.l2(0.01))(conv1_drop)
    hidden1 = MaxPooling2D((1, 125))(conv1_drop)
    hidden2 = AveragePooling2D((1, 125))(conv2)
    conv3 = Conv2D(128, (1, 4), padding='same', kernel_regularizer=regularizers.l2(0.01))(conv2)
    hidden3 = MaxPooling2D((1, 125))(conv3)
    conv4 = Conv2D(128, (1, 4), padding='same', kernel_regularizer=regularizers.l2(0.01))(conv2)
    hidden4 = AveragePooling2D((1, 125))(conv4)
    concat = concatenate([hidden1, hidden2, hidden3, hidden4], axis=-1)
    flatten = Flatten()(concat)
    dense1 = Dense(256, activation='relu')(flatten)
    dense2 = Dense(64, activation='relu')(dense1)
    dense2_drop = Dropout(0.1)(dense2)
    outputs = Dense(10, activation='softmax')(dense2_drop)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=keras.optimizers.Adadelta(),
                  loss=keras.losses.categorical_crossentropy,
                  metrics=['accuracy'])
    return model


class Music_Genre_CNN(object):
    def __init__(self, ann_model):
        self.model = ann_model()

    def load_model(self, model_path, custom_objects=None):
        self.model = load_model(model_path, custom_objects=custom_objects)

    def summary(self):
        print(self.model.summary())

    def train_model(self, input_spectrograms, labels, cv=False,
                    validation_spectrograms=None, validation_labels=None,
                    small_batch_size=hp.small_batch_size, max_iteration=hp.max_iteration,
                    print_interval=hp.print_interval):
        """
        train the CNN model
        :param print_interval:
        :param input_spectrograms: number of training examplex * num of mel bands * number of fft windows * 1
            type: 4D numpy array
        :param labels: vectorized class labels
            type:
        :param cv: whether do cross validation
        :param validation_spectrograms: data used for cross validation
            type: as input_spectrogram
        :param validation_labels: used for cross validation
        :param small_batch_size: size of each training batch
        :param max_iteration:
            maximum number of iterations allowed for one training
        :return:
            trained model
        """
        validation_accuracy_list = []
        for iii in range(max_iteration):

            st_time = time.time()

            # split training data into even batches
            num_training_data = len(input_spectrograms)
            batch_idx = np.random.permutation(num_training_data)
            num_batches = int(num_training_data / small_batch_size)

            for jjj in range(num_batches - 1):

                sample_idx = np.random.randint(input_spectrograms.shape[2] - hp.num_fft_windows)
                training_idx = batch_idx[jjj * small_batch_size: (jjj + 1) * small_batch_size]
                training_data = input_spectrograms[training_idx, :, sample_idx:sample_idx + hp.num_fft_windows, :]
                # (150, 128, 128, 1)
                training_label = labels[training_idx]  # (150, 10)
                # training_label = K.reshape(training_label, (150, 1, 1, 10))
                # checkpoint = ModelCheckpoint(hp.best_model, monitor='val_acc', verbose=0, save_best_only=True,
                #                              mode='max', period=1)
                # checkpoint_list = [checkpoint]
                self.model.train_on_batch(training_data, training_label)
                # self.model.evaluate(training_data, training_label, verbose=0)
                # print("Training accuracy is: %f" % (training_accuracy))

            # end_time = time.time()
            # elapsed_time = end_time - st_time

            if iii % print_interval == 0:
                sample_idx = np.random.randint(input_spectrograms.shape[2] - hp.num_fft_windows)
                # training_idx = batch_idx[jjj * small_batch_size: (jjj + 1) * small_batch_size]  
                training_data = input_spectrograms[:, :, sample_idx:sample_idx + hp.num_fft_windows, :]
                training_label = labels[:]
                training_loss, _ = self.model.evaluate(training_data, training_label, verbose=0)
                training_accuracy, _ = self.test_model(input_spectrograms[:, :, :, :], training_label)
    
                validation_accuracy, _ = self.test_model(validation_spectrograms, validation_labels)
                print("\nIteration:%d  Loss: %f; Training accuracy: %f, Validation accuracy: %f\n" 
                      %(iii, training_loss, training_accuracy, validation_accuracy)) 

            """
            if cv:
                validation_accuracy = self.model.evaluate(
                    validation_spectrograms[:, :, sample_idx:sample_idx + hp.num_fft_windows, :]
                    , validation_labels, verbose=0)
                validation_accuracy_list.append(validation_accuracy[1])
            else:
                validation_accuracy = [-1.0, -1.0]

            if iii % print_interval == 0:
                with open(hp.loss_log, "a") as text_file:
                    things2write = "iter: " + str(iii) + "\t" + "loss: " + str(training_accuracy[0]) + "\t" + str(training_accuracy[1])  + "\t" + str(validation_accuracy[1]) + "\n"
                    text_file.write(things2write)
                print("\nIteration:%d  Loss: %f; Training accuracy: %f, Validation accuracy: %f\n" %
                      (iii, training_accuracy[0], training_accuracy[1], validation_accuracy[1]))
        if cv:
            return np.asarray(validation_accuracy_list)
            """

    def song_spectrogram_prediction(self, song_mel_spectrogram, overlap):
        """
        give the predicted_probability for each class and each segment
        :param song_mel_spectrogram:
            4D numpy array: num of time windows * mel bands * 1 (depth)
        :param overlap:
            overlap between segments, overlap = 0 means no overlap between segments
        :return:
            predictions: numpy array (number of segments * num classes)
        """
        largest_idx = song_mel_spectrogram.shape[1] - hp.num_fft_windows - 1
        step_size = int((1 - overlap) * hp.num_fft_windows)
        num_segments = int(largest_idx / step_size)
        segment_edges = np.arange(num_segments) * step_size
        segment_list = []
        for idx in segment_edges:
            segment = song_mel_spectrogram[:, idx: idx + hp.num_fft_windows]
            segment_list.append(segment)
        segment_array = np.asarray(segment_list)[:, :, :, np.newaxis]
        # predictions = self.model.predict_proba(segment_array, batch_size=len(segment_array), verbose=0)
        predictions = self.model.predict(segment_array, batch_size=len(segment_array), verbose=0)
        summarized_prediction = np.argmax(predictions.sum(axis=0))
        return summarized_prediction, predictions

    def test_model(self, test_X, test_T, overlap=0.5):
        # test the accuracy of the model using testing data
        # test_T (100, 10), one_hot vector
        num_sample = len(test_T)  # 1000 * 1/10 = 100
        correct_labels = np.argmax(test_T, axis=1)
        predicted_labels = np.zeros(num_sample)
        # test_X (100, 128, 1292)
        for iii in range(len(test_X)):
            song_mel_spectrogram = test_X[iii].squeeze()
            predicted_labels[iii], _ = self.song_spectrogram_prediction(song_mel_spectrogram, overlap=overlap)
        confusion_data = np.vstack((predicted_labels, correct_labels)).T
        accuracy = np.sum(correct_labels == predicted_labels) * 1.0 / num_sample
        return accuracy, confusion_data

    def backup_model(self, model_bk_name=False):
        if not model_bk_name:
            year, month, day, hour, minute = time.strftime("%Y,%m,%d,%H,%M").split(',')
            model_type = hp.model_name
            model_bk_name = "saved_model/mgcnn_" + model_type + "_" + month + day + hour + minute + ".h5"
        self.model.save(model_bk_name)

    def song_genre_prediction(self, audio_path):
        # resample the song into single channel, 22050 sampling frequency

        # convert into mel-scale spectrogram

        # predict using trained model

        x, sr = librosa.core.load(audio_path)

        xm = mel_spectrogram(ys=x, sr=sr)

        summarized_predictions, predictions = self.song_spectrogram_prediction(xm, 0.5)

        print("predict result: " + str(np.argmax(predictions, axis=1)))

        genres_dict = {'blues': 0, 'classical': 1, 'country': 2, 'disco': 3, 'hiphop': 4,
                       'jazz': 5, 'metal': 6, 'pop': 7, 'reggae': 8, 'rock': 9}

        print("summarized_predictions: " +
              str(list(genres_dict.keys())[list(genres_dict.values()).index(summarized_predictions)]))



