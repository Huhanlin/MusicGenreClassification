# import modules
import numpy as np
import librosa
import music_gen_lib as mgl
from keras.utils import np_utils
import time
from keras import backend as K
from keras.engine.topology import Layer
import tensorflow as tf
import keras
from sklearn.metrics import classification_report, confusion_matrix
from hyperparams import HyperParams as hp
from keras.backend.tensorflow_backend import set_session


def main(random_seed=hp.random_seed, visualize_label=hp.random_seed, model=hp.model_name):
    # determine the random seed so that results are reproducible
    # random_seed = 11 # also determines which shuffled index to use

    np.random.seed(random_seed)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))

    # 1st part: load all the data and organize data
    if model == "nnet1":
        MGCNN = mgl.Music_Genre_CNN(mgl.model_nnet1_128)
    elif model == "nnet2":
        MGCNN = mgl.Music_Genre_CNN(mgl.model_nnet2)
    elif model == "nnet1_128":
        MGCNN = mgl.Music_Genre_CNN(mgl.model_nnet1_128)
    elif model == "nnet3":
        MGCNN = mgl.Music_Genre_CNN(mgl.model_nnet3_128)
    elif model == "lstm":
        MGCNN = mgl.Music_Genre_CNN(mgl.model_lstm)
    elif model == "net1_correct":
        MGCNN = mgl.Music_Genre_CNN(mgl.net1_correct)
    else:
        MGCNN = mgl.Music_Genre_CNN(mgl.baseline_model_128)

    if hp.summary:
        MGCNN.summary()

    if not hp.training_flag:
        # try to load pre-trained model
        # saved_model_name = "mgcnn_rs_" + str(random_seed) + ".h5"
        # saved_model_name = "mgcnn_poisson_rs_" + str(random_seed) + ".h5"
        # MGCNN.load_model(saved_model_name, custom_objects={'PoissonLayer': ann_model})
        MGCNN.load_model("./saved_model/" + hp.saved_model_name)

        if hp.test_flag:
            MGCNN.song_genre_prediction(hp.single_music_path)
            exit(0)

    if hp.data_convert:
        # load all the data
        X, SR, T = mgl.load_original_data()
        # data format:
        #       x: 1d numpy array
        #       t: 1d numpy array with numsic genre names (numeric arrays or multinomial vector?)

        # convert the data into mel-scale spectrogram
        st = time.time()
        newX = mgl.batch_mel_spectrogram(X, SR)
        # save the data into npz
        np.savez_compressed(hp.data, X=newX, SR=SR, T=T)
        print('finish data convert')
        exit(3)

    elif not hp.test_flag:
        st = time.time()
        data = np.load(hp.data)
        X = data["X"]
        SR = data["SR"]
        T = data["T"]
        loading_time = time.time() - st
        print("Loading takes %f seconds." % loading_time)
        print(np.shape(X))
        print(np.shape(T))

    # Use log transformation to preserve the order but shrink the range
    # X = np.log(X + 1)
    X = X[:, :, :, np.newaxis]  # image channel should be the last dimension, check by using print K.image_data_format()

    # convert string type labels to vectors
    genres = np.unique(T)
    genres_dict = dict([[label, value] for value, label in enumerate(genres)])
    T_numeric = np.asarray([genres_dict[label] for label in T])
    T_vectorized = np_utils.to_categorical(T_numeric)

    # split data into training, cross-validation,  testing data
    # following is used to generate random see used to split the data into different sets

    if hp.split_data:
        # split_idxes = np.asarray([0, 0.5, 0.7, 1])
        split_idxes = np.asarray([0, hp.training_set,
                                  hp.training_set + hp.validation_set,
                                  hp.training_set + hp.validation_set + hp.test_set])
        training_idxes_list, validation_idxes_list, testing_idxes_list = [], [], []
        for idx in range(30):
            training_idxes, validation_idxes, testing_idxes = mgl.split_data(T, split_idxes)
            training_idxes_list.append(training_idxes)
            validation_idxes_list.append(validation_idxes)
            testing_idxes_list.append(testing_idxes)

        training_idxes_list = np.asarray(training_idxes_list)
        validation_idxes_list = np.asarray(validation_idxes_list)
        testing_idxes_list = np.asarray(testing_idxes_list)

        np.savez_compressed(hp.shuffled_idx_list, training_idxes_list=training_idxes_list,
                            validation_idxes_list=validation_idxes_list, testing_idxes_list=testing_idxes_list)

    # load one fixed data shuffling indexes
    idxes_list = np.load(hp.shuffled_idx_list)
    training_idxes = idxes_list["training_idxes_list"][random_seed]
    validation_idxes = idxes_list["validation_idxes_list"][random_seed]
    # testing_idxes = idxes_list["testing_idxes_list"][random_seed]
    testing_idxes = idxes_list["validation_idxes_list"][random_seed]

    # shuffled_idx = np.random.permutation(num_total_data) # shuffle or not
    # shuffled_idx_list = np.asarray([np.random.permutation(num_total_data) for x in xrange(30)])
    # np.savez_compressed("shuffled_idx_list.npz", shuffled_idx_list=shuffled_idx_list)

    training_X = X[training_idxes]  # [1000 * training_proportion, 128, 2584, 1]
    # validation_X = X[validation_idxes]
    validation_X = X[testing_idxes]
    # testing_X = X[testing_idxes]
    testing_X = X[validation_idxes]

    training_T = T_vectorized[training_idxes]  # [1000 * training_proportion, 10]
    #validation_T = T_vectorized[validation_idxes]
    validation_T = T_vectorized[testing_idxes]
    #testing_T = T_vectorized[testing_idxes]
    testing_T = T_vectorized[validation_idxes]

    if hp.training_flag:
        print("The model hasn't been trained before.")
        # training the model
        epoch = hp.max_epoch
        training_flag = hp.training_flag
        while training_flag and epoch >= 0:
            validation_accuracies = MGCNN.train_model(training_X, training_T, cv=True,
                                                      validation_spectrograms=validation_X,
                                                      validation_labels=validation_T,
                                                      small_batch_size=hp.small_batch_size,
                                                      max_iteration=hp.max_iteration)

            # diff = np.mean(validation_accuracies[-10:]) - np.mean(validation_accuracies[:10])
            MGCNN.backup_model()  # backup in case error occurred
            # if np.abs(diff) < 0.01:
            #     training_flag = False
            test_accuracy, confusion_data = MGCNN.test_model(testing_X, testing_T)
            print("\n ******* epoch%d: test accuracy is %f. ******\n" % (hp.max_epoch-epoch, test_accuracy))

            with open(hp.train_log, "a") as text_file:
                year, month, day, hour, minute = time.strftime("%Y,%m,%d,%H,%M").split(',')
                model_type = hp.model_name
                model_bk_name = "saved_model/mgcnn_" + model_type + "_" + month + day + hour + minute + ".h5"
                things2write = model_bk_name + "\t" + "epoch: " + str(hp.max_epoch-epoch) \
                               + "\t accuracy: " + str(test_accuracy) + "\n"
                text_file.write(things2write)

            epoch -= 1

        MGCNN.backup_model()

    test_accuracy, confusion_data = MGCNN.test_model(testing_X, testing_T)
    print("\n ****** The final test accuracy is %f. ******\n" % (test_accuracy))

    with open(hp.model_accuracy_log, "a") as text_file:
        things2write = hp.saved_model_name + "\t" + "accuracy: " + str(test_accuracy) + "\n"
        text_file.write(things2write)

    # analyze the confusion matrix
    cm = confusion_matrix(confusion_data[:, 1], confusion_data[:, 0]) / (len(testing_T) * 1.0 / len(genres))
    with open(hp.confusion_matrix, "a") as text_file:
        things2write = hp.saved_model_name + "\n" + str(cm) + "\n"
        text_file.write(things2write)

    # visualize
    if visualize_label:
        pass
        import matplotlib.pylab as plt
        import seaborn as sns
        # # plt.matshow(cm)
        # # plt.colorbar()
        # # plt.show()
        fig = plt.figure()
        # plt.ylabel('True label')
        # plt.xlabel('Predicted label')
        lable = ('blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock')
        sns_plot = sns.heatmap(cm, cmap='Blues', annot=True)  # Greys
        plt.xticks(np.arange(10), lable, rotation=30)
        plt.yticks(np.arange(10), lable, rotation=30)
        plt.savefig('cm.pdf')
        #plt.show()
    # return cs, cm


main()
