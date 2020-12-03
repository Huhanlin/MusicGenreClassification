class HyperParams:
    # -------------Handle_data---------------- #
    data_convert = False
    data = "audio_sr_label.npz"
    shuffled_idx_list = "shuffled_idx_list.npz"
    random_seed = 20
    split_data = True
    training_set = 0.8
    validation_set = 0.1
    test_set = 0.1
    data_folder = "genres"

    # -------------Train_params--------------- #
    model_name = "nnet3"
    saved_model_name = "mgcnn_nnet3_02121133.h5"
    # nnet3_128_20_best.h5 86
    # mgcnn_baseline_128_20_02111236.h5 74
    # mgcnn_nnet3_02121133.h5 84
    best_model = "saved_model/best_model.h5"
    training_flag = False
    model_accuracy_log = "model_accuracy_log.txt"
    train_log = "train_log.txt"
    confusion_matrix = "confusion_matrix_log.txt"
    loss_log = "loss_log.txt"

    small_batch_size = 50
    max_epoch = 10
    max_iteration = 500
    print_interval = 50

    sr = 22050  # if sampling rate is different, resample it to this
    # parameters for calculating spectrogram in mel scale
    fmax = 10000  # maximum frequency considered
    fft_window_points = 512
    fft_window_dur = fft_window_points * 1.0 / sr  # 23ms windows
    hop_size = int(fft_window_points / 2)  # 50% overlap between consecutive frames
    n_mels = 128
    pre_emphasis = 0

    # segment duration
    num_fft_windows = 128  # num fft windows per music segment
    segment_in_points = num_fft_windows * 512  # number of data points that insure the spectrogram has size: 128 * 128
    segment_dur = segment_in_points * 1.0 / sr

    n_fft = 512

    num_genres = 10
    # input_shape = (256, 128, 1)
    input_shape = (128, 128, 1)

    # --------------Test_params------------------ #
    summary = True
    test_flag = True
    single_music_path = "genres/pop/pop.00089.wav"
    visualize_label = True


