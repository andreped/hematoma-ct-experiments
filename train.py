from math import ceil
from tensorflow.python.keras.callbacks import ModelCheckpoint, Callback
import os
from numpy.random import shuffle
from batch_generator import *
from keras_tools.network import *
from math import ceil
from datetime import date


def import_dataset(tmp, path, name_curr):
    file = h5py.File(path + 'dataset_' + name_curr + '.h5', 'r')
    tmp = np.array(file[tmp])
    tmp = [tmp[i].decode("UTF-8") for i in range(len(tmp))]
    file.close()
    return tmp


def unique_patients(tmp):
    l = []
    for t in tmp:
        l.append(int(t.split("/")[-2]))
    return np.unique(l)


def unique_patients_in_set(tmp):
    ll = []
    for l in tmp:
        ll.append(int(l.split("/")[-2]))
    return np.unique(ll)


def count_samples_from_each_patient(tmp):
    uniques = unique_patients_in_set(tmp)
    counts = []
    for unique in uniques:
        cnt = 0
        for l in tmp:
            tmps = int(l.split("/")[-2])
            if unique == tmps:
                cnt += 1
        counts.append(cnt)
    return counts


def balance_upsample(dirs):
    ll = []
    for t in dirs:
        l = []
        for t2 in os.listdir(t):
            l.append(t2)
        ll.append(l)
    lengths = [len(x) for x in ll]
    longest = max(lengths)
    sets = []
    for i, d in enumerate(dirs):
        length = lengths[i]
        if length == longest:
            news = []
            for l in ll[i]:
                news.append(d + "/" + l)
        else:
            curr = ll[i]
            shuffle(curr)
            new = np.tile(curr, int(ceil(longest / lengths[i])))[:longest]
            news = []
            for n in new:
                news.append(d + "/" + n)
        for n in news:
            sets.append(n)
    return sets


def balance_posneg(dirs):
    posnegs = []
    ll = []
    sets = []
    for t in np.sort(dirs):
        l = []
        posneg = {}
        posneg["0"] = []
        posneg["1"] = []
        for t2 in np.sort(os.listdir(t)):
            tmp = t2.split(".")[0].split("_")[-1]
            posneg[tmp].append(t2)
            l.append(t2)
        ll.append(l)
        posnegs.append(posneg)
        length1 = len(posneg["1"])
        if length1 > 0:
            tmp2 = posneg["0"]
            shuffle(tmp2)
            tmp2 = tmp2[:length1]
            tmp_set = posneg["1"] + tmp2
            for t2 in tmp_set:
                sets.append(t + "/" + t2)
    return sets



if __name__ == "__main__":

    # use single GPU (first one)
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"

    # current date
    curr_date = "_".join(date.today().strftime("%d/%m/%Y").split("/")[:2]) + "_"

    # aug = {}, bs = 8
    nb_classes = 2

    #out_dim = (16, 512, 512)
    out_dim = (64, 256, 256)
    # limits = (0, 140)
    limits = (-230, 230)
    stride = 4

    # fine-tune flag
    fn_flag = False

    # model name
    name = curr_date + "data_" + str(out_dim[0]) + "_" + str(out_dim[1]) + "_stride_" + str(stride) + "_balanced_posneg" + "_out_dim_"\
			   + str(out_dim).replace(" ", "") + "_limits_" + str(limits).replace(" ", "")
    print(name)
    # name = '16_08_unet_16_512_test_balanced_pos_only_stride_4'

    slices = int(name.split('_')[3])
    window = int(name.split('_')[4])

    # paths
    # data_path = "/home/andrep/hematoma/datasets/data_16_512_pos_only_stride_4/"
    #data_path = "/home/andrep/workspace/hematoma/datasets/data_16_512_all_stride_4/"
    data_path = "/home/andrep/workspace/hematoma/datasets/data_16_512_all_stride_4_" + "out_dim_"\
			   + str(out_dim).replace(" ", "") + "_limits_" + str(limits).replace(" ", "") + "/"
    data_path = "/home/andrep/workspace/hematoma/datasets/data_16_512_all_stride_4_out_dim_(64,256,256)_limits_(0,140)/"
    save_model_path = '/home/andrep/workspace/hematoma/output/models/'
    history_path = '/home/andrep/workspace/hematoma/output/history/'
    datasets_path = '/home/andrep/workspace/hematoma/output/datasets/'

    if not fn_flag:
        # assign WSIs randomly to train, val and test
        images = os.listdir(data_path)
        images = [data_path + i for i in images]
        shuffle(images)

        # split
        val = 6
        test_dir = images[:val]
        val_dir = images[val:int(2 * val)]  # first 5 in val/test
        train_dir = images[int(2 * val):]  # rest in training

        # get the actual patches
        sets = []
        for t in test_dir:
            for p in os.listdir(t):
                sets.append(t + '/' + p)
        test_set = sets.copy()
        val_set = test_set.copy()

        sets = []
        for t in train_dir:
            for p in os.listdir(t):
                sets.append(t + '/' + p)
        train_set = sets.copy()

        print("Before balancing: ")
        print(count_samples_from_each_patient(train_set))
        print(count_samples_from_each_patient(val_set))
        print(count_samples_from_each_patient(test_set))

        configs = 2

        if configs == 1:
            print('Balancing number of samples from each scan...')
            # balance samples from each scan
            train_set = balance_upsample(train_dir)
            val_set = balance_upsample(val_dir)
            test_set = balance_upsample(test_dir)

            print("After balancing: ")
            print(count_samples_from_each_patient(train_set))
            print(count_samples_from_each_patient(val_set))
            print(count_samples_from_each_patient(test_set))

        elif configs == 2:
            print('balancing number of positive and negative samples')
            # balance positve/negative samples
            train_set = balance_posneg(train_dir)
            val_set = balance_posneg(val_dir)
            test_set = balance_posneg(test_dir)

        else:
            print('No balancing done...')
            exit()

        print(len(val_set))
        print("Unique patients in each set: ")
        print(unique_patients_in_set(train_set))
        print(unique_patients_in_set(val_set))
        print(unique_patients_in_set(test_set))

        # make larger val_set:
        # val_set = test_set + val_set # <- val_set = test_set -> cross validation

    else:
        path_curr = name
        test_set = import_dataset('test', datasets_path, path_curr)
        val_set = import_dataset('val', datasets_path, path_curr)
        train_set = import_dataset('train', datasets_path, path_curr)

    # save random generated data sets
    f = h5py.File((datasets_path + 'dataset_' + name + '.h5'), 'w')
    f.create_dataset("test", data=np.array(test_set).astype('S200'), compression="gzip", compression_opts=4)
    f.create_dataset("val", data=np.array(val_set).astype('S200'), compression="gzip", compression_opts=4)
    f.create_dataset("train", data=np.array(train_set).astype('S200'), compression="gzip", compression_opts=4)
    f.close()

    # define model
    network = Unet(input_shape=(slices, window, window, 1), nb_classes=2)
    network.encoder_spatial_dropout = 0.1  # 0.2
    network.decoder_spatial_dropout = 0.1
    #network.set_convolutions([4, 8, 16, 32, 64, 128, 256, 512, 256, 128, 64, 32, 16, 8, 4]) # <- for 16x512x512
    network.set_convolutions([8, 16, 32, 64, 128, 128, 256, 128, 128, 64, 32, 16, 8]) # <- 64x256x256
    # network.set_convolutions([4, 8, 16, 32, 64, 64, 128, 256, 128, 64, 64, 32, 16, 8, 4])
    model = network.create()
    print(model.summary())

    # load model <- if want to fine-tune, or train further on some previously trained model
    if fn_flag:
        model.load_weights(save_model_path + 'model_' + name + '.h5', by_name=True)

    model.compile(
        # optimizer = Adam(1e-3), # 1e-2 best?
        optimizer='adadelta',
        # loss='binary_crossentropy'
        loss=network.get_dice_loss()
    )

    # augmentation
    batch_size = 4  # 8
    epochs = 1000

    aug = {'flip': 1, 'rotate': 20, 'shift': int(np.round(window * 0.1))}  # , 'zoom':[0.75, 1.25]}
    val_aug = {}

    # define generators for sampling of data
    train_gen = batch_gen3(train_set, batch_size=batch_size, aug=aug, epochs=epochs, nb_classes=nb_classes,
                           input_shape=(slices, window, window, 1))
    val_gen = batch_gen3(val_set, batch_size=batch_size, aug=val_aug, epochs=epochs, nb_classes=nb_classes,
                         input_shape=(slices, window, window, 1))

    train_length = len(train_set)
    val_length = len(val_set)

    save_best = ModelCheckpoint(
        save_model_path + 'model_' + name + '.h5',
        monitor='val_loss',
        verbose=0,
        save_best_only=True,
        save_weights_only=False,
        mode='auto',  # 'auto',
        period=1
    )



    class LossHistory(Callback):
        def on_train_begin(self, logs={}):
            self.losses = []
            self.losses.append(['loss', 'val_loss',
                                'acc', 'val_acc'])

        def on_epoch_end(self, batch, logs={}):
            self.losses.append([logs.get('loss'), logs.get('val_loss'),
                                logs.get('acc'), logs.get('val_acc')])
            # save history:
            f = h5py.File((history_path + 'history_' + name + '.h5'), 'w')
            f.create_dataset("history", data=np.array(self.losses).astype('|S9'), compression="gzip",
                             compression_opts=4)
            f.close()


    history_log = LossHistory()

    history = model.fit_generator(
        train_gen,
        steps_per_epoch=int(ceil(train_length / batch_size)),
        epochs=epochs,
        validation_data=val_gen,
        validation_steps=int(ceil(val_length / batch_size)),
        callbacks=[save_best, history_log],
        use_multiprocessing=False,
        workers=1
    )
