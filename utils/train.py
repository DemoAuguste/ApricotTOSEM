from .utils import *
from keras.preprocessing.image import ImageDataGenerator
from datetime import datetime


def train_model(model_name, num_classes=10, dataset='cifar10', ver=1, num_submodels=20, train_sub=True,
                subset_size=10000, **kwargs):
    img_rows = -1
    img_cols = -1
    img_channels = -1
    if 'cifar' in dataset or dataset == 'svhn':
        img_rows, img_cols = 32, 32
        img_channels = 3
    elif dataset == 'fashion-mnist' or dataset == 'mnist':
        img_rows, img_cols = 28, 28
        img_channels = 1
    else:  # TODO
        pass

    model_weights_save_dir = os.path.join(WORKING_DIR, 'weights')
    model_weights_save_dir = os.path.join(model_weights_save_dir, model_name)
    model_weights_save_dir = os.path.join(model_weights_save_dir, dataset)
    model_weights_save_dir = os.path.join(model_weights_save_dir, str(ver))
    log_path = os.path.join(model_weights_save_dir, 'train.log')

    logger("########################", log_path)
    logger("model name: {}".format(model_name), log_path)
    logger("dataset: {}".format(dataset), log_path)
    logger("version: {}".format(ver), log_path)
    logger("num of classes: {}".format(num_classes), log_path)
    logger("num of submodels: {}".format(num_submodels), log_path)
    logger("########################", log_path)

    model = build_networks(model_name, num_classes=num_classes, input_size=(img_rows, img_cols, img_channels))
    model.summary()

    x_train, x_test, y_train, y_test = load_dataset(dataset)
    print("x_train shape: {}".format(x_train.shape))
    print("y_train shape: {}".format(y_train.shape))
    x_train_val, x_val, y_train_val, y_val = split_validation_dataset(x_train, y_train)

    if not os.path.exists(model_weights_save_dir):
        os.makedirs(model_weights_save_dir)

    pretrained_path = os.path.join(model_weights_save_dir, 'pretrained.h5')
    trained_path = os.path.join(model_weights_save_dir, 'trained.h5')
    log_path = os.path.join(model_weights_save_dir, 'log-train.txt')

    datagen = ImageDataGenerator(horizontal_flip=True,
                                 width_shift_range=0.125,
                                 height_shift_range=0.125,
                                 fill_mode='constant', cval=0.)
    datagen.fit(x_train_val)

    start = datetime.now()
    if not os.path.exists(pretrained_path):
        # pretrain the model, using the x_train_val.
        model.fit_generator(datagen.flow(x_train_val, y_train_val, batch_size=BATCH_SIZE),
                            steps_per_epoch=len(x_train_val) // BATCH_SIZE + 1,
                            validation_data=(x_val, y_val),
                            epochs=kwargs['pre_epochs'])
        model.save_weights(pretrained_path)
    else:
        model.load_weights(pretrained_path)

    if not os.path.exists(trained_path):
        # train the original model
        model.fit_generator(datagen.flow(x_train_val, y_train_val, batch_size=BATCH_SIZE),
                            steps_per_epoch=len(x_train_val) // BATCH_SIZE + 1,
                            validation_data=(x_val, y_val),
                            epochs=kwargs['after_epochs'])
        model.save_weights(trained_path)
        end = datetime.now()
        logger('time for training the original DL model: {}'.format(end - start), log_path)

    # print('time for training the original DL model: {}'.format(end-start))

    if train_sub:
        submodel_dir = os.path.join(model_weights_save_dir, 'submodels')
        if not os.path.exists(submodel_dir):
            os.makedirs(submodel_dir)

        step = int((x_train_val.shape[0] - subset_size) / num_submodels)
        for i in range(num_submodels):
            submodel_save_path = os.path.join(submodel_dir, 'sub_{}.h5'.format(i))
            if os.path.exists(submodel_save_path):
                continue

            sub_x_train_val = x_train_val[step * i: subset_size + step * i]
            sub_y_train_val = y_train_val[step * i: subset_size + step * i]

            # load the pretrained model
            start = datetime.now()
            model.load_weights(pretrained_path)
            model.fit_generator(datagen.flow(sub_x_train_val, sub_y_train_val, batch_size=BATCH_SIZE),
                                steps_per_epoch=len(sub_x_train_val) // BATCH_SIZE + 1,
                                validation_data=(x_val, y_val),
                                epochs=kwargs['sub_epochs'])
            model.save_weights(submodel_save_path)
            end = datetime.now()
            logger('time for training sub-{}: {}'.format(i, end - start), log_path)
