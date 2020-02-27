from model import *
from settings import *
from sklearn.model_selection import train_test_split
from keras.layers import Input
from .utils import *
from keras.preprocessing.image import ImageDataGenerator


def train_model(model_name, num_classes=10, dataset='cifar10', ver=1, num_submodels=20, train_sub=False, save_path=None, top_k=1):
    if 'cifar' in dataset:
        img_rows, img_cols = 32, 32
        img_channels = 3
    else: # TODO
        pass

    model = build_networks(model_name, num_classes=num_classes, input_size=(img_rows, img_cols, img_channels))
    AFTER_EPOCHS = 190 # NOTE: in previous experiment, different models have different AFTER_EPOCHS.

    # region previous codes
    # input_tensor = Input(shape=(img_rows, img_cols, img_channels))
    # if model_name == 'resnet20':
    #     AFTER_EPOCHS = 190
    #     model = build_resnet(img_rows, img_cols, img_channels, num_classes=num_classes, stack_n=3, k=top_k)
    # elif model_name == 'resnet32':
    #     AFTER_EPOCHS = 190
    #     model = build_resnet(img_rows, img_cols, img_channels, num_classes=num_classes, stack_n=5, k=top_k)
    # elif model_name == 'mobilenet':
    #     AFTER_EPOCHS = 190
    #     model = build_mobilenet(input_tensor, num_classes, k=top_k)
    # elif model_name == 'mobilenet_v2':
    #     AFTER_EPOCHS = 190
    #     model = build_mobilenet_v2(input_tensor, num_classes, k=top_k)
    # elif model_name == 'densenet':
    #     if dataset == 'cifar10':
    #         AFTER_EPOCHS = 190
    #     else:
    #         AFTER_EPOCHS = 100
    #     model = build_densenet(input_tensor, num_classes, k=top_k)
    # endregion

    x_train, x_test, y_train, y_test = load_dataset(dataset)
    x_train_val, x_val, y_train_val, y_val = split_validation_dataset(x_train, y_train)

    model_weights_save_dir = os.path.join(WORKING_DIR, 'weights')
    model_weights_save_dir = os.path.join(model_weights_save_dir, model_name)
    model_weights_save_dir = os.path.join(model_weights_save_dir, str(ver))

    if not os.path.exists(model_weights_save_dir):
        os.makedirs(model_weights_save_dir)
    
    pretrained_path = os.path.join(model_weights_save_dir, 'pretrained.h5')
    trained_path = os.path.join(model_weights_save_dir, 'trained.h5')

    if not os.path.exists(pretrained_path):
        # pretrain the model, using the x_train_val.
        datagen = ImageDataGenerator(horizontal_flip=True,
                                width_shift_range=0.125,
                                height_shift_range=0.125,
                                fill_mode='constant', cval=0.)
        datagen.fit(x_train_val)
        model.fit_generator(datagen.flow(x_train_val, y_train_val, batch_size=BATCH_SIZE), 
                                        steps_per_epoch=len(x_train_val) // BATCH_SIZE + 1, 
                                        validation_data=(x_val, y_val), 
                                        epochs=PRE_EPOCHS)
        model.save_weights(pretrained_path)
    else:
        model.load_weights(pretrained_path)
    
    
    if not os.path.exists(trained_path):
        # train the original model
        model.fit_generator(datagen.flow(x_train_val, y_train_val, batch_size=BATCH_SIZE), 
                                            steps_per_epoch=len(x_train_val) // BATCH_SIZE + 1, 
                                            validation_data=(x_val, y_val), 
                                            epochs=AFTER_EPOCHS)
        model.save_weights(trained_path)

    if train_sub:
        submodel_dir = os.path.join(model_weights_save_dir, 'submodels')
        if not os.path.exists(submodel_dir):
            os.makedirs(submodel_dir)
            step = int((x_train_val.shape[0] - subset_size) / num_submodels)
            for i in range(num_submodels):
                submodel_save_path = os.path.join(submodel_dir, 'sub_{}.h5'.format(i))

                sub_x_train_val = x_train_val[step*i : subset_size + step*i]
                sub_y_train_val = y_train_val[step*i : subset_size + step*i]
                
                # load the pretrained model
                model.load_weights(pretrained_path)
                model.fit_generator(datagen.flow(sub_x_train_val, sub_y_train_val, batch_size=BATCH_SIZE), 
                                            steps_per_epoch=len(sub_x_train_val) // BATCH_SIZE + 1, 
                                            validation_data=(x_val, y_val), 
                                            epochs=SUB_EPOCHS)
                model.save_weights(sub_weights_path)
        



