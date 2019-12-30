'''
This program is written to implement Vehicle ReID based on https://arxiv.org/abs/1708.02386
This part is to fine-tune the backbone-attribute branch on multi-task classification (predict model and colour)
'''
import os
import numpy as np
from math import ceil
from keras.applications.inception_v3 import InceptionV3
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.layers import Dense
from keras.models import Model, load_model
from keras.optimizers import SGD
from keras.utils import multi_gpu_model
from sklearn.utils import class_weight
from utils import generator_batch_multitask
#CUDA_VISIBLE_DEVICES=""
GPUS = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = GPUS


NUM_MODELS = 250
NUM_COLORS = 7
LEARNING_RATE = 0.001
IMG_WIDTH = 299
IMG_HEIGHT = 299
BATCH_SIZE = 32
RANDOM_SCALE = True
NUM_EPOCHS=5
INITIAL_EPOCH = 0
nbr_gpus = len(GPUS.split(','))


def get_attribute_branch(img_w,img_h,use_imagenet=True):
    inception=InceptionV3(include_top=False, weights='imagenet', input_tensor=None, input_shape=(img_w,img_h,3), pooling='avg' )
    f_base=inception.get_layer(index=-1).output

    f_acs=Dense(1024,name='f_acs')(f_base)
    f_model=Dense(NUM_MODELS,activation='softmax',name='predict_model')(f_acs)
    f_colour=Dense(NUM_COLORS,activation='softmax',name='predict_colour')(f_acs)

    model=Model(inputs=inception.input, outputs=[f_model,f_colour])
    model.summary()
    return model

def train_attribute_branch(BATCH_SIZE):

    #model
    model=get_attribute_branch(IMG_WIDTH,IMG_HEIGHT)

    if nbr_gpus > 1:
        print('Using multiple GPUS: {}\n'.format(GPUS))
        model = multi_gpu_model(model, gpus = nbr_gpus)
        BATCH_SIZE *= nbr_gpus
    else:
        print('Using a single GPU.\n')
    optimizer = SGD(lr=LEARNING_RATE, momentum=0.9, decay=0.0, nesterov=True)
    model.compile(loss=['categorical_crossentropy','categorical_crossentropy'], optimizer=optimizer, metrics=['accuracy'])
    model_file_saved='./weights/vehicleModelColor.h5'
    checkpoint = ModelCheckpoint(model_file_saved, monitor='val_loss', verbose=1, save_best_only=True,
                                 save_weights_only=False, mode='auto', period=1)
    reduce_lr=ReduceLROnPlateau(monitor='val_'+'loss', factor=0.5, patience=3, verbose=1, min_lr=0.00001)
    early_stop=EarlyStopping(monitor='val_'+'loss', patience=20, verbose=1)

    #data:
    train_path = './dataPath/path_model_color_train.txt'  # ./train_vehicleModelColor_list.txt
    val_path = './dataPath/path_model_color_test.txt'
    train_data_lines=open(train_path).readlines()
    train_data_lines = [w for w in train_data_lines if os.path.exists(w.strip().split(' ')[0])]
    train_data_lines=train_data_lines
    num_train=len(train_data_lines)
    steps_per_epoch=int(ceil(num_train*1.0/BATCH_SIZE))

    val_data_lines=open(val_path).readlines()
    val_data_lines = [w for w in val_data_lines if os.path.exists(w.strip().split(' ')[0])]
    val_data_lines=val_data_lines
    num_val=len(val_data_lines)
    validation_steps=int(ceil(num_val*1.0/BATCH_SIZE))

    if os.path.exists(model_file_saved):
        model = load_model(model_file_saved)

    model.fit_generator(
                        generator_batch_multitask(train_data_lines,
                        nbr_class_one=NUM_MODELS, nbr_class_two=NUM_COLORS,
                        batch_size=BATCH_SIZE, img_width=IMG_WIDTH,
                        img_height=IMG_HEIGHT, random_scale=RANDOM_SCALE,
                        shuffle=True, augment=False),
                        steps_per_epoch=steps_per_epoch,epochs=NUM_EPOCHS, verbose=1,
                        validation_data=generator_batch_multitask(val_data_lines,
                        nbr_class_one = NUM_MODELS, nbr_class_two = NUM_COLORS, batch_size = BATCH_SIZE,
                        img_width = IMG_WIDTH, img_height = IMG_HEIGHT,
                        shuffle = False, augment = False),
                        validation_steps=validation_steps,
                        callbacks=[checkpoint,reduce_lr],initial_epoch= INITIAL_EPOCH,
                        max_queue_size=80, workers=1, use_multiprocessing=False)


if __name__=='__main__':
    train_attribute_branch(BATCH_SIZE)
