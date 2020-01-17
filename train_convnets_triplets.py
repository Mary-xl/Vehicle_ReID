import os
CUDA_VISIBLE_DEVICES="" # 使用cpu,但是要装cpu 版的tensorflow
#GPUS = "2"
#os.environ["CUDA_VISIBLE_DEVICES"] = GPUS
from math import ceil
import numpy as np
import copy
import keras.backend as K
from keras.applications.inception_v3 import InceptionV3
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.layers import Dense, Input, concatenate, subtract, dot, Activation, add, merge, Lambda
from keras.models import Model
from keras.models import load_model
from keras.optimizers import SGD, RMSprop
from sklearn.utils import class_weight
from utils import generator_batch_triplet, generator_batch
#from keras.utils.training_utils import multi_gpu_model
import keras.backend as K

np.random.seed(1024)

FINE_TUNE = False
SAVE_FILTERED_LIST = True
FINE_TUNE_ON_ATTRIBUTES = True
LEARNING_RATE = 0.00001
NBR_EPOCHS = 100
BATCH_SIZE = 32
IMG_WIDTH = 299
IMG_HEIGHT = 299
monitor_index = 'loss'
NBR_MODELS = 250
NBR_COLORS = 7
RANDOM_SCALE = True
#nbr_gpus = len(GPUS.split(','))
INITIAL_EPOCH = 0
MARGIN = 1.
LOCAL=False

train_path = './dataPath/path_vehicle_model_color_train.txt' # each line:  imgPath vehicleID modelID colorID
val_path = './dataPath/path_vehicle_model_color_val.txt'  # each line:  imgPath vehicleID modelID colorID
if LOCAL==True:
    root_path = '/home/mary/AI'
else:
    root_path=''
# Refer to https://github.com/maciejkula/triplet_recommendations_keras

def identity_loss(y_true, y_pred):
    return K.mean(y_pred - 0 * y_true)

def triplet_loss(vects):
    # f_anchor.shape = (batch_size, 256)
    f_anchor, f_positive, f_negative = vects
    # L2 normalize anchor, positive and negative, otherwise,
    # the loss will result in ''nan''!
    f_anchor = K.l2_normalize(f_anchor, axis = -1)
    f_positive = K.l2_normalize(f_positive, axis = -1)
    f_negative = K.l2_normalize(f_negative, axis = -1)

    dis_anchor_positive = K.sum(K.square(K.abs(f_anchor - f_positive)),
                                          axis = -1, keepdims = True)

    dis_anchor_negative = K.sum(K.square(K.abs(f_anchor - f_negative)),
                                         axis = -1, keepdims = True)
    loss = K.sum(K.maximum(dis_anchor_positive + MARGIN - dis_anchor_negative,0),axis=0)
    return loss

# 过滤掉一些不能进行triplet loss计算的样本
def filter_data_list(data_list):
    # data_list  : a list of [img_path, vehicleID, modelID, colorID]
    # {modelID: {colorID: {vehicleID: [imageName, ...]}}, ...}
    # dic helps us to sample positive and negative samples for each anchor.
    # https://arxiv.org/abs/1708.02386
    # The original paper says that "only the hardest triplets in which the three images have exactly
    # the same coarse-level attributes (e.g. color and model), can be used for similarity learning."
    dic = { }
    # We construct a new data list so that we could sample enough positives and negatives.
    new_data_list = [ ]

    # 遍历txt的每一行。构建嵌套字典。目的是方便根据车型、颜色、车ID查找，主要是用于选择anchor对应的positive、negative
    # [modelID [colorId [vehicleID]]]
    for line in data_list:
        imgPath, vehicleID, modelID, colorID = line.strip().split(' ') # 以空格符为分隔符来取值
        dic.setdefault(modelID, { }) # 如果dic的键中没有modelID,则新建这个键，键值为空字典。如果有，啥都不做。dic的键是数据集的车型ID
        dic[modelID].setdefault(colorID, { }) # 如果某个车型ID字典的键中没有这个颜色ID,就添加这个键，键值为空字典。车型ID的键是数据集的颜色ID
        dic[modelID][colorID].setdefault(vehicleID, [ ]).append(imgPath)
        # 如果某个颜色ID字典的键中没有这个车辆ID,就添加这个键，键值为空字典。车牌ID的键是数据集的车牌ID
        # setdefault,如果字典中包含有给定键，则返回该键对应的值，否则返回为该键设置的值。
        # 然后给字典追加图像地址。一个车也许有多个图像地址。

    # 有些车辆只有一张图片，不能做triplet loss，所以要排除。同一辆车至少要两张才行，因为需要positive
    for line in data_list:
        imgPath, vehicleID, modelID, colorID = line.strip().split(' ')
        #print(imgPath, vehicleID, modelID, colorID)
        if modelID in dic and colorID in dic[modelID] and vehicleID in dic[modelID][colorID] and \
                                                      len(dic[modelID][colorID][vehicleID]) == 1:
            dic[modelID][colorID].pop(vehicleID, None)
            # https://stackoverflow.com/questions/11277432/how-to-remove-a-key-from-a-python-dictionary

    # 同一颜色的车如果只有一俩车，那也不行。因为需要negative(颜色一样的其他的车都没有，那这辆车是没有相近的negative的)
    for line in data_list:
        imgPath, vehicleID, modelID, colorID = line.strip().split(' ')
        if modelID in dic and colorID in dic[modelID] and len(dic[modelID][colorID].keys()) == 1:
            dic[modelID].pop(colorID, None)

    # 这里是筛选后的输入数据
    for modelID in dic:
        for colorID in dic[modelID]:
            for vehicleID in dic[modelID][colorID]:
                for imgPath in dic[modelID][colorID][vehicleID]:
                    new_data_list.append('{} {} {} {}'.format(imgPath, vehicleID, modelID, colorID))
                    # new_data_list的格式是 n行(n张图片),每行的格式是：imgPath vehicleID modelID colorID

    print('The original data list has {} samples, the new data list has {} samples.'.format(
                                 len(data_list), len(new_data_list)))
    # new_data_list: a list of[img_path, vehicleID, modelID, colorID]
    # dic: a dictionary: {modelID: {colorID: {vehicleID: [imageName, ...]}}, ...}
    return new_data_list, dic

def get_triplet_branch():
    attributes_branch=load_model('./weights/vehicleModelColor.h5')
    attributes_branch.get_layer(name='global_average_pooling2d_1').name='f_base' #inception.get_layer(index=-1)??
    f_base=attributes_branch.get_layer(name='f_base').output  #why not just use f_base=attributes_branch.get_layer(name='global_average_pooling2d_1').output
    attributes_branch.summary()

    anchor=attributes_branch.input
    positive=Input(shape=(IMG_WIDTH,IMG_HEIGHT,3),name='positive')
    negative=Input(shape=(IMG_WIDTH,IMG_HEIGHT,3),name='negative')

    f_acs=attributes_branch.get_layer(name='f_acs').output
    f_model=attributes_branch.get_layer(name='predict_model').output
    f_colour=attributes_branch.get_layer(name='predict_colour').output

    f_sls1=Dense(1024, name='sls1')(f_base)
    f_sls2=concatenate([f_sls1,f_acs], axis=-1, name='sls1_concatenate')
    f_sls2=Dense(1024, name='sls2')(f_sls2)
    f_sls2=Activation('relu',name='sls2_relu')(f_sls2) #why not use f_sls2=Dense(1024, activation='relu',name='sls2')(f_sls2)
    f_sls3=Dense(256, name='sls3')(f_sls2)#why 256?
    sls_branch=Model(attributes_branch.input,f_sls3)

    # why sls_branch can extract the features without compiling and training the branch model?
    f_sls_anchor=sls_branch(anchor)
    f_sls3_positive=sls_branch(positive)
    f_sls3_negative=sls_branch(negative)

    loss=Lambda(triplet_loss, output_shape=(1,))([f_sls_anchor,f_sls3_positive,f_sls3_negative])
    model = Model(inputs=[anchor, positive, negative], outputs=[f_model, f_colour, loss])
    #model = Model(inputs=[anchor, positive, negative], outputs=loss)
    model.summary()
    return model

def train_model():

    model=get_triplet_branch()
    optimizer=SGD(lr=LEARNING_RATE, momentum=0.9, decay=0.0, nesterov=True)
    #identity loss?
    model.compile(loss=["categorical_crossentropy","categorical_crossentropy",identity_loss],
                  loss_weights=[0.2,0.2,1],
                  optimizer=optimizer, metrics=["accuracy"] )
    # model.compile(loss= identity_loss,
    #               optimizer=optimizer, metrics=["accuracy"])
    model.summary()

    model_file_saved="./weights/Triplet_epoch={epoch:04d}-loss={loss:.4f}-modelAcc={predictions_model_acc:.4f}-colorAcc={predictions_color_acc:.4f}-val_loss={val_loss:.4f}-val_modelAcc={val_predictions_model_acc:.4f}-val_colorAcc={val_predictions_color_acc:.4f}.h5"
    checkpoint=ModelCheckpoint(model_file_saved,verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_'+monitor_index, factor=0.5,
                  patience=5, verbose=1, min_lr=0.00001)
    early_stop = EarlyStopping(monitor='val_'+monitor_index, patience=15, verbose=1)

    # 读取训练集数据
    train_data_lines = open(train_path).readlines()
    # Check if image path exists.
    train_data_lines = [root_path+w for w in train_data_lines if os.path.exists(root_path+w.strip().split(' ')[0])]

    train_data_lines, dic_train_data_lines = filter_data_list(train_data_lines)  # 过滤掉一些不能进行triplet loss计算的样本
    train_data_lines = train_data_lines
    nbr_train = len(train_data_lines)
    print('# Train Images: {}.'.format(nbr_train))
    steps_per_epoch = int(ceil(nbr_train * 1. / BATCH_SIZE))  # 批次的数目

    # 保存过滤后的图片到txt文件中
    if SAVE_FILTERED_LIST:
        # Write filtered data lines into disk.
        filtered_train_list_path = './train_vehicleModelColor_list_filtered.txt'
        f_new_train_list = open(filtered_train_list_path, 'w')
        for line in train_data_lines:
            f_new_train_list.write(line + '\n')
        f_new_train_list.close()
        print('{} has been successfully saved!'.format(filtered_train_list_path))

    # 读取验证集数据  验证集不需要过滤吗？因为验证集不需要计算损失，只需要计算评价指标，也就是车型、颜色分类准确率。
    val_data_lines = open(val_path).readlines()
    val_data_lines = [root_path+w for w in val_data_lines if os.path.exists(root_path+w.strip().split(' ')[0])]
    val_data_lines = val_data_lines
    nbr_val = len(val_data_lines)
    print('# Val Images: {}.'.format(nbr_val))
    validation_steps = int(ceil(nbr_val * 1. / BATCH_SIZE))

    model.fit_generator(generator_batch_triplet(train_data_lines, dic_train_data_lines,mode = 'train', nbr_class_one = NBR_MODELS, nbr_class_two = NBR_COLORS,batch_size = BATCH_SIZE, img_width = IMG_WIDTH,img_height = IMG_HEIGHT, random_scale = RANDOM_SCALE,shuffle = True, augment = False),
                        steps_per_epoch = steps_per_epoch, epochs = NBR_EPOCHS, verbose = 1,
                        validation_data = generator_batch_triplet(val_data_lines, { },# 这里是空的，原因是不需要求损失，不需要找anchor的positive或negative
                                         mode = 'val', nbr_class_one = NBR_MODELS, nbr_class_two = NBR_COLORS, batch_size = BATCH_SIZE, img_width = IMG_WIDTH, img_height = IMG_HEIGHT,shuffle = False, augment = False),

                        validation_steps = validation_steps,
                        callbacks = [checkpoint], initial_epoch =INITIAL_EPOCH,
                        max_queue_size = 100, workers = 1, use_multiprocessing=False)




if __name__=='__main__':
    train_model()






