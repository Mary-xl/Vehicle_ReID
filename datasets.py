# -*- coding: utf-8 -*-
datapath = '/data/VehicleID_V1.0'
train_data_lines = open(datapath+'/train_test_split/train_list.txt').readlines()

color_data_lines = open(datapath+'/attribute/color_attr.txt').readlines()

# color_data_lines

model_data_lines = open(datapath+'/attribute/model_attr.txt').readlines()


def list_dic(tmplist):
    dic={}
    for i in tmplist:
        item1,item2=i.strip().split(' ')
        dic[item1]=item2
    return dic


img_veh_dic= list_dic(train_data_lines)

veh_col_dic = list_dic(color_data_lines)

veh_mod_dic = list_dic(model_data_lines)

import os
def path_model_color_train_val_method(imgpath,ratio):
    path_model_color_list = []
    for img_veh_dic_1,img_veh_dic_2 in img_veh_dic.items():
        if veh_col_dic.get(img_veh_dic_2):
            img = os.path.join(imgpath,img_veh_dic_1+'.jpg')
            path_model_color_list.append(img+' '+veh_mod_dic[img_veh_dic_2]+' '+veh_col_dic[img_veh_dic_2]+'\n')
    
    train_num = int(len(path_model_color_list)*ratio)
    with open('path_model_color_train.txt','w') as f:
        for i in path_model_color_list[0:train_num] :
            f.write(i)
    with open('path_model_color_test.txt','w') as f:
        for i in path_model_color_list[train_num:] :
            f.write(i)


path_model_color_train_val_method(datapath+'/image/',0.8)
