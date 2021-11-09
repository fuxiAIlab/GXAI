# -*- coding: utf-8 -*-
# @Time    : 2020/1/19 11:04
# @Author  : Yu Xiong
# @File    : sequence_nsh_pic.py
# @Software: PyCharm

import numpy as np
import json
import os
import random
import shap
from keras.models import load_model
import matplotlib.pyplot as plt


def explain_pic(shap_value_normal_path, shap_value_plug_path):
    # 构建SHAP解释器对测试数据进行解释
    MODEL_PATH = 'model_nsh_sequence/Cnn.weights.005-0.9110.hdf5'
    train_100_path = 'data_sequence/train_data_100.json'
    x_bg = json.load(open(train_100_path))
    bg_x = [np.array(x_bg[0][:50]), np.array(x_bg[1][:50]), np.array(x_bg[2][:50]), np.array(x_bg[3][:50]),
            np.array(x_bg[4][:50])]
    model = load_model(MODEL_PATH, compile=False)
    explainer = shap.DeepExplainer(model, bg_x)
    print('build explainer')

    test_path = 'data_sequence/test_data.json'
    label_path = 'data_sequence/label.json'
    voc_path = 'data_sequence/logdisign_id_voc.json'
    x_test = json.load(open(test_path))
    y = json.load(open(label_path))
    voc = json.load(open(voc_path))
    voc_re = {value: key for key, value in voc.items()}
    voc_re[0] = 'pad'
    train_end, valid_end = int(0.7 * len(y)), int(0.8 * len(y))
    r = random.random
    random.seed(2)
    role_ids = list(y.keys())
    role_ids.sort()
    random.shuffle(role_ids, random=r)
    train_id, valid_id, test_id = role_ids[:train_end], role_ids[train_end + 1:valid_end], role_ids[valid_end:]
    plug_file_list = os.listdir(shap_value_plug_path)
    for file in plug_file_list:
        shap_value_np = np.load(shap_value_plug_path + file)
        role_id = file.split('.')[0]
        id_index = test_id.index(role_id)
        x = [np.array(x_test[0][id_index]), np.array(x_test[1][id_index]), np.array(x_test[2][id_index]),
             np.array(x_test[3][id_index]), np.array(x_test[4][id_index])]
        voc_cnt = {}
        voc_shap_value = {}
        for i in range(5):
            for j in range(x[i].shape[0]):
                id = voc_re[x[i][j]]
                shap_value = shap_value_np[i][j]
                if id not in voc_cnt:
                    voc_cnt[id] = 1
                    voc_shap_value[id] = shap_value
                else:
                    voc_cnt[id] += 1
                    voc_shap_value[id] += shap_value
        id_name = []
        data = []
        shap_values = []
        for key, value in voc_cnt.items():
            id_name.append(key)
            data.append(value)
            shap_values.append(voc_shap_value[key])
        shap.force_plot(explainer.expected_value, np.array(shap_values), np.array(data), feature_names=id_name,
                        matplotlib=True, show=False)
        plt.savefig('data_sequence/cnn/balance_local_bot/{}.pdf'.format(role_id), dpi=100, bbox_inches='tight')
        plt.close()

    normal_file_list = os.listdir(shap_value_normal_path)
    for file in normal_file_list:
        shap_value_np = np.load(shap_value_normal_path + file)
        role_id = file.split('.')[0]
        id_index = test_id.index(role_id)
        x = [np.array(x_test[0][id_index]),np.array(x_test[1][id_index]),np.array(x_test[2][id_index]),np.array(x_test[3][id_index]),np.array(x_test[4][id_index])]
        voc_cnt = {}
        voc_shap_value = {}
        for i in range(5):
            for j in range(x[i].shape[0]):
                id = voc_re[x[i][j]]
                shap_value = shap_value_np[i][j]
                if id not in voc_cnt:
                    voc_cnt[id] = 1
                    voc_shap_value[id] = shap_value
                else:
                    voc_cnt[id] += 1
                    voc_shap_value[id] += shap_value
        id_name = []
        data = []
        shap_values = []
        for key, value in voc_cnt.items():
            id_name.append(key)
            data.append(value)
            shap_values.append(voc_shap_value[key])
        shap.force_plot(explainer.expected_value, np.array(shap_values), np.array(data), feature_names=id_name, matplotlib=True, show=False)
        plt.savefig('data_sequence/cnn/balance_local_normal/{}.pdf'.format(role_id), dpi=100, bbox_inches='tight')
        plt.close()


def explain_summary_pic(shap_value_normal_path, shap_value_plug_path, expected_value):
    test_path = 'data_sequence/test_data.json'
    label_path = 'data_sequence/label.json'
    voc_path = 'data_sequence/logdisign_id_voc.json'
    x_test = json.load(open(test_path))
    y = json.load(open(label_path))
    voc = json.load(open(voc_path))
    voc['pad'] = 0
    voc_re = {value: key for key, value in voc.items()}
    # voc_re[0] = 'pad'
    train_end, valid_end = int(0.7 * len(y)), int(0.8 * len(y))
    r = random.random
    random.seed(2)
    role_ids = list(y.keys())
    role_ids.sort()
    random.shuffle(role_ids, random=r)
    train_id, valid_id, test_id = role_ids[:train_end], role_ids[train_end + 1:valid_end], role_ids[valid_end:]

    feature_names = list(voc.keys())
    # print(feature_names)
    plug_file_list = os.listdir(shap_value_plug_path)
    shap_values_plug = np.zeros((len(plug_file_list), len(voc)))
    for idx in range(len(plug_file_list)):
        file = plug_file_list[idx]
        shap_value_np = np.load(shap_value_plug_path + file)
        role_id = file.split('.')[0]
        id_index = test_id.index(role_id)
        x = [np.array(x_test[0][id_index]), np.array(x_test[1][id_index]), np.array(x_test[2][id_index]),
             np.array(x_test[3][id_index]), np.array(x_test[4][id_index])]
        voc_cnt = {}
        voc_shap_value = {}
        for i in range(5):
            for j in range(x[i].shape[0]):
                id = voc_re[x[i][j]]
                shap_value = shap_value_np[i][j]
                if id not in voc_cnt:
                    voc_cnt[id] = 1
                    voc_shap_value[id] = shap_value
                else:
                    voc_cnt[id] += 1
                    voc_shap_value[id] += shap_value
        for key, value in voc_cnt.items():
            feature_index = feature_names.index(key)
            shap_values_plug[idx][feature_index] = voc_shap_value[key]
    print('plug shap value done')
    normal_file_list = os.listdir(shap_value_normal_path)
    shap_values_normal = np.zeros((len(normal_file_list), len(voc)))
    for idx in range(len(normal_file_list)):
        file = normal_file_list[idx]
        shap_value_np = np.load(shap_value_normal_path + file)
        role_id = file.split('.')[0]
        id_index = test_id.index(role_id)
        x = [np.array(x_test[0][id_index]), np.array(x_test[1][id_index]), np.array(x_test[2][id_index]),
             np.array(x_test[3][id_index]), np.array(x_test[4][id_index])]
        voc_cnt = {}
        voc_shap_value = {}
        for i in range(5):
            for j in range(x[i].shape[0]):
                id = voc_re[x[i][j]]
                shap_value = shap_value_np[i][j]
                if id not in voc_cnt:
                    voc_cnt[id] = 1
                    voc_shap_value[id] = shap_value
                else:
                    voc_cnt[id] += 1
                    voc_shap_value[id] += shap_value
        for key, value in voc_cnt.items():
            feature_index = feature_names.index(key)
            shap_values_normal[idx][feature_index] = voc_shap_value[key]
    print('normal shap_value done')
    shap_values = np.concatenate((shap_values_plug, shap_values_normal))

    # 获取行为重要性排名列表
    feature_inds = shap.summary_plot(shap_values, feature_names=feature_names, max_display=shap_values.shape[1], plot_type='bar', show=False)
    feature_inds = feature_inds[::-1]
    with open('data_sequence/blstm/feature_inds.txt', 'w') as f:
        for feat in feature_inds:
            print(feature_names[feat])
            f.write(str(feature_names[feat])+'\n')
    # 画group图
    # shap_values = shap_values[:, feature_inds[:100]]
    # feature_names = np.array(feature_names)[feature_inds[:100]]
    # group = shap.force_plot(expected_value, shap_values, feature_names=feature_names)
    # save_path = 'data_sequence/cnn/group_100.html'
    # shap.save_html(save_path, group)

    # 画global图
    # plt.savefig('data_sequence/cnn/global_bar_16.pdf', dpi=100, bbox_inches='tight')
    # plt.close()
    # shap.summary_plot(shap_values, feature_names=feature_names, max_display=100, show=False)
    # plt.savefig('data_sequence/cnn/global_16.pdf', dpi=100, bbox_inches='tight')
    # plt.close()


if __name__ == '__main__':
    shap_value_normal_path = 'data_sequence/shap_value_normal/'
    shap_value_plug_path = 'data_sequence/shap_value_plug/'
    explain_summary_pic(shap_value_normal_path, shap_value_plug_path)
    # explain_pic(shap_value_normal_path, shap_value_plug_path)

