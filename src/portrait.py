# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import lightgbm as lgb
import catboost as cbst
import time
import pickle
from keras.layers import Input, Dense
from keras.models import Model
from keras.callbacks import ModelCheckpoint
import shap
from plot_helper import sample_force_plot, group_force_plot, summary_plot


# 基于表格数据训练树模型
def train_model(portrait_dir, label_dir, model_type='xgb'):
    plug_id = []
    normal_id = []
    with open(label_dir, 'r') as f:
        for line in f.readlines():
            line = line.strip().split(',')
            role_id = int(line[0])
            label = int(line[1])
            if label == 0:
                normal_id.append(role_id)
            else:
                plug_id.append(role_id)
    data_df = pd.read_csv(portrait_dir)
    normal_df = data_df[data_df['role_id'].isin(normal_id)]
    plug_df = data_df[data_df['role_id'].isin(plug_id)]
    data_df = pd.concat([normal_df, plug_df])
    label = np.append(np.zeros(len(normal_id)), np.ones(len(plug_id)))
    data_df['create_date'] = pd.to_datetime(data_df['ds']) - pd.to_datetime(data_df['create_date'])
    data_df['create_date'] = data_df['create_date'].dt.days
    data_df.drop(['role_id', 'ds', 'role_account_name', 'role_name', 'server', 'create_time', 'phone_num', 'idn', 'dsn',
                  'mac', 'mid', 'smb', 'date', 'punish_cnt', 'ban_status', 'pk_amt', 'jiahei_cnt', 'is_forbidden',
                  'pfv_total_score', 'pfv_skill_score', 'pfv_practice_score', 'pfv_level', 'pfv_sub_level', 'pfv_ori_level',
                  'pfv_ori_sub_level', 'ttsw', 'qldb', '2w_d_avg_send_flower_cnt', '1st_apprentice_level',
                  '1st_apprentice_date', 'late_apprentice_level', 'late_apprentice_date', 'chushi_level', 'chushi_date',
                  '2w_bfr_baishi_d_avg_onl_tm', '1w_bfr_chushi_d_avg_onl_tm', '1w_aft_chushi_d_avg_onl_tm', 'shifu_id',
                  'couple_id', 'is_join_bhps_sj_task', '2w_d_avg_wyc_time', '2w_d_avg_cy_time', '2w_d_avg_qm_time',
                  '2w_d_avg_qm_yx_time', '2w_d_avg_jd_time', '1w_wyc_time', '1w_cy_time', '1w_qm_yx_time', '1w_jd_time',
                  '1w_fxtlg_time', '1w_wyc_yx_time', '1w_lf_yx_time', '1w_wlfyl_yx_time', '1w_jhtz_time', '1w_hsc_time',
                  'season_single_win_rate', 'season_team_win_rate', 'season_task_unfinish_rate', 'season_gua_ji_rate',
                  'season_leaving_early_rate', 'season_negatie_fight_rate', 'season_d_avg_final_score', 'season_d_avg_class_score',
                  'season_d_avg_strategy_score', 'season_d_avg_all_rank', 'season_d_avg_side_rank', 'season_d_avg_class_rank',
                  'season_d_avg_win_lose_score', 'season_d_avg_raw_score', 'season_d_avg_kill_count', 'season_d_avg_help_count',
                  'season_d_avg_hurted_num', 'season_d_avg_output', 'season_d_avg_cure_num', 'season_d_avg_die_cnt',
                  'season_d_avg_max_kill_player_count', 'season_d_avg_fight_strategy_score', 'season_d_avg_fight_score',
                  'season_d_avg_home_total_score', 'season_d_avg_opponent_total_score', 'season_kill_rate', 'season_help_rate',
                  'season_hurted_rate', 'season_output_rate', 'season_cure_rate', 'season_single_rate', 'season_two_rate',
                  'season_two_more_rate', 'season_team_single_count', 'season_team_more_count', '2w_single_win_rate',
                  '2w_team_win_rate', '2w_task_unfinish_rate', '2w_gua_ji_rate', '2w_leaving_early_rate', '2w_negatie_fight_rate',
                  '2w_d_avg_final_score', '2w_d_avg_class_score', '2w_d_avg_strategy_score', '2w_d_avg_all_rank', '2w_d_avg_side_rank',
                  '2w_d_avg_class_rank', '2w_d_avg_win_lose_score', '2w_d_avg_raw_score', '2w_d_avg_kill_count',
                  '2w_d_avg_help_count', '2w_d_avg_hurted_num', '2w_d_avg_output', '2w_d_avg_cure_num', '2w_d_avg_die_cnt',
                  '2w_d_avg_max_kill_player_count', '2w_d_avg_fight_strategy_score', '2w_d_avg_fight_score', '2w_d_avg_home_total_score',
                  '2w_d_avg_opponent_total_score', '2w_kill_rate', '2w_help_rate', '2w_hurted_rate', '2w_output_rate',
                  '2w_cure_rate', '2w_single_rate', '2w_two_rate', '2w_two_more_rate', '2w_team_single_count',
                  '2w_team_more_count', '3m_single_win_rate', '3m_team_win_rate', '3m_task_unfinish_rate', '3m_gua_ji_rate',
                  '3m_leaving_early_rate', '3m_negatie_fight_rate', '3m_d_avg_final_score', '3m_d_avg_class_score',
                  '3m_d_avg_strategy_score', '3m_d_avg_all_rank', '3m_d_avg_side_rank', '3m_d_avg_class_rank',
                  '3m_d_avg_win_lose_score', '3m_d_avg_raw_score', '3m_d_avg_kill_count', '3m_d_avg_help_count',
                  '3m_d_avg_hurted_num', '3m_d_avg_output', '3m_d_avg_cure_num', '3m_d_avg_die_cnt', '3m_d_avg_max_kill_player_count',
                  '3m_d_avg_fight_strategy_score', '3m_d_avg_fight_score', '3m_d_avg_home_total_score', '3m_d_avg_opponent_total_score',
                  '3m_kill_rate', '3m_help_rate', '3m_hurted_rate', '3m_output_rate', '3m_cure_rate', '3m_single_rate',
                  '3m_two_rate', '3m_two_more_rate', '3m_team_single_count', '3m_team_more_count', 'is_join_feb_mtjt_bhhd',
                  'is_join_feb_yxjm_bhhd', 'is_join_bhls', 'is_join_szww', 'is_join_yq_sx', 'is_join_ss', 'is_join_sh',
                  'is_join_klxyb', 'is_join_wyc', 'is_join_fxtlg', 'is_join_wyc_yx', 'is_join_wlfyl_yx', 'is_join_yqw',
                  'is_join_qwl', 'is_join_lpzdz', 'is_join_bwdh', 'is_join_qm_yx', 'is_join_lf_yx', 'is_join_fys',
                  'is_join_hsc', 'is_join_lszd', 'is_join_ljzg', 'is_join_klxyj', '2w_d_avg_yqt_cnt', '2w_d_avg_lyzsryb_cnt',
                  '2w_d_avg_wyc_die_cnt', '2w_d_avg_wyc_yx_die_cnt', '2w_d_avg_wlfyl_yx_die_cnt', '2w_d_avg_tlg_dit_cnt',
                  '2w_d_avg_bounty_task_cnt', '1m_d_avg_pvp_task_rto', '1m_d_avg_pve_task_rto', '2w_d_avg_xyd_use_cnt',
                  '1m_charge_yuanbao_amt', '2w_shop_money_get_amt', 'acm_up_9_level_skill_amt', 'shop_num', 'latest_rare_word_num_sjwq',
                  'acm_sjbl_cnt', 'churn_friends_ratio', '2w_friends_chat_num', 'del_shitu_acm_cnt', 'deled_shitu_acm_cnt',
                  'shifu_log_off_day', 'acm_divorce_cnt', 'acm_chujia_cnt', 'couple_log_off_day', '2w_wyc_time_rto',
                  '2w_qmlf_yx_time_rto', '2w_cjg_die_cnt', '2w_leisure_yabiao_die_cnt', '2w_passionfight_die_cnt',
                  '2w_sjtx_die_cnt', '2w_leisure_task_die_cnt', '2w_shitu_play_cnt', 'shop_bankrupt_num'], axis=1, inplace=True)
    data_df.fillna(value=0, inplace=True)

    if model_type == 'xgb':
        data_df.drop(['cpu_name', 'device_id', 'device_name', 'dx_version', 'os_name', 'os_version', 'vendor_id',
                      'role_gender', 'clothes', 'money_source', 'marital_relation', 'latest_ml_chapter', '45level_latest_ml_chapter',
                      '60level_latest_ml_chapter', 'latest_log_time'], axis=1, inplace=True)
        train_data, test_data, train_label, test_label = train_test_split(data_df, label, test_size=0.2, random_state=42)
        dtrain = xgb.DMatrix(train_data, label=train_label)
        dtest = xgb.DMatrix(test_data, label=test_label)
        params = {
            'booster': 'gbtree',
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'max_depth': 7,
            'lambda': 1,
            'subsample': 0.75,
            'colsample_bytree': 0.75,
            'min_child_weight': 2,
            'eta': 0.025,
            'seed': 0,
            'nthread': 8,
            'silent': 1
        }
        watchlist = [(dtest, 'validation')]
        start = time.clock()
        model = xgb.train(params, dtrain, num_boost_round=1000, early_stopping_rounds=30, evals=watchlist)
        mid = time.clock()
        predictions = model.predict(dtest)
        end = time.clock()
        print('Training time is {}'.format(str(mid - start)))
        print('Inference time is {}'.format(str(end - mid)))
        auc = roc_auc_score(test_label, predictions)
        print('The roc of prediction is {}'.format(auc))
        model.save_model('model_nsh/xgb_auc{:.4f}.model'.format(auc))
        pred_norm = [round(score) for score in predictions]
        print('The acc of prediction is {}'.format(accuracy_score(test_label, pred_norm)))
        print('The precision of prediction is {}'.format(precision_score(test_label, pred_norm)))
        print('The recall of prediction is {}'.format(recall_score(test_label, pred_norm)))
        print('The F1 score of prediction is {}'.format(f1_score(test_label, pred_norm)))

    if model_type == 'lgb':
        data_df.drop(['cpu_name', 'device_id', 'device_name', 'dx_version', 'os_name', 'os_version', 'vendor_id',
                      'role_gender', 'clothes', 'money_source', 'marital_relation', 'latest_ml_chapter', '45level_latest_ml_chapter',
                      '60level_latest_ml_chapter', 'latest_log_time'], axis=1, inplace=True)
        train_data, test_data, train_label, test_label = train_test_split(data_df, label, test_size=0.2,
                                                                          random_state=42)
        lgb_train = lgb.Dataset(train_data, train_label)
        lgb_test = lgb.Dataset(test_data, test_label, reference=lgb_train)
        params = {
            'task': 'train',
            'boosting_type': 'gbdt',
            'objective': 'binary',
            'metric': {'binary_logloss', 'auc'},
            'num_leaves': 31,
            'min_data_in_leaf': 20,
            'max_depth': 6,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': 0
        }
        start = time.clock()
        model = lgb.train(params, lgb_train, num_boost_round=1000, valid_sets=lgb_test, early_stopping_rounds=20)
        mid = time.clock()
        predictions = model.predict(test_data, num_iteration=model.best_iteration)
        end = time.clock()
        print('Training time is {}'.format(str(mid - start)))
        print('Inference time is {}'.format(str(end - mid)))
        auc = roc_auc_score(test_label, predictions)
        print('The roc of prediction is {}'.format(auc))
        model.save_model('model_nsh/lgb_auc{:.4f}.model'.format(auc))
        pred_norm = [round(score) for score in predictions]
        print('The acc of prediction is {}'.format(accuracy_score(test_label, pred_norm)))
        print('The precision of prediction is {}'.format(precision_score(test_label, pred_norm)))
        print('The recall of prediction is {}'.format(recall_score(test_label, pred_norm)))
        print('The F1 score of prediction is {}'.format(f1_score(test_label, pred_norm)))

    if model_type == 'cbst':
        data_df.drop(['cpu_name', 'device_id', 'device_name', 'dx_version', 'os_name', 'os_version', 'vendor_id'],
                     axis=1, inplace=True)
        train_data, test_data, train_label, test_label = train_test_split(data_df, label, test_size=0.2,
                                                                          random_state=42)
        model = cbst.CatBoostClassifier(iterations=1000, depth=6, learning_rate=0.5,
                                        cat_features=['role_gender', 'clothes', 'money_source', 'marital_relation', 'latest_ml_chapter', '45level_latest_ml_chapter',
                                                    '60level_latest_ml_chapter', 'latest_log_time'],
                                        loss_function='Logloss', eval_metric='AUC', random_seed=696, reg_lambda=3,
                                        verbose=True)
        start = time.clock()
        model.fit(train_data, train_label, eval_set=(test_data, test_label), early_stopping_rounds=20)
        mid = time.clock()
        predictions = model.predict_proba(test_data)[:, 1]
        end = time.clock()
        print('Training time is {}'.format(str(mid - start)))
        print('Inference time is {}'.format(str(end - mid)))
        auc = roc_auc_score(test_label, predictions)
        print('The roc of prediction is {}'.format(auc))
        model.save_model('model_nsh/cbst_auc{:.4f}.model'.format(auc))
        pred_norm = [round(score) for score in predictions]
        print('The acc of prediction is {}'.format(accuracy_score(test_label, pred_norm)))
        print('The precision of prediction is {}'.format(precision_score(test_label, pred_norm)))
        print('The recall of prediction is {}'.format(recall_score(test_label, pred_norm)))
        print('The F1 score of prediction is {}'.format(f1_score(test_label, pred_norm)))

    if model_type == 'rf':
        data_df.drop(['cpu_name', 'device_id', 'device_name', 'dx_version', 'os_name', 'os_version', 'vendor_id',
                      'role_gender', 'clothes', 'money_source', 'marital_relation', 'latest_ml_chapter',
                      '45level_latest_ml_chapter', '60level_latest_ml_chapter', 'latest_log_time'], axis=1, inplace=True)

        train_data, test_data, train_label, test_label = train_test_split(data_df, label, test_size=0.2,
                                                                          random_state=42)
        clf = RandomForestClassifier(oob_score=True, random_state=10, n_estimators=500, max_depth=7)
        start = time.clock()
        clf.fit(train_data, train_label)
        mid = time.clock()
        predictions = clf.predict_proba(test_data)[:, 1]
        end = time.clock()
        print('Training time is {}'.format(str(mid - start)))
        print('Inference time is {}'.format(str(end - mid)))
        auc = roc_auc_score(test_label, predictions)
        print('The roc of prediction is {}'.format(auc))
        with open('model_nsh/rf_auc{:.4f}.pickle'.format(auc), 'wb') as f:
            pickle.dump(clf, f)
        # with open('model', 'rb') as f:
        #     clf = pickle.load(f)
        pred_norm = [round(score) for score in predictions]
        print('The acc of prediction is {}'.format(accuracy_score(test_label, pred_norm)))
        print('The precision of prediction is {}'.format(precision_score(test_label, pred_norm)))
        print('The recall of prediction is {}'.format(recall_score(test_label, pred_norm)))
        print('The F1 score of prediction is {}'.format(f1_score(test_label, pred_norm)))

    if model_type == 'lr':
        data_df.drop(['cpu_name', 'device_id', 'device_name', 'dx_version', 'os_name', 'os_version', 'vendor_id',
                      'role_gender', 'clothes', 'money_source', 'marital_relation', 'latest_ml_chapter',
                      '45level_latest_ml_chapter', '60level_latest_ml_chapter', 'latest_log_time'], axis=1,
                     inplace=True)
        train_data, test_data, train_label, test_label = train_test_split(data_df, label, test_size=0.2,
                                                                          random_state=42)
        ss = StandardScaler()
        train_data = ss.fit_transform(train_data)
        test_data = ss.fit_transform(test_data)
        lr = LogisticRegression()
        start = time.clock()
        lr.fit(train_data, train_label)
        mid = time.clock()
        predictions = lr.predict_proba(test_data)[:, 1]
        end = time.clock()
        print('Training time is {}'.format(str(mid - start)))
        print('Inference time is {}'.format(str(end - mid)))
        auc = roc_auc_score(test_label, predictions)
        print('The roc of prediction is {}'.format(auc))
        with open('model_nsh/lr_auc{:.4f}.pickle'.format(auc), 'wb') as f:
            pickle.dump(lr, f)
        pred_norm = [round(score) for score in predictions]
        print('The acc of prediction is {}'.format(accuracy_score(test_label, pred_norm)))
        print('The precision of prediction is {}'.format(precision_score(test_label, pred_norm)))
        print('The recall of prediction is {}'.format(recall_score(test_label, pred_norm)))
        print('The F1 score of prediction is {}'.format(f1_score(test_label, pred_norm)))

    if model_type == 'mlp':
        data_df.drop(['cpu_name', 'device_id', 'device_name', 'dx_version', 'os_name', 'os_version', 'vendor_id',
                      'role_gender', 'clothes', 'money_source', 'marital_relation', 'latest_ml_chapter',
                      '45level_latest_ml_chapter', '60level_latest_ml_chapter', 'latest_log_time'], axis=1,
                     inplace=True)
        train_data, test_data, train_label, test_label = train_test_split(data_df, label, test_size=0.2,
                                                                          random_state=42)
        scaler = StandardScaler().fit(train_data)
        train_data = scaler.transform(train_data)
        test_data = scaler.transform(test_data)
        inputs = Input(shape=(train_data.shape[1],))
        dense1 = Dense(64, activation='tanh')(inputs)
        dense2 = Dense(64, activation='tanh')(dense1)
        outputs = Dense(1, activation='sigmoid')(dense2)
        model = Model(inputs, outputs)
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        weight_path = 'model_nsh/mlp.hdf5'
        check_point = ModelCheckpoint(weight_path, monitor='val_acc', verbose=1, save_best_only=True, mode='auto')
        callbacks_list = [check_point]
        start = time.clock()
        model.fit(train_data, train_label, epochs=15, batch_size=32, validation_data=(test_data, test_label),
                  callbacks=callbacks_list)
        mid = time.clock()
        # model.load_weights(weight_path)
        predictions = model.predict(test_data)
        predictions = [score[0] for score in predictions]
        print(predictions)
        end = time.clock()
        print('Training time is {}'.format(str(mid - start)))
        print('Inference time is {}'.format(str(end - mid)))
        auc = roc_auc_score(test_label, predictions)
        print('The roc of prediction is {}'.format(auc))
        pred_norm = [round(score) for score in predictions]
        print('The acc of prediction is {}'.format(accuracy_score(test_label, pred_norm)))
        print('The precision of prediction is {}'.format(precision_score(test_label, pred_norm)))
        print('The recall of prediction is {}'.format(recall_score(test_label, pred_norm)))
        print('The F1 score of prediction is {}'.format(f1_score(test_label, pred_norm)))


# 解释+绘图
def explain_pic(portrait_dir, label_dir, model_type='xgb'):
    plug_id = []
    normal_id = []
    with open(label_dir, 'r') as f:
        for line in f.readlines():
            line = line.strip().split(',')
            role_id = int(line[0])
            label = int(line[1])
            if label == 0:
                normal_id.append(role_id)
            else:
                plug_id.append(role_id)
    data_df = pd.read_csv(portrait_dir)
    normal_df = data_df[data_df['role_id'].isin(normal_id)]
    normal_size = normal_df.shape[0]
    plug_df = data_df[data_df['role_id'].isin(plug_id)]
    plug_size = plug_df.shape[0]
    data_df = pd.concat([normal_df, plug_df])
    label = np.append(np.zeros(len(normal_id)), np.ones(len(plug_id)))
    data_df['create_date'] = pd.to_datetime(data_df['ds']) - pd.to_datetime(data_df['create_date'])
    data_df['create_date'] = data_df['create_date'].dt.days
    role_ids = data_df['role_id']
    data_df.drop(['role_id', 'ds', 'role_account_name', 'role_name', 'server', 'create_time', 'phone_num', 'idn', 'dsn',
                  'mac', 'mid', 'smb', 'date', 'punish_cnt', 'ban_status', 'pk_amt', 'jiahei_cnt', 'is_forbidden',
                  'pfv_total_score', 'pfv_skill_score', 'pfv_practice_score', 'pfv_level', 'pfv_sub_level', 'pfv_ori_level',
                  'pfv_ori_sub_level', 'ttsw', 'qldb', '2w_d_avg_send_flower_cnt', '1st_apprentice_level',
                  '1st_apprentice_date', 'late_apprentice_level', 'late_apprentice_date', 'chushi_level', 'chushi_date',
                  '2w_bfr_baishi_d_avg_onl_tm', '1w_bfr_chushi_d_avg_onl_tm', '1w_aft_chushi_d_avg_onl_tm', 'shifu_id',
                  'couple_id', 'is_join_bhps_sj_task', '2w_d_avg_wyc_time', '2w_d_avg_cy_time', '2w_d_avg_qm_time',
                  '2w_d_avg_qm_yx_time', '2w_d_avg_jd_time', '1w_wyc_time', '1w_cy_time', '1w_qm_yx_time', '1w_jd_time',
                  '1w_fxtlg_time', '1w_wyc_yx_time', '1w_lf_yx_time', '1w_wlfyl_yx_time', '1w_jhtz_time', '1w_hsc_time',
                  'season_single_win_rate', 'season_team_win_rate', 'season_task_unfinish_rate', 'season_gua_ji_rate',
                  'season_leaving_early_rate', 'season_negatie_fight_rate', 'season_d_avg_final_score', 'season_d_avg_class_score',
                  'season_d_avg_strategy_score', 'season_d_avg_all_rank', 'season_d_avg_side_rank', 'season_d_avg_class_rank',
                  'season_d_avg_win_lose_score', 'season_d_avg_raw_score', 'season_d_avg_kill_count', 'season_d_avg_help_count',
                  'season_d_avg_hurted_num', 'season_d_avg_output', 'season_d_avg_cure_num', 'season_d_avg_die_cnt',
                  'season_d_avg_max_kill_player_count', 'season_d_avg_fight_strategy_score', 'season_d_avg_fight_score',
                  'season_d_avg_home_total_score', 'season_d_avg_opponent_total_score', 'season_kill_rate', 'season_help_rate',
                  'season_hurted_rate', 'season_output_rate', 'season_cure_rate', 'season_single_rate', 'season_two_rate',
                  'season_two_more_rate', 'season_team_single_count', 'season_team_more_count', '2w_single_win_rate',
                  '2w_team_win_rate', '2w_task_unfinish_rate', '2w_gua_ji_rate', '2w_leaving_early_rate', '2w_negatie_fight_rate',
                  '2w_d_avg_final_score', '2w_d_avg_class_score', '2w_d_avg_strategy_score', '2w_d_avg_all_rank', '2w_d_avg_side_rank',
                  '2w_d_avg_class_rank', '2w_d_avg_win_lose_score', '2w_d_avg_raw_score', '2w_d_avg_kill_count',
                  '2w_d_avg_help_count', '2w_d_avg_hurted_num', '2w_d_avg_output', '2w_d_avg_cure_num', '2w_d_avg_die_cnt',
                  '2w_d_avg_max_kill_player_count', '2w_d_avg_fight_strategy_score', '2w_d_avg_fight_score', '2w_d_avg_home_total_score',
                  '2w_d_avg_opponent_total_score', '2w_kill_rate', '2w_help_rate', '2w_hurted_rate', '2w_output_rate',
                  '2w_cure_rate', '2w_single_rate', '2w_two_rate', '2w_two_more_rate', '2w_team_single_count',
                  '2w_team_more_count', '3m_single_win_rate', '3m_team_win_rate', '3m_task_unfinish_rate', '3m_gua_ji_rate',
                  '3m_leaving_early_rate', '3m_negatie_fight_rate', '3m_d_avg_final_score', '3m_d_avg_class_score',
                  '3m_d_avg_strategy_score', '3m_d_avg_all_rank', '3m_d_avg_side_rank', '3m_d_avg_class_rank',
                  '3m_d_avg_win_lose_score', '3m_d_avg_raw_score', '3m_d_avg_kill_count', '3m_d_avg_help_count',
                  '3m_d_avg_hurted_num', '3m_d_avg_output', '3m_d_avg_cure_num', '3m_d_avg_die_cnt', '3m_d_avg_max_kill_player_count',
                  '3m_d_avg_fight_strategy_score', '3m_d_avg_fight_score', '3m_d_avg_home_total_score', '3m_d_avg_opponent_total_score',
                  '3m_kill_rate', '3m_help_rate', '3m_hurted_rate', '3m_output_rate', '3m_cure_rate', '3m_single_rate',
                  '3m_two_rate', '3m_two_more_rate', '3m_team_single_count', '3m_team_more_count', 'is_join_feb_mtjt_bhhd',
                  'is_join_feb_yxjm_bhhd', 'is_join_bhls', 'is_join_szww', 'is_join_yq_sx', 'is_join_ss', 'is_join_sh',
                  'is_join_klxyb', 'is_join_wyc', 'is_join_fxtlg', 'is_join_wyc_yx', 'is_join_wlfyl_yx', 'is_join_yqw',
                  'is_join_qwl', 'is_join_lpzdz', 'is_join_bwdh', 'is_join_qm_yx', 'is_join_lf_yx', 'is_join_fys',
                  'is_join_hsc', 'is_join_lszd', 'is_join_ljzg', 'is_join_klxyj', '2w_d_avg_yqt_cnt', '2w_d_avg_lyzsryb_cnt',
                  '2w_d_avg_wyc_die_cnt', '2w_d_avg_wyc_yx_die_cnt', '2w_d_avg_wlfyl_yx_die_cnt', '2w_d_avg_tlg_dit_cnt',
                  '2w_d_avg_bounty_task_cnt', '1m_d_avg_pvp_task_rto', '1m_d_avg_pve_task_rto', '2w_d_avg_xyd_use_cnt',
                  '1m_charge_yuanbao_amt', '2w_shop_money_get_amt', 'acm_up_9_level_skill_amt', 'shop_num', 'latest_rare_word_num_sjwq',
                  'acm_sjbl_cnt', 'churn_friends_ratio', '2w_friends_chat_num', 'del_shitu_acm_cnt', 'deled_shitu_acm_cnt',
                  'shifu_log_off_day', 'acm_divorce_cnt', 'acm_chujia_cnt', 'couple_log_off_day', '2w_wyc_time_rto',
                  '2w_qmlf_yx_time_rto', '2w_cjg_die_cnt', '2w_leisure_yabiao_die_cnt', '2w_passionfight_die_cnt',
                  '2w_sjtx_die_cnt', '2w_leisure_task_die_cnt', '2w_shitu_play_cnt', 'shop_bankrupt_num'], axis=1, inplace=True)
    data_df.fillna(value=0, inplace=True)

    if model_type == 'xgb':
        data_df.drop(['cpu_name', 'device_id', 'device_name', 'dx_version', 'os_name', 'os_version', 'vendor_id',
                      'role_gender', 'clothes', 'money_source', 'marital_relation', 'latest_ml_chapter', '45level_latest_ml_chapter',
                      '60level_latest_ml_chapter', 'latest_log_time'], axis=1, inplace=True)
        # ddata = xgb.DMatrix(data_df)
        # model = xgb.Booster(model_file='model_nsh/xgb_auc0.9876.model')
        # predictions = model.predict(ddata)
        # explainer = shap.TreeExplainer(model)
        # start = time.clock()
        # shap_values = explainer.shap_values(ddata)
        # end = time.clock()
        # print('Explaining time is {}'.format(str(end - start)))
        # np.save('data_nsh/xgb_shap_values.npy', shap_values)
        col_list = list(data_df.columns)
        need_features = ['physical_memory_size', 'watch_movie_acm_pct_avg', '2w_d_avg_team_chat_cnt', '2w_d_avg_world_send_msg_cnt',
          '2w_d_avg_use_expression_cnt', '2w_task_giveup_rto', '2w_bl_task_rto', '1m_d_avg_pvp_time_rto',
          'fund', '45level_f_bl_task_acm_num', 'display_memory_localized_size', '2w_equip_play_time_rto',
          '2w_create_team_rto', 'nie_lian_time', '2w_equip_score_upgrade', '2w_sjtx_time_rto']
        data_df = data_df[need_features]
        need_index = []
        for feat in need_features:
            need_index.append(col_list.index(feat))
        data_df.rename(columns={'fund': 'guild_fund'}, inplace=True)
        dic = {'physical_memory_size': '内存', 'watch_movie_acm_pct_avg': '累计观看剧情平均百分比', '2w_d_avg_team_chat_cnt': '近两周日均队伍聊天次数',
               '2w_d_avg_world_send_msg_cnt': '近两周日均世界频道聊天次数', '2w_d_avg_use_expression_cnt': '近两周日均表情使用次数', '2w_task_giveup_rto': '近两周内放弃任务占比',
               '2w_bl_task_rto': '近两周支线任务占比', '1m_d_avg_pvp_time_rto': '近一月日均pvp时长占比', 'guild_fund': '帮会资金', '45level_f_bl_task_acm_num': '45前累计完成的支线任务个数',
               'display_memory_localized_size': '显存', '2w_equip_play_time_rto': '近两周装备玩法时长占比', '2w_create_team_rto': '近两周创建队伍占比',
               'nie_lian_time': '捏脸时长', '2w_equip_score_upgrade': '近两周装备分提升', '2w_sjtx_time_rto': '近两周试剑天下时长占比'}
        data_df.rename(columns=dic, inplace=True)


        shap_values = np.load('data_nsh/xgb_shap_values.npy')
        shap_values = shap_values[:,need_index]
        print(shap_values.shape, data_df.shape)
        summary_plot(shap_values, data_df, max_display=data_df.shape[1], save_path='data_nsh/xgb/global_bar16_cn.pdf',
                     plot_type='bar')
        summary_plot(shap_values, data_df, max_display=data_df.shape[1], save_path='data_nsh/xgb/global16_cn.pdf')
        # # group_force_plot(explainer.expected_value, shap_values, data_df, save_path='data_nsh/xgb/balance_group.html', balance=True, sample_size=(normal_size, plug_size))
        # sample_force_plot(explainer.expected_value, role_ids, shap_values, data_df, save_path='data_nsh/xgb/balance_local/', balance=True, sample_size=(normal_size, plug_size))

    if model_type == 'lgb':
        data_df.drop(['cpu_name', 'device_id', 'device_name', 'dx_version', 'os_name', 'os_version', 'vendor_id',
                      'role_gender', 'clothes', 'money_source', 'marital_relation', 'latest_ml_chapter',
                      '45level_latest_ml_chapter',
                      '60level_latest_ml_chapter', 'latest_log_time'], axis=1, inplace=True)
        model = lgb.Booster(model_file='model_nsh/lgb_auc0.9859.model')
        model.params['objective'] = 'binary'
        explainer = shap.TreeExplainer(model)
        # start = time.clock()
        # shap_values = explainer.shap_values(data_df)
        # end = time.clock()
        # print('Explaining time is {}'.format(str(end - start)))
        # np.save('data_nsh/lgb_shap_values.npy', shap_values)
        shap_values = np.load('data_nsh/lgb_shap_values.npy')[1]
        print(shap_values.shape, data_df.shape)
        # summary_plot(shap_values, data_df, max_display=data_df.shape[1], save_path='data_nsh/lgb/global_bar.pdf',
        #              plot_type='bar')
        # summary_plot(shap_values, data_df, max_display=data_df.shape[1], save_path='data_nsh/lgb/global.pdf')
        print(explainer.expected_value)
        # group_force_plot(explainer.expected_value, shap_values, data_df, save_path='data_nsh/lgb/balance_group.html', balance=True, sample_size=(normal_size, plug_size))
        # sample_force_plot(explainer.expected_value, role_ids, shap_values, data_df, save_path='data_nsh/lgb/balance_local/', balance=True, sample_size=(normal_size, plug_size))

    if model_type == 'cbst':
        data_df.drop(['cpu_name', 'device_id', 'device_name', 'dx_version', 'os_name', 'os_version', 'vendor_id',
                      'role_gender', 'clothes', 'money_source', 'marital_relation','latest_ml_chapter',
                      '45level_latest_ml_chapter', '60level_latest_ml_chapter', 'latest_log_time'],
                     axis=1, inplace=True)
        train_data, test_data, train_label, test_label = train_test_split(data_df, label, test_size=0.2,
                                                                          random_state=42)
        model = cbst.CatBoostClassifier(iterations=1000, depth=6, learning_rate=0.5,
                                        cat_features=[],
                                        loss_function='Logloss', eval_metric='AUC', random_seed=696, reg_lambda=3,
                                        verbose=True)
        model.fit(train_data, train_label, eval_set=(test_data, test_label), early_stopping_rounds=20)
        # model.load_model('model_nsh/cbst_auc0.9834.model')
        predictions = model.predict_proba(test_data)[:, 1]
        explainer = shap.TreeExplainer(model)
        start = time.clock()
        shap_values = explainer.shap_values(data_df)
        end = time.clock()
        print('Explaining time is {}'.format(str(end - start)))
        np.save('data/cbst_shap_values.npy', shap_values)
        print(shap_values.shape, data_df.shape)
        # summary_plot(shap_values, data_df, max_display=data_df.shape[1], save_path='data_nsh/cbst/global_bar.pdf',
        #              plot_type='bar')
        # summary_plot(shap_values, data_df, max_display=data_df.shape[1], save_path='data_nsh/cbst/global.pdf')
        # print(explainer.expected_value)
        group_force_plot(explainer.expected_value, shap_values, data_df, save_path='data_nsh/cbst/balance_group.html', balance=True, sample_size=(normal_size, plug_size))
        sample_force_plot(explainer.expected_value, role_ids, shap_values, data_df, save_path='data_nsh/cbst/balance_local/', balance=True, sample_size=(normal_size, plug_size))

    if model_type == 'rf':
        data_df.drop(['cpu_name', 'device_id', 'device_name', 'dx_version', 'os_name', 'os_version', 'vendor_id',
                      'role_gender', 'clothes', 'money_source', 'marital_relation', 'latest_ml_chapter',
                      '45level_latest_ml_chapter', '60level_latest_ml_chapter', 'latest_log_time'], axis=1,
                     inplace=True)
        with open('model_nsh/rf_auc0.9764.pickle', 'rb') as f:
            model = pickle.load(f)
        explainer = shap.TreeExplainer(model)
        # start = time.clock()
        # shap_values = explainer.shap_values(data_df)[1]
        # end = time.clock()
        # print('Explaining time is {}'.format(str(end - start)))
        # np.save('data_nsh/rf_shap_values.npy', shap_values)
        # print(shap_values.shape, data_df.shape)
        shap_values = np.load('data_nsh/rf_shap_values.npy')
        # summary_plot(shap_values, data_df, max_display=data_df.shape[1], save_path='data_nsh/rf/global_bar.pdf',
        #              plot_type='bar')
        # summary_plot(shap_values, data_df, max_display=data_df.shape[1], save_path='data_nsh/rf/global.pdf')
        # print(explainer.expected_value)
        group_force_plot(explainer.expected_value[1], shap_values, data_df, save_path='data_nsh/rf/balance_group.html', balance=True, sample_size=(normal_size, plug_size))
        sample_force_plot(explainer.expected_value[1], role_ids, shap_values, data_df, save_path='data_nsh/rf/balance_local/', balance=True, sample_size=(normal_size, plug_size))

    if model_type == 'mlp':
        data_df.drop(['cpu_name', 'device_id', 'device_name', 'dx_version', 'os_name', 'os_version', 'vendor_id',
                      'role_gender', 'clothes', 'money_source', 'marital_relation', 'latest_ml_chapter',
                      '45level_latest_ml_chapter', '60level_latest_ml_chapter', 'latest_log_time'], axis=1,
                     inplace=True)
        train_data, test_data, train_label, test_label = train_test_split(data_df, label, test_size=0.2,
                                                                          random_state=42)
        bg_data = train_data.sample(n=100)
        scaler = StandardScaler().fit(train_data)
        train_data = scaler.transform(train_data)
        test_data = scaler.transform(test_data)
        all_data = scaler.transform(data_df)
        bg_data = scaler.transform(bg_data)
        inputs = Input(shape=(train_data.shape[1],))
        dense1 = Dense(64, activation='tanh')(inputs)
        dense2 = Dense(64, activation='tanh')(dense1)
        outputs = Dense(1, activation='sigmoid')(dense2)
        model = Model(inputs, outputs)
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        weight_path = 'model_nsh/mlp.hdf5'
        model.load_weights(weight_path)
        predictions = model.predict(test_data)
        explainer = shap.DeepExplainer(model, bg_data)
        # start = time.clock()
        # shap_values = explainer.shap_values(all_data)
        # end = time.clock()
        # print('Explaining time is {}'.format(str(end - start)))
        # np.save('data_nsh/mlp_shap_values.npy', shap_values)
        shap_values = np.load('data_nsh/mlp_shap_values.npy')[0]
        print(len(shap_values), all_data.shape)
        # summary_plot(shap_values, data_df, max_display=data_df.shape[1], save_path='data_nsh/mlp/global_bar.pdf',
        #              plot_type='bar')
        # summary_plot(shap_values, data_df, max_display=data_df.shape[1], save_path='data_nsh/mlp/global.pdf')
        # print(explainer.expected_value)
        group_force_plot(explainer.expected_value, shap_values, data_df, save_path='data_nsh/mlp/balance_group.html', balance=True, sample_size=(normal_size, plug_size))
        sample_force_plot(explainer.expected_value, role_ids, shap_values, data_df, save_path='data_nsh/mlp/balance_local/', balance=True, sample_size=(normal_size, plug_size))

    if model_type == 'lr':
        data_df.drop(['cpu_name', 'device_id', 'device_name', 'dx_version', 'os_name', 'os_version', 'vendor_id',
                      'role_gender', 'clothes', 'money_source', 'marital_relation', 'latest_ml_chapter',
                      '45level_latest_ml_chapter', '60level_latest_ml_chapter', 'latest_log_time'], axis=1,
                     inplace=True)
        with open('model_nsh/lr_auc0.9567.pickle', 'rb') as f:
            model = pickle.load(f)
        weight = model.coef_[0]
        columns = list(data_df.columns)
        print(len(columns), len(weight))
        with open('data_nsh/lr/weight.txt', 'w') as f:
            for idx in range(len(columns)):
                f.write(columns[idx] + '   ' + str(weight[idx]) + '\n')


def feature_selection(portrait_dir, label_dir, model_type='xgb'):
    plug_id = []
    normal_id = []
    with open(label_dir, 'r') as f:
        for line in f.readlines():
            line = line.strip().split(',')
            role_id = int(line[0])
            label = int(line[1])
            if label == 0:
                normal_id.append(role_id)
            else:
                plug_id.append(role_id)
    data_df = pd.read_csv(portrait_dir)
    normal_df = data_df[data_df['role_id'].isin(normal_id)]
    normal_size = normal_df.shape[0]
    plug_df = data_df[data_df['role_id'].isin(plug_id)]
    plug_size = plug_df.shape[0]
    data_df = pd.concat([normal_df, plug_df])
    label = np.append(np.zeros(len(normal_id)), np.ones(len(plug_id)))
    data_df['create_date'] = pd.to_datetime(data_df['ds']) - pd.to_datetime(data_df['create_date'])
    data_df['create_date'] = data_df['create_date'].dt.days
    role_ids = data_df['role_id']
    data_df.drop(['role_id', 'ds', 'role_account_name', 'role_name', 'server', 'create_time', 'phone_num', 'idn', 'dsn',
                  'mac', 'mid', 'smb', 'date', 'punish_cnt', 'ban_status', 'pk_amt', 'jiahei_cnt', 'is_forbidden',
                  'pfv_total_score', 'pfv_skill_score', 'pfv_practice_score', 'pfv_level', 'pfv_sub_level',
                  'pfv_ori_level',
                  'pfv_ori_sub_level', 'ttsw', 'qldb', '2w_d_avg_send_flower_cnt', '1st_apprentice_level',
                  '1st_apprentice_date', 'late_apprentice_level', 'late_apprentice_date', 'chushi_level', 'chushi_date',
                  '2w_bfr_baishi_d_avg_onl_tm', '1w_bfr_chushi_d_avg_onl_tm', '1w_aft_chushi_d_avg_onl_tm', 'shifu_id',
                  'couple_id', 'is_join_bhps_sj_task', '2w_d_avg_wyc_time', '2w_d_avg_cy_time', '2w_d_avg_qm_time',
                  '2w_d_avg_qm_yx_time', '2w_d_avg_jd_time', '1w_wyc_time', '1w_cy_time', '1w_qm_yx_time', '1w_jd_time',
                  '1w_fxtlg_time', '1w_wyc_yx_time', '1w_lf_yx_time', '1w_wlfyl_yx_time', '1w_jhtz_time', '1w_hsc_time',
                  'season_single_win_rate', 'season_team_win_rate', 'season_task_unfinish_rate', 'season_gua_ji_rate',
                  'season_leaving_early_rate', 'season_negatie_fight_rate', 'season_d_avg_final_score',
                  'season_d_avg_class_score',
                  'season_d_avg_strategy_score', 'season_d_avg_all_rank', 'season_d_avg_side_rank',
                  'season_d_avg_class_rank',
                  'season_d_avg_win_lose_score', 'season_d_avg_raw_score', 'season_d_avg_kill_count',
                  'season_d_avg_help_count',
                  'season_d_avg_hurted_num', 'season_d_avg_output', 'season_d_avg_cure_num', 'season_d_avg_die_cnt',
                  'season_d_avg_max_kill_player_count', 'season_d_avg_fight_strategy_score', 'season_d_avg_fight_score',
                  'season_d_avg_home_total_score', 'season_d_avg_opponent_total_score', 'season_kill_rate',
                  'season_help_rate',
                  'season_hurted_rate', 'season_output_rate', 'season_cure_rate', 'season_single_rate',
                  'season_two_rate',
                  'season_two_more_rate', 'season_team_single_count', 'season_team_more_count', '2w_single_win_rate',
                  '2w_team_win_rate', '2w_task_unfinish_rate', '2w_gua_ji_rate', '2w_leaving_early_rate',
                  '2w_negatie_fight_rate',
                  '2w_d_avg_final_score', '2w_d_avg_class_score', '2w_d_avg_strategy_score', '2w_d_avg_all_rank',
                  '2w_d_avg_side_rank',
                  '2w_d_avg_class_rank', '2w_d_avg_win_lose_score', '2w_d_avg_raw_score', '2w_d_avg_kill_count',
                  '2w_d_avg_help_count', '2w_d_avg_hurted_num', '2w_d_avg_output', '2w_d_avg_cure_num',
                  '2w_d_avg_die_cnt',
                  '2w_d_avg_max_kill_player_count', '2w_d_avg_fight_strategy_score', '2w_d_avg_fight_score',
                  '2w_d_avg_home_total_score',
                  '2w_d_avg_opponent_total_score', '2w_kill_rate', '2w_help_rate', '2w_hurted_rate', '2w_output_rate',
                  '2w_cure_rate', '2w_single_rate', '2w_two_rate', '2w_two_more_rate', '2w_team_single_count',
                  '2w_team_more_count', '3m_single_win_rate', '3m_team_win_rate', '3m_task_unfinish_rate',
                  '3m_gua_ji_rate',
                  '3m_leaving_early_rate', '3m_negatie_fight_rate', '3m_d_avg_final_score', '3m_d_avg_class_score',
                  '3m_d_avg_strategy_score', '3m_d_avg_all_rank', '3m_d_avg_side_rank', '3m_d_avg_class_rank',
                  '3m_d_avg_win_lose_score', '3m_d_avg_raw_score', '3m_d_avg_kill_count', '3m_d_avg_help_count',
                  '3m_d_avg_hurted_num', '3m_d_avg_output', '3m_d_avg_cure_num', '3m_d_avg_die_cnt',
                  '3m_d_avg_max_kill_player_count',
                  '3m_d_avg_fight_strategy_score', '3m_d_avg_fight_score', '3m_d_avg_home_total_score',
                  '3m_d_avg_opponent_total_score',
                  '3m_kill_rate', '3m_help_rate', '3m_hurted_rate', '3m_output_rate', '3m_cure_rate', '3m_single_rate',
                  '3m_two_rate', '3m_two_more_rate', '3m_team_single_count', '3m_team_more_count',
                  'is_join_feb_mtjt_bhhd',
                  'is_join_feb_yxjm_bhhd', 'is_join_bhls', 'is_join_szww', 'is_join_yq_sx', 'is_join_ss', 'is_join_sh',
                  'is_join_klxyb', 'is_join_wyc', 'is_join_fxtlg', 'is_join_wyc_yx', 'is_join_wlfyl_yx', 'is_join_yqw',
                  'is_join_qwl', 'is_join_lpzdz', 'is_join_bwdh', 'is_join_qm_yx', 'is_join_lf_yx', 'is_join_fys',
                  'is_join_hsc', 'is_join_lszd', 'is_join_ljzg', 'is_join_klxyj', '2w_d_avg_yqt_cnt',
                  '2w_d_avg_lyzsryb_cnt',
                  '2w_d_avg_wyc_die_cnt', '2w_d_avg_wyc_yx_die_cnt', '2w_d_avg_wlfyl_yx_die_cnt',
                  '2w_d_avg_tlg_dit_cnt',
                  '2w_d_avg_bounty_task_cnt', '1m_d_avg_pvp_task_rto', '1m_d_avg_pve_task_rto', '2w_d_avg_xyd_use_cnt',
                  '1m_charge_yuanbao_amt', '2w_shop_money_get_amt', 'acm_up_9_level_skill_amt', 'shop_num',
                  'latest_rare_word_num_sjwq',
                  'acm_sjbl_cnt', 'churn_friends_ratio', '2w_friends_chat_num', 'del_shitu_acm_cnt',
                  'deled_shitu_acm_cnt',
                  'shifu_log_off_day', 'acm_divorce_cnt', 'acm_chujia_cnt', 'couple_log_off_day', '2w_wyc_time_rto',
                  '2w_qmlf_yx_time_rto', '2w_cjg_die_cnt', '2w_leisure_yabiao_die_cnt', '2w_passionfight_die_cnt',
                  '2w_sjtx_die_cnt', '2w_leisure_task_die_cnt', '2w_shitu_play_cnt', 'shop_bankrupt_num'], axis=1,
                 inplace=True)
    data_df.fillna(value=0, inplace=True)
    if model_type == 'rd':
        data_df.drop(['cpu_name', 'device_id', 'device_name', 'dx_version', 'os_name', 'os_version', 'vendor_id',
                      'role_gender', 'clothes', 'money_source', 'marital_relation', 'latest_ml_chapter',
                      '45level_latest_ml_chapter',
                      '60level_latest_ml_chapter', 'latest_log_time'], axis=1, inplace=True)
        col_list = np.array(data_df.columns)
        p = np.random.permutation(len(col_list))
        n = [20, 40, 60, 80, 100]
        for x in n:
            p_tmp = p[:x]
            # print(col_list, p_tmp)
            col_list_tmp = col_list[p_tmp]
            data_df_tmp = data_df[col_list_tmp]
            train_data, test_data, train_label, test_label = train_test_split(data_df_tmp, label, test_size=0.2,
                                                                              random_state=42)
            dtrain = xgb.DMatrix(train_data, label=train_label)
            dtest = xgb.DMatrix(test_data, label=test_label)
            params = {
                'booster': 'gbtree',
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'max_depth': 7,
                'lambda': 1,
                'subsample': 0.75,
                'colsample_bytree': 0.75,
                'min_child_weight': 2,
                'eta': 0.025,
                'seed': 0,
                'nthread': 8,
                'silent': 1
            }
            watchlist = [(dtest, 'validation')]
            start = time.clock()
            model = xgb.train(params, dtrain, num_boost_round=1000, early_stopping_rounds=30, evals=watchlist)
            mid = time.clock()
            predictions = model.predict(dtest)
            end = time.clock()
            # print('Training time is {}'.format(str(mid - start)))
            # print('Inference time is {}'.format(str(end - mid)))
            auc = roc_auc_score(test_label, predictions)
            print(x, ' features')
            print('The roc of prediction is {}'.format(auc))
            # model.save_model('model_nsh/xgb_auc{:.4f}.model'.format(auc))
            pred_norm = [round(score) for score in predictions]
            print('The acc of prediction is {}'.format(accuracy_score(test_label, pred_norm)))
            print('The precision of prediction is {}'.format(precision_score(test_label, pred_norm)))
            print('The recall of prediction is {}'.format(recall_score(test_label, pred_norm)))
            print('The F1 score of prediction is {}'.format(f1_score(test_label, pred_norm)))

    if model_type == 'xgb':
        data_df.drop(['cpu_name', 'device_id', 'device_name', 'dx_version', 'os_name', 'os_version', 'vendor_id',
                      'role_gender', 'clothes', 'money_source', 'marital_relation', 'latest_ml_chapter',
                      '45level_latest_ml_chapter',
                      '60level_latest_ml_chapter', 'latest_log_time'], axis=1, inplace=True)
        ddata = xgb.DMatrix(data_df)
        model = xgb.Booster(model_file='model_nsh/xgb_auc0.9876.model')
        predictions = model.predict(ddata)
        explainer = shap.TreeExplainer(model)
        # start = time.clock()
        # shap_values = explainer.shap_values(ddata)
        # end = time.clock()
        # print('Explaining time is {}'.format(str(end - start)))
        # np.save('data_nsh/xgb_shap_values.npy', shap_values)
        shap_values = np.load('data_nsh/xgb_shap_values.npy')
        print(shap_values.shape, data_df.shape)
        feature_inds = shap.summary_plot(shap_values, data_df, max_display=data_df.shape[1], plot_type='bar', show=False)
        col_list = np.array(data_df.columns)
        col_rank = col_list[feature_inds[::-1]]
        n = [20, 40, 60, 80, 100]
        for x in n:
            col_list_tmp = col_rank[:x]
            # print(col_list, p_tmp)
            data_df_tmp = data_df[col_list_tmp]
            train_data, test_data, train_label, test_label = train_test_split(data_df_tmp, label, test_size=0.2,
                                                                              random_state=42)
            dtrain = xgb.DMatrix(train_data, label=train_label)
            dtest = xgb.DMatrix(test_data, label=test_label)
            params = {
                'booster': 'gbtree',
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'max_depth': 7,
                'lambda': 1,
                'subsample': 0.75,
                'colsample_bytree': 0.75,
                'min_child_weight': 2,
                'eta': 0.025,
                'seed': 0,
                'nthread': 8,
                'silent': 1
            }
            watchlist = [(dtest, 'validation')]
            start = time.clock()
            model = xgb.train(params, dtrain, num_boost_round=1000, early_stopping_rounds=30, evals=watchlist)
            mid = time.clock()
            predictions = model.predict(dtest)
            # print('Training time is {}'.format(str(mid - start)))
            # print('Inference time is {}'.format(str(end - mid)))
            auc = roc_auc_score(test_label, predictions)
            print(x, ' features')
            print('The roc of prediction is {}'.format(auc))
            # model.save_model('model_nsh/xgb_auc{:.4f}.model'.format(auc))
            pred_norm = [round(score) for score in predictions]
            print('The acc of prediction is {}'.format(accuracy_score(test_label, pred_norm)))
            print('The precision of prediction is {}'.format(precision_score(test_label, pred_norm)))
            print('The recall of prediction is {}'.format(recall_score(test_label, pred_norm)))
            print('The F1 score of prediction is {}'.format(f1_score(test_label, pred_norm)))

    if model_type == 'lgb':
        data_df.drop(['cpu_name', 'device_id', 'device_name', 'dx_version', 'os_name', 'os_version', 'vendor_id',
                      'role_gender', 'clothes', 'money_source', 'marital_relation', 'latest_ml_chapter',
                      '45level_latest_ml_chapter',
                      '60level_latest_ml_chapter', 'latest_log_time'], axis=1, inplace=True)
        model = lgb.Booster(model_file='model_nsh/lgb_auc0.9859.model')
        model.params['objective'] = 'binary'
        explainer = shap.TreeExplainer(model)
        # start = time.clock()
        # shap_values = explainer.shap_values(data_df)
        # end = time.clock()
        # print('Explaining time is {}'.format(str(end - start)))
        # np.save('data_nsh/lgb_shap_values.npy', shap_values)
        shap_values = np.load('data_nsh/lgb_shap_values.npy')[1]
        print(shap_values.shape, data_df.shape)
        feature_inds = shap.summary_plot(shap_values, data_df, max_display=data_df.shape[1], plot_type='bar',
                                         show=False)
        col_list = np.array(data_df.columns)
        col_rank = col_list[feature_inds[::-1]]
        n = [20, 40, 60, 80, 100]
        for x in n:
            col_list_tmp = col_rank[:x]
            # print(col_list, p_tmp)
            data_df_tmp = data_df[col_list_tmp]
            train_data, test_data, train_label, test_label = train_test_split(data_df_tmp, label, test_size=0.2,
                                                                              random_state=42)
            dtrain = xgb.DMatrix(train_data, label=train_label)
            dtest = xgb.DMatrix(test_data, label=test_label)
            params = {
                'booster': 'gbtree',
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'max_depth': 7,
                'lambda': 1,
                'subsample': 0.75,
                'colsample_bytree': 0.75,
                'min_child_weight': 2,
                'eta': 0.025,
                'seed': 0,
                'nthread': 8,
                'silent': 1
            }
            watchlist = [(dtest, 'validation')]
            start = time.clock()
            model = xgb.train(params, dtrain, num_boost_round=1000, early_stopping_rounds=30, evals=watchlist)
            mid = time.clock()
            predictions = model.predict(dtest)
            end = time.clock()
            # print('Training time is {}'.format(str(mid - start)))
            # print('Inference time is {}'.format(str(end - mid)))
            auc = roc_auc_score(test_label, predictions)
            print(x, ' features')
            print('The roc of prediction is {}'.format(auc))
            # model.save_model('model_nsh/xgb_auc{:.4f}.model'.format(auc))
            pred_norm = [round(score) for score in predictions]
            print('The acc of prediction is {}'.format(accuracy_score(test_label, pred_norm)))
            print('The precision of prediction is {}'.format(precision_score(test_label, pred_norm)))
            print('The recall of prediction is {}'.format(recall_score(test_label, pred_norm)))
            print('The F1 score of prediction is {}'.format(f1_score(test_label, pred_norm)))

    if model_type == 'cbst':
        data_df.drop(['cpu_name', 'device_id', 'device_name', 'dx_version', 'os_name', 'os_version', 'vendor_id',
                      'role_gender', 'clothes', 'money_source', 'marital_relation', 'latest_ml_chapter',
                      '45level_latest_ml_chapter', '60level_latest_ml_chapter', 'latest_log_time'],
                     axis=1, inplace=True)
        train_data, test_data, train_label, test_label = train_test_split(data_df, label, test_size=0.2,
                                                                          random_state=42)
        model = cbst.CatBoostClassifier(iterations=1000, depth=6, learning_rate=0.5,
                                        cat_features=[],
                                        loss_function='Logloss', eval_metric='AUC', random_seed=696, reg_lambda=3,
                                        verbose=True)
        model.fit(train_data, train_label, eval_set=(test_data, test_label), early_stopping_rounds=20)
        # model.load_model('model_nsh/cbst_auc0.9834.model')
        predictions = model.predict_proba(test_data)[:, 1]
        explainer = shap.TreeExplainer(model)
        # start = time.clock()
        shap_values = explainer.shap_values(data_df)
        # end = time.clock()
        # print('Explaining time is {}'.format(str(end - start)))
        # np.save('data/cbst_shap_values.npy', shap_values)
        print(shap_values.shape, data_df.shape)
        feature_inds = shap.summary_plot(shap_values, data_df, max_display=data_df.shape[1], plot_type='bar',
                                         show=False)
        col_list = np.array(data_df.columns)
        col_rank = col_list[feature_inds[::-1]]
        n = [20, 40, 60, 80, 100]
        for x in n:
            col_list_tmp = col_rank[:x]
            # print(col_list, p_tmp)
            data_df_tmp = data_df[col_list_tmp]
            train_data, test_data, train_label, test_label = train_test_split(data_df_tmp, label, test_size=0.2,
                                                                              random_state=42)
            dtrain = xgb.DMatrix(train_data, label=train_label)
            dtest = xgb.DMatrix(test_data, label=test_label)
            params = {
                'booster': 'gbtree',
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'max_depth': 7,
                'lambda': 1,
                'subsample': 0.75,
                'colsample_bytree': 0.75,
                'min_child_weight': 2,
                'eta': 0.025,
                'seed': 0,
                'nthread': 8,
                'silent': 1
            }
            watchlist = [(dtest, 'validation')]
            start = time.clock()
            model = xgb.train(params, dtrain, num_boost_round=1000, early_stopping_rounds=30, evals=watchlist)
            mid = time.clock()
            predictions = model.predict(dtest)
            end = time.clock()
            # print('Training time is {}'.format(str(mid - start)))
            # print('Inference time is {}'.format(str(end - mid)))
            auc = roc_auc_score(test_label, predictions)
            print(x, ' features')
            print('The roc of prediction is {}'.format(auc))
            # model.save_model('model_nsh/xgb_auc{:.4f}.model'.format(auc))
            pred_norm = [round(score) for score in predictions]
            print('The acc of prediction is {}'.format(accuracy_score(test_label, pred_norm)))
            print('The precision of prediction is {}'.format(precision_score(test_label, pred_norm)))
            print('The recall of prediction is {}'.format(recall_score(test_label, pred_norm)))
            print('The F1 score of prediction is {}'.format(f1_score(test_label, pred_norm)))

    if model_type == 'rf':
        data_df.drop(['cpu_name', 'device_id', 'device_name', 'dx_version', 'os_name', 'os_version', 'vendor_id',
                      'role_gender', 'clothes', 'money_source', 'marital_relation', 'latest_ml_chapter',
                      '45level_latest_ml_chapter', '60level_latest_ml_chapter', 'latest_log_time'], axis=1,
                     inplace=True)
        with open('model_nsh/rf_auc0.9764.pickle', 'rb') as f:
            model = pickle.load(f)
        explainer = shap.TreeExplainer(model)
        # start = time.clock()
        # shap_values = explainer.shap_values(data_df)[1]
        # end = time.clock()
        # print('Explaining time is {}'.format(str(end - start)))
        # np.save('data_nsh/rf_shap_values.npy', shap_values)
        # print(shap_values.shape, data_df.shape)
        shap_values = np.load('data_nsh/rf_shap_values.npy')
        feature_inds = shap.summary_plot(shap_values, data_df, max_display=data_df.shape[1], plot_type='bar',
                                         show=False)
        col_list = np.array(data_df.columns)
        col_rank = col_list[feature_inds[::-1]]
        n = [20, 40, 60, 80, 100]
        for x in n:
            col_list_tmp = col_rank[:x]
            # print(col_list, p_tmp)
            data_df_tmp = data_df[col_list_tmp]
            train_data, test_data, train_label, test_label = train_test_split(data_df_tmp, label, test_size=0.2,
                                                                              random_state=42)
            dtrain = xgb.DMatrix(train_data, label=train_label)
            dtest = xgb.DMatrix(test_data, label=test_label)
            params = {
                'booster': 'gbtree',
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'max_depth': 7,
                'lambda': 1,
                'subsample': 0.75,
                'colsample_bytree': 0.75,
                'min_child_weight': 2,
                'eta': 0.025,
                'seed': 0,
                'nthread': 8,
                'silent': 1
            }
            watchlist = [(dtest, 'validation')]
            start = time.clock()
            model = xgb.train(params, dtrain, num_boost_round=1000, early_stopping_rounds=30, evals=watchlist)
            mid = time.clock()
            predictions = model.predict(dtest)
            end = time.clock()
            # print('Training time is {}'.format(str(mid - start)))
            # print('Inference time is {}'.format(str(end - mid)))
            auc = roc_auc_score(test_label, predictions)
            print(x, ' features')
            print('The roc of prediction is {}'.format(auc))
            # model.save_model('model_nsh/xgb_auc{:.4f}.model'.format(auc))
            pred_norm = [round(score) for score in predictions]
            print('The acc of prediction is {}'.format(accuracy_score(test_label, pred_norm)))
            print('The precision of prediction is {}'.format(precision_score(test_label, pred_norm)))
            print('The recall of prediction is {}'.format(recall_score(test_label, pred_norm)))
            print('The F1 score of prediction is {}'.format(f1_score(test_label, pred_norm)))

    if model_type == 'mlp':
        data_df.drop(['cpu_name', 'device_id', 'device_name', 'dx_version', 'os_name', 'os_version', 'vendor_id',
                      'role_gender', 'clothes', 'money_source', 'marital_relation', 'latest_ml_chapter',
                      '45level_latest_ml_chapter', '60level_latest_ml_chapter', 'latest_log_time'], axis=1,
                     inplace=True)
        shap_values = np.load('data_nsh/mlp_shap_values.npy')[0]
        feature_inds = shap.summary_plot(shap_values, data_df, max_display=data_df.shape[1], plot_type='bar',
                                         show=False)
        col_list = np.array(data_df.columns)
        col_rank = col_list[feature_inds[::-1]]
        n = [20, 40, 60, 80, 100]
        for x in n:
            col_list_tmp = col_rank[:x]
            # print(col_list, p_tmp)
            data_df_tmp = data_df[col_list_tmp]
            train_data, test_data, train_label, test_label = train_test_split(data_df_tmp, label, test_size=0.2,
                                                                              random_state=42)
            dtrain = xgb.DMatrix(train_data, label=train_label)
            dtest = xgb.DMatrix(test_data, label=test_label)
            params = {
                'booster': 'gbtree',
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'max_depth': 7,
                'lambda': 1,
                'subsample': 0.75,
                'colsample_bytree': 0.75,
                'min_child_weight': 2,
                'eta': 0.025,
                'seed': 0,
                'nthread': 8,
                'silent': 1
            }
            watchlist = [(dtest, 'validation')]
            start = time.clock()
            model = xgb.train(params, dtrain, num_boost_round=1000, early_stopping_rounds=30, evals=watchlist)
            mid = time.clock()
            predictions = model.predict(dtest)
            end = time.clock()
            # print('Training time is {}'.format(str(mid - start)))
            # print('Inference time is {}'.format(str(end - mid)))
            auc = roc_auc_score(test_label, predictions)
            print(x, ' features')
            print('The roc of prediction is {}'.format(auc))
            # model.save_model('model_nsh/xgb_auc{:.4f}.model'.format(auc))
            pred_norm = [round(score) for score in predictions]
            print('The acc of prediction is {}'.format(accuracy_score(test_label, pred_norm)))
            print('The precision of prediction is {}'.format(precision_score(test_label, pred_norm)))
            print('The recall of prediction is {}'.format(recall_score(test_label, pred_norm)))
            print('The F1 score of prediction is {}'.format(f1_score(test_label, pred_norm)))

    if model_type == 'lr':
        data_df.drop(['cpu_name', 'device_id', 'device_name', 'dx_version', 'os_name', 'os_version', 'vendor_id',
                      'role_gender', 'clothes', 'money_source', 'marital_relation', 'latest_ml_chapter',
                      '45level_latest_ml_chapter', '60level_latest_ml_chapter', 'latest_log_time'], axis=1,
                     inplace=True)
        with open('model_nsh/lr_auc0.9567.pickle', 'rb') as f:
            model = pickle.load(f)
        weight = np.abs(model.coef_[0])
        col_list = np.array(data_df.columns)
        p = np.argsort(-weight)
        col_rank = col_list[p]
        n = [20, 40, 60, 80, 100]
        for x in n:
            col_list_tmp = col_rank[:x]
            data_df_tmp = data_df[col_list_tmp]
            train_data, test_data, train_label, test_label = train_test_split(data_df_tmp, label, test_size=0.2,
                                                                              random_state=42)
            dtrain = xgb.DMatrix(train_data, label=train_label)
            dtest = xgb.DMatrix(test_data, label=test_label)
            params = {
                'booster': 'gbtree',
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'max_depth': 7,
                'lambda': 1,
                'subsample': 0.75,
                'colsample_bytree': 0.75,
                'min_child_weight': 2,
                'eta': 0.025,
                'seed': 0,
                'nthread': 8,
                'silent': 1
            }
            watchlist = [(dtest, 'validation')]
            start = time.clock()
            model = xgb.train(params, dtrain, num_boost_round=1000, early_stopping_rounds=30, evals=watchlist)
            mid = time.clock()
            predictions = model.predict(dtest)
            end = time.clock()
            # print('Training time is {}'.format(str(mid - start)))
            # print('Inference time is {}'.format(str(end - mid)))
            auc = roc_auc_score(test_label, predictions)
            print(x, ' features')
            print('The roc of prediction is {}'.format(auc))
            # model.save_model('model_nsh/xgb_auc{:.4f}.model'.format(auc))
            pred_norm = [round(score) for score in predictions]
            print('The acc of prediction is {}'.format(accuracy_score(test_label, pred_norm)))
            print('The precision of prediction is {}'.format(precision_score(test_label, pred_norm)))
            print('The recall of prediction is {}'.format(recall_score(test_label, pred_norm)))
            print('The F1 score of prediction is {}'.format(f1_score(test_label, pred_norm)))


if __name__ == '__main__':
    LABEL_DIR = '/project/fumo/nsh/xai/label'
    PORTRAIT_DIR = '/project/fumo/nsh/xai/portrait.csv'
    # train_model(PORTRAIT_DIR, LABEL_DIR, model_type='lr')
    explain_pic(PORTRAIT_DIR, LABEL_DIR, model_type='xgb')
    # explain_pic(PORTRAIT_DIR, LABEL_DIR, model_type='lgb')
    # explain_pic(PORTRAIT_DIR, LABEL_DIR, model_type='cbst')
    # explain_pic(PORTRAIT_DIR, LABEL_DIR, model_type='rf')
    # explain_pic(PORTRAIT_DIR, LABEL_DIR, model_type='mlp')
    # explain_pic(PORTRAIT_DIR, LABEL_DIR, model_type='lr')
    # feature_selection(PORTRAIT_DIR, LABEL_DIR, model_type='lr')




