import os
import kuti
import torch
from kuti import applications as apps
from kuti import model_helper as mh
from kuti import tensor_ops as ops
from kuti import image_utils
from kuti import generic
import pandas as pd, numpy as np

# MODEL DEF
from keras.layers import Input, GlobalAveragePooling2D
from keras.models import Model


def prepare_dataframe(datadict):
    split_mapping = {'train': 'training', 'validation': 'validation', 'test': 'test'}

    dataset = []
    for s, d in datadict.items():
        tmp = pd.DataFrame(d.dataset)
        tmp['set'] = [split_mapping[s]] * len(tmp.index)
        dataset.append(tmp)

    dataset = pd.concat(dataset, ignore_index=True)
    dataset = dataset.rename(columns={'im_name': 'image_name', 'im_paths': 'image_name'})
    dataset = dataset.rename(columns={'mean_score': 'MOS'})
    return dataset


def extract_features(data_dir, ids, features_path):
    input_shape = (None, None, 3)
    model = apps.model_inceptionresnet_pooled(input_shape)
    pre   = apps.process_input[apps.InceptionResNetV2]
    model_name = 'irnv2_mlsp_wide'
    gen_params = dict(batch_size  = 1,
                  data_path   = data_dir,
                  input_shape = ('orig',),
                  inputs = ['image_name',],
                  process_fn  = pre,
                  fixed_batches = False)

    helper = mh.ModelHelper(model, model_name + '_orig', ids,
                            features_root = features_path,
                            gen_params    = gen_params)

    print('Saving features')
    batch_size = 1024
    numel = len(ids)
    for i in range(0, numel, batch_size):
        istop = min(i+batch_size, numel)
        print('Processing images', i, ':', istop)
        ids_batch = ids[i:istop].reset_index(drop=True)
        helper.save_activations(ids=ids_batch, verbose=True,\
                                save_as_type=np.float16)


def helper_init(dataset_name, ckpt_path, features_file, ids):
    fc1_size = 2048
    image_size = '[orig]'
    input_size = (5,5,16928)
    model_name = features_file.split('/')[-2]
    loss = 'MSE'
    bn = 2
    fc_sizes = [fc1_size, fc1_size/2, fc1_size/8,  1]
    dropout_rates = [0.25, 0.25, 0.5, 0]

    monitor_metric = 'val_plcc_tf'; monitor_mode = 'max'
    metrics = ["MAE", ops.plcc_tf]
    outputs = 'MOS'

    input_feats = Input(shape=input_size, dtype='float32')

    # SINGLE-block
    x = apps.inception_block(input_feats, size=1024)
    x = GlobalAveragePooling2D(name='final_GAP')(x)

    pred = apps.fc_layers(x,
                          name  = 'head',
                          fc_sizes      = fc_sizes,
                          dropout_rates = dropout_rates,
                          batch_norm    = bn)

    model = Model(inputs=input_feats, outputs=pred)

    gen_params = dict(batch_size    = 128,
                      data_path     = features_file,                  
                      input_shape   = input_size,
                      inputs        = ['image_name'],
                      outputs       = [outputs], 
                      random_group  = False,
                      fixed_batches = True)

    helper = mh.ModelHelper(model, model_name, ids, 
                            max_queue_size = 128,
                            loss           = loss,
                            metrics        = metrics,
                            monitor_metric = monitor_metric, 
                            monitor_mode   = monitor_mode,
                            multiproc      = False, workers = 1,
        #                     multiproc      = True, workers = 3,
                            early_stop_patience = 5,
                            logs_root      = ckpt_path + 'logs',
                            models_root    = ckpt_path + 'models',
                            gen_params     = gen_params)

    helper.model_name.update(fc1 = '[%d]' % fc1_size, 
                             im  = image_size,
                             bn  = bn,
                             do  = str(dropout_rates).replace(' ',''),
                             mon = '[%s]' % monitor_metric,
                             ds  = dataset_name)

    print(helper.model_name())
    return helper


def train(dataset_name, dataset, batch_size=64, epochs=5, early_stopping_patience=10):
    ckpt_path = 'ckpts/MLSP/%s' % dataset_name
    features_path = ckpt_path + '/features/'
    ids = prepare_dataframe(dataset)
    data_dir = str(dataset['train'].image_folder)

    if (not os.path.exists(features_path)) or (len(os.listdir(features_path)) == 0):
        extract_features(data_dir, ids, features_path)
    else:
        print("Found features in directory: ", features_path)

    features_file = ckpt_path + '/features/irnv2_mlsp_wide_orig/grp:1 i:1[orig] lay:final o:1[5,5,16928].h5'
    helper = helper_init(dataset_name, ckpt_path, features_file, ids)
    for lr in [1e-4, 1e-5, 1e-6]:
        helper.load_model()
        helper.train(lr=lr, epochs=20)



def evaluate(dataset_name, dataset):
    ckpt_path = 'ckpts/MLSP/%s' % dataset_name
    model_name = 'irnv2_mlsp_wide_orig/bn:2 bsz:128 do:[0.25,0.25,0.5,0] ds:PCCD fc1:[2048] i:1[5,5,16928] im:[orig] l:MSE mon:[val_plcc_tf] o:1[1]'

    ids = pd.DataFrame(dataset.dataset)
    ids['set'] = ['test'] * len(ids.index)
    ids = ids.rename(columns={'im_name': 'image_name'})
    ids = ids.rename(columns={'mean_score': 'MOS'})

    features_file = ckpt_path + '/features/irnv2_mlsp_wide_orig/grp:1 i:1[orig] lay:final o:1[5,5,16928].h5'
    helper = helper_init(dataset_name, ckpt_path, features_file, ids)
    if helper.load_model(model_name=model_name):
        y_test, y_pred, SRCC_test, PLCC_test = apps.test_rating_model(helper)

    torch.save({'gt': y_test, 'pred': y_pred},
                ckpt_path+"/predictions.pth")