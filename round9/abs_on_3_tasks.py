# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

for_submission = True
import os

import datasets
import numpy as np
import torch
import torch.nn.functional as F
import transformers
import json
import jsonschema
import jsonpickle
import random
import csv
import pickle
import time

import scipy.stats
import scipy.spatial
import scipy.special
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.preprocessing import normalize
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
import sklearn.metrics
np.set_printoptions(precision=2)


import warnings

from abs_pytorch_r9_1_1_4_1_24 import sc_trojan_detector
# from abs_pytorch_r9_1_1_4_2_3_16 import ner_trojan_detector
from abs_pytorch_r9_1_1_4_2_24 import ner_trojan_detector
from abs_pytorch_r9_1_1_4_3_3_13_5 import qa_trojan_detector

warnings.filterwarnings("ignore")

if not for_submission:
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"

random_seed = 333
torch.backends.cudnn.enabled = False
# deterministic
torch.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)

def defer_output(output, model_type):

    # output = np.clip(output, 0.115, 0.885)

    ten_digit = np.floor(output * 10)

    if model_type == 'ElectraForSequenceClassification':
        new_output = ten_digit/10. + 0.015
    elif model_type == 'DistilBertForSequenceClassification':
        new_output = ten_digit/10. + 0.025
    elif model_type == 'RobertaForSequenceClassification':
        new_output = ten_digit/10. + 0.035
    elif model_type == 'ElectraForTokenClassification':
        new_output = ten_digit/10. + 0.045
    elif model_type == 'DistilBertForTokenClassification':
        new_output = ten_digit/10. + 0.055
    elif model_type == 'RobertaForTokenClassification':
        new_output = ten_digit/10. + 0.065
    elif model_type == 'ElectraForQuestionAnswering':
        new_output = ten_digit/10. + 0.075
    elif model_type == 'DistilBertForQuestionAnswering':
        new_output = ten_digit/10. + 0.085
    elif model_type == 'RobertaForQuestionAnswering':
        new_output = ten_digit/10. + 0.095
    else:
        print('error model type unseen', model_type, output)
        new_output = output

    print('deter output', output, new_output)

    return new_output


def example_trojan_detector(model_filepath,
                            tokenizer_filepath,
                            result_filepath,
                            scratch_dirpath,
                            examples_dirpath,
                            round_training_dataset_dirpath,
                            learned_parameters_dirpath,
                            features_filepath, 
                            parameters):
    print('model_filepath = {}'.format(model_filepath))
    print('tokenizer_filepath = {}'.format(tokenizer_filepath))
    print('result_filepath = {}'.format(result_filepath))
    print('scratch_dirpath = {}'.format(scratch_dirpath))
    print('examples_dirpath = {}'.format(examples_dirpath))
    print('round_training_dataset_dirpath = {}'.format(round_training_dataset_dirpath))
    print('features_filepath = {}'.format(features_filepath))

    print('Using learned parameters_dirpath = {}'.format(learned_parameters_dirpath))
    os.system('mkdir -p '+scratch_dirpath)

    start_time = time.time()

    model = torch.load(model_filepath, map_location=torch.device('cpu'))
    model_type = model.__class__.__name__
    print('model type', model_type, )
    # sys.exit()
    
    is_configure = False

    print('parameters', parameters)

    if model_type.endswith('SequenceClassification'):
        # # TODO maunally set the examples dir
        # if not for_submission:
        #     examples_dirpath = './sc_clean_example_data/'
        # else:
        #     examples_dirpath = '/sc_clean_example_data/'
        print('examples_dirpath = {}'.format(examples_dirpath))
        sc_parameters = [for_submission, is_configure]
        sc_parameters += parameters[:3]
        output, features = sc_trojan_detector(model_filepath, tokenizer_filepath, result_filepath, scratch_dirpath, examples_dirpath, round_training_dataset_dirpath, learned_parameters_dirpath, features_filepath, sc_parameters)[:2]
        print('features', features)
        fields = ['emb_id'] + ['sc_asr_'+str(_) for _ in range(len(features)-1)]
    elif model_type.endswith('TokenClassification'): 
        # # TODO maunally set the examples dir
        # if not for_submission:
        #     examples_dirpath = './ner_clean_example_data/'
        # else:
        #     examples_dirpath = '/ner_clean_example_data/'
        print('examples_dirpath = {}'.format(examples_dirpath))
        ner_parameters = [for_submission, is_configure]
        ner_parameters += parameters[3:6]
        output, features = ner_trojan_detector(model_filepath, tokenizer_filepath, result_filepath, scratch_dirpath, examples_dirpath, round_training_dataset_dirpath, learned_parameters_dirpath, features_filepath, ner_parameters)[:2]
        fields = ['emb_id'] + ['ner_asr_'+str(_) for _ in range(len(features)-1)]
    elif model_type.endswith('QuestionAnswering'): 
        # # TODO maunally set the examples dir
        # if not for_submission:
        #     examples_dirpath = './qa_clean_example_data/'
        # else:
        #     examples_dirpath = '/qa_clean_example_data/'
        print('examples_dirpath = {}'.format(examples_dirpath))
        qa_parameters = [for_submission, is_configure]
        qa_parameters += parameters[6:9]
        output, features = qa_trojan_detector(model_filepath, tokenizer_filepath, result_filepath, scratch_dirpath, examples_dirpath, round_training_dataset_dirpath, learned_parameters_dirpath, features_filepath, qa_parameters)[:2]
        fields = ['emb_id'] + ['qa_asr_'+str(_) for _ in range(len(features)-1)]

    end_time = time.time()
    print('model type', model_type, 'time', end_time - start_time)

    # output differetidifferetiation
    output = defer_output(output, model_type)

    with open(features_filepath, 'w') as csvfile: 
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(fields) 
        csvwriter.writerow(features) 

    with open(result_filepath, 'w') as fh:
        fh.write("{}".format(output))

    if not for_submission:
        with open(os.path.join(scratch_dirpath,'result.txt'), 'a') as fh:
            fh.write("{} {}\n".format(model_filepath, output))

        mname = model_filepath.split('/')[-2]
        with open(os.path.join(scratch_dirpath,'features_v2_{0}.csv'.format(mname)), 'a') as csvfile: 
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(fields) 
            csvwriter.writerow(features) 


def feature_extractor(mname, output_parameters_dirpath, configure_models_dirpath, scratch_dirpath, parameters):

    model_filepath = os.path.join(configure_models_dirpath, 'models', mname, 'model.pt')

    examples_dirpath = os.path.join(configure_models_dirpath, 'models', mname, )
    # examples_dirpath = '/data/share/trojai/trojai-round8-v1-dataset/models/id-00000007/example_data/'

    round_training_dataset_dirpath = configure_models_dirpath
    result_filepath = os.path.join(scratch_dirpath, mname+'_output.txt')
    features_filepath = os.path.join(scratch_dirpath, mname+'_features.csv')
    learned_parameters_dirpath = output_parameters_dirpath

    print('model_filepath = {}'.format(model_filepath))
    print('result_filepath = {}'.format(result_filepath))
    print('scratch_dirpath = {}'.format(scratch_dirpath))
    print('round_training_dataset_dirpath = {}'.format(round_training_dataset_dirpath))
    print('features_filepath = {}'.format(features_filepath))

    print('Using learned parameters_dirpath = {}'.format(learned_parameters_dirpath))
    os.system('mkdir -p '+scratch_dirpath)

    is_configure = True

    model = torch.load(model_filepath, map_location=torch.device('cpu'))
    model_type = model.__class__.__name__
    print('model type', model_type, )

    print('parameters', parameters)

    if model_type.startswith('Electra'):
        tokenizer_filepath = '{0}/tokenizers/google-electra-small-discriminator.pt'.format(configure_models_dirpath)
    elif model_type.startswith('Roberta'):
        tokenizer_filepath = '{0}/tokenizers/roberta-base.pt'.format(configure_models_dirpath)
    elif model_type.startswith('DistilBert'):
        tokenizer_filepath = '{0}/tokenizers/distilbert-base-cased.pt'.format(configure_models_dirpath)

    print('tokenizer_filepath = {}'.format(tokenizer_filepath))

    if model_type.endswith('SequenceClassification'):
        # # TODO maunally set the examples dir
        # if not for_submission:
        #     examples_dirpath = './sc_clean_example_data/'
        # else:
        #     examples_dirpath = '/sc_clean_example_data/'
        sc_parameters = [for_submission, is_configure]
        sc_parameters += parameters[:3]
        output, features, roberta_features = sc_trojan_detector(model_filepath, tokenizer_filepath, result_filepath, scratch_dirpath, examples_dirpath, round_training_dataset_dirpath, learned_parameters_dirpath, features_filepath, sc_parameters)
        fields = ['emb_id'] + ['sc_asr_'+str(_) for _ in range(len(features)-1)]
    elif model_type.endswith('TokenClassification'): 
        # # TODO maunally set the examples dir
        # if not for_submission:
        #     examples_dirpath = './ner_clean_example_data/'
        # else:
        #     examples_dirpath = '/ner_clean_example_data/'
        sc_parameters = [for_submission, is_configure]
        sc_parameters += parameters[3:6]
        output, features, roberta_features = ner_trojan_detector(model_filepath, tokenizer_filepath, result_filepath, scratch_dirpath, examples_dirpath, round_training_dataset_dirpath, learned_parameters_dirpath, features_filepath, sc_parameters)
        fields = ['emb_id'] + ['ner_asr_'+str(_) for _ in range(len(features)-1)]
    elif model_type.endswith('QuestionAnswering'): 
        # # TODO maunally set the examples dir
        # if not for_submission:
        #     examples_dirpath = './qa_clean_example_data/'
        # else:
        #     examples_dirpath = '/qa_clean_example_data/'
        sc_parameters = [for_submission, is_configure]
        sc_parameters += parameters[6:9]
        output, features, roberta_features = qa_trojan_detector(model_filepath, tokenizer_filepath, result_filepath, scratch_dirpath, examples_dirpath, round_training_dataset_dirpath, learned_parameters_dirpath, features_filepath, sc_parameters)
        fields = ['emb_id'] + ['qa_asr_'+str(_) for _ in range(len(features)-1)]

    with open(features_filepath, 'w') as csvfile: 
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(fields) 
        csvwriter.writerow(features) 

    with open(result_filepath, 'w') as fh:
        fh.write("{}".format(output))

    return features, roberta_features, model_type


def train_bounds(train_X, train_y, test_X, test_y, signs):
    full_bounds = []
    for arch_i in range(3):
        bounds = [[] for _ in range(train_X.shape[1]-1)]
        bvs = [[] for _ in range(train_X.shape[1]-1)]
        tvs = [[] for _ in range(train_X.shape[1]-1)]
        for i in range(train_y.shape[0]):
            if train_X[i,0] == arch_i and train_y[i] == 0:
                for j in range(len(bvs)):
                    bvs[j].append(train_X[i,j+1])
            elif train_X[i,0] == arch_i and train_y[i] == 1:
                for j in range(len(tvs)):
                    tvs[j].append(train_X[i,j+1])

        if len(bvs[0]) == 0 or len(tvs[0]) == 0:
            full_bounds.append([])
            continue

        for i in range(len(bounds)):
            sign = signs[i]
            if sign:
                larger_tvs = []
                for j in range(len(tvs[i])):
                    if tvs[i][j] > np.amax(bvs[i]):
                        larger_tvs.append(tvs[i][j])
                if len(larger_tvs) > 0:
                    lowest_larger_tvs = np.amin(larger_tvs)
                else:
                    lowest_larger_tvs = np.amax(bvs[i]) + abs(np.amax(bvs[i]) - np.amax(tvs[i]))
                bounds[i] = [sign, (lowest_larger_tvs + np.amax(bvs[i])) / 2.]
            else:
                lower_tvs = []
                for j in range(len(tvs[i])):
                    if tvs[i][j] < np.amin(bvs[i]):
                        lower_tvs.append(tvs[i][j])
                if len(lower_tvs) > 0:
                    highest_lower_tvs = np.amax(lower_tvs)
                else:
                    highest_lower_tvs = np.amin(bvs[i]) - abs(np.amin(bvs[i]) - np.amin(tvs[i]))
                bounds[i] = [sign, (highest_lower_tvs + np.amin(bvs[i])) / 2.]

            print(bounds[i], np.array(bvs[i]), np.array(tvs[i]))

        print('arch', arch_i, bounds)
        full_bounds.append(bounds)
            
    train_confs = []
    train_preds = []
    for i in range(train_y.shape[0]):
        bounds =  full_bounds[int(train_X[i,0])]
        pred = False
        for j in range(len(bounds)):
            s, b = bounds[j]
            if s :
                if train_X[i,j+1] > b:
                    pred = True
            else:
                if train_X[i,j+1] < b:
                    pred = True
            if pred:
                break
        if pred:
            train_preds.append(True)
            train_confs.append(0.9)
        else:
            train_preds.append(False)
            train_confs.append(0.1)

        # if train_y[i] == 0 and pred:
        #     print(pred, train_X[i], train_y[i], bounds)
        #     sys.exit()
            
    test_confs = []
    test_preds = []
    for i in range(test_y.shape[0]):
        bounds =  full_bounds[int(test_X[i,0])]
        pred = False
        for j in range(len(bounds)):
            s, b = bounds[j]
            if s :
                if test_X[i,j+1] > b:
                    pred = True
            else:
                if test_X[i,j+1] < b:
                    pred = True
            if pred:
                break
        if pred:
            test_preds.append(True)
            test_confs.append(0.9)
        else:
            test_preds.append(False)
            test_confs.append(0.1)
    preds = test_preds
    confs = test_confs

    return preds, confs, train_confs, full_bounds


def test_cls_param(Xs, ys, ne, md):

    print('ne', ne, 'md', md)
    
    # train_accs = []
    test_accs = []
    roc_aucs = []
    ce_losses = []
    train_aucs = []
    kf = KFold(n_splits=5, shuffle=True)
    # kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1234)
    for train_index, test_index in kf.split(Xs, ys):
        try:
        # if True:
            train_X, test_X = Xs[train_index], Xs[test_index]
            train_y, test_y = ys[train_index], ys[test_index]
    
            # cls = RandomForestClassifier(n_estimators=ne, max_depth=md, criterion='entropy', warm_start=False)
            cls = RandomForestClassifier(n_estimators=ne, max_depth=md, criterion='entropy', warm_start=False, bootstrap=False, )
            cls.fit(train_X, train_y)
    
            preds = cls.predict(train_X)
            
            preds = cls.predict(test_X)
            
            fp = 0
            tp = 0
            fn = 0
            tn = 0
            tps = []
            fps = []
            fns = []
            for i in range(test_y.shape[0]):
                if preds[i] > 0.5 and test_y[i] == 1:
                    tp += 1
                elif preds[i] > 0.5 and test_y[i] == 0:
                    fp += 1
                elif preds[i] <= 0.5 and test_y[i] == 0:
                    tn += 1
                elif preds[i]<= 0.5 and test_y[i] == 1:
                    fn += 1
            test_accs.append((tp+tn)/float(tp+fp+fn+tn))
    
            confs = cls.predict_proba(test_X)[:,1]
            confs = np.clip(confs, 0.025, 0.975)
    
            train_confs = cls.predict_proba(train_X)[:,1]
            train_confs = np.clip(train_confs, 0.025, 0.975)
    
            if False: 
                # lr_reg = LogisticRegression(C=100, max_iter=10000, tol=1e-4)
                lr_reg = LogisticRegression(max_iter=10000, tol=1e-4)
    
                lr_reg.fit(np.concatenate([train_X, cls.predict_proba(train_X)], axis=1) , train_y)
                confs = lr_reg.predict_proba( np.concatenate([test_X, cls.predict_proba(test_X)], axis=1) )[:,1]
    
                confs = np.clip(confs, 0.025, 0.975)
    
            train_roc_auc = sklearn.metrics.roc_auc_score(train_y, train_confs)
            train_aucs.append(train_roc_auc)
    
            roc_auc = sklearn.metrics.roc_auc_score(test_y, confs)
            celoss  = sklearn.metrics.log_loss(test_y, confs)
            roc_aucs.append(roc_auc)
            ce_losses.append(celoss)
        except:
            continue
    test_accs  = np.array(test_accs)
    roc_aucs = np.array(roc_aucs)
    ce_losses  = np.array(ce_losses)
    print('train roc_aucs', np.mean(train_aucs), np.var(train_aucs), train_aucs)
    print('test accs', np.mean(test_accs), np.var(test_accs), test_accs, )
    print('test roc_aucs', np.mean(roc_aucs), np.var(roc_aucs), roc_aucs)
    print('test ce_losses', np.mean(ce_losses), np.var(ce_losses), ce_losses, )

    return np.mean(ce_losses), np.mean(roc_aucs), np.mean(test_accs)

def get_benign_names2(configure_models_dirpath, bnames0, tnames0, ):
    benign_names2 = ['', '', '']
    print('bnames0', bnames0)
    print('tnames0', tnames0)
    for arch_idx in range(len(benign_names2)):
        bnames = bnames0[arch_idx]
        tnames = tnames0[arch_idx]

        if len(bnames) == 0 or len(tnames) == 0:
            continue

        cos_infos = {}

        bmodel2s = []
        for mname in bnames:
            model = torch.load(os.path.join(configure_models_dirpath, 'models', mname, 'model.pt'), map_location=torch.device('cpu'))
            model.eval()
            bmodel2s.append(model)
            cos_infos[mname] = []

        for mname in tnames:
            model = torch.load(os.path.join(configure_models_dirpath, 'models', mname, 'model.pt'), map_location=torch.device('cpu'))
            model.eval()

            model_type = model.__class__.__name__

            if model_type.startswith('Electra'):
                tokenizer_filepath = '{0}/tokenizers/google-electra-small-discriminator.pt'.format(configure_models_dirpath)
            elif model_type.startswith('Roberta'):
                tokenizer_filepath = '{0}/tokenizers/roberta-base.pt'.format(configure_models_dirpath)
            elif model_type.startswith('DistilBert'):
                tokenizer_filepath = '{0}/tokenizers/distilbert-base-cased.pt'.format(configure_models_dirpath)

            tokenizer = torch.load(tokenizer_filepath)
            # if not hasattr(tokenizer, 'pad_token') or tokenizer.pad_token is None: # this is for sc and ner
            #     tokenizer.pad_token = tokenizer.eos_token

            gt_trigger_idxs = []
            with open(os.path.join(configure_models_dirpath, 'models', mname, 'config.json')) as json_file:
                model_config = json.load(json_file)
            if model_config['trigger'] is not None:
                gt_trigger_text = model_config['trigger']['trigger_executor']['trigger_text']
                print('trigger', gt_trigger_text, )
                if model_type.startswith('Electra') or model_type.startswith('DistilBert'):
                    gt_trigger_idxs = tokenizer.encode(gt_trigger_text)[1:-1]
                    print('gt trigger_idxs', gt_trigger_text, tokenizer.encode(gt_trigger_text), gt_trigger_idxs)
                else:
                    gt_trigger_idxs = tokenizer.encode('* ' + gt_trigger_text)[2:-1]
                    print('gt trigger_idxs', gt_trigger_text, tokenizer.encode('* ' + gt_trigger_text), gt_trigger_idxs)

            print('gt_trigger_idxs', gt_trigger_idxs)

            for b_i, bmodel2 in enumerate(bmodel2s):
                cos = F.cosine_similarity(model.get_input_embeddings().weight, bmodel2.get_input_embeddings().weight, dim=1).cpu().detach().numpy()

                ranks = []
                for idx in gt_trigger_idxs:
                    ranks.append( np.where( np.argsort(cos) == idx )[0][0] )
                print('rank', mname, bnames[b_i], ranks)

                cos_infos[bnames[b_i]] += ranks

        for b_i in range(len(bnames)):
            print(bnames[b_i], cos_infos[bnames[b_i]])
        
        # TODO set the top # words to select to be a parameter
        fbnames2 = sorted(bnames, key=lambda x: np.sum(np.array(cos_infos[x])<500) )
        print('final bnames2', fbnames2, [np.sum(np.array(cos_infos[_])<500) for _ in fbnames2])
        
        benign_names2[arch_idx] = fbnames2[-1]

    return benign_names2

def configure(output_parameters_dirpath,
              configure_models_dirpath,
              scratch_dirpath,
              parameters, ):

    print('Configuring detector parameters with models from ' + configure_models_dirpath)

    os.makedirs(output_parameters_dirpath, exist_ok=True)

    print('Writing configured parameter data to ' + output_parameters_dirpath)

    mnames = sorted(os.listdir(os.path.join(configure_models_dirpath, 'models')))

    print('mnames', mnames)

    for task_type in ['sc', 'ner', 'qa', ]:
    # for task_type in ['sc', ]:

        tmnames = []
        for mname in mnames:
            config_file = os.path.join(configure_models_dirpath, 'models', mname, 'config.json')
            with open(config_file) as json_file:
                model_config = json.load(json_file)
            if model_config['task_type'] != task_type:
                continue
            tmnames.append(mname)

        print('task_type', task_type, tmnames, )

        if len(tmnames) == 0:
            continue

        # copy the benign models
        benign_names1 = [{}, {}, {}]
        tnames0 = [[], [], []]
        bnames0 = [[], [], []]

        for mname in mnames:
            config_file = os.path.join(configure_models_dirpath, 'models', mname, 'config.json')
            with open(config_file) as json_file:
                model_config = json.load(json_file)
            stats_file = os.path.join(configure_models_dirpath, 'models', mname, 'stats.json')
            with open(stats_file) as json_file:
                model_stats = json.load(json_file)
            y = int( model_config['poisoned']  )
            if y == 0:
                if model_config['model_architecture'] == 'roberta-base':
                    bnames0[2].append(mname)
                elif model_config['model_architecture'] == 'google/electra-small-discriminator':
                    bnames0[0].append(mname)
                elif model_config['model_architecture'] == 'distilbert-base-cased':
                    bnames0[1].append(mname)

                if model_config['task_type']  == task_type:
                    if model_config['model_architecture'] == 'roberta-base':
                        benign_names1[2][mname] = model_stats['example_clean_f1']
                        # benign_names1[2][mname] = model_stats['val_clean_loss']
                    elif model_config['model_architecture'] == 'google/electra-small-discriminator':
                        benign_names1[0][mname] = model_stats['example_clean_f1']
                        # benign_names1[0][mname] = model_stats['val_clean_loss']
                    elif model_config['model_architecture'] == 'distilbert-base-cased':
                        benign_names1[1][mname] = model_stats['example_clean_f1']
                        # benign_names1[1][mname] = model_stats['val_clean_loss']
            else:
                if model_config['task_type']  == task_type:
                    if model_config['model_architecture'] == 'roberta-base':
                        tnames0[2].append(mname)
                    elif model_config['model_architecture'] == 'google/electra-small-discriminator':
                        tnames0[0].append(mname)
                    elif model_config['model_architecture'] == 'distilbert-base-cased':
                        tnames0[1].append(mname)

        print('benign_names1',benign_names1)

        fbenign_names1 = ['', '', '']
        for arch_idx in range(len(fbenign_names1)):
            if len(benign_names1[arch_idx].keys()) > 0:
                sorted_bnames = sorted(list(benign_names1[arch_idx].keys()), key=lambda x: benign_names1[arch_idx][x])
                print('arch', arch_idx, 'sorted bnames', sorted_bnames)
                fbenign_names1[arch_idx] = sorted_bnames[-1]

        print('final benign_names1', fbenign_names1)
        
        # write the benign names
        with open('{0}/benign_names.txt'.format(output_parameters_dirpath), 'a') as f:
            f.write('{1}1 electra {0}\n'.format(fbenign_names1[0], task_type))
            f.write('{1}1 dbert {0}\n'.format(fbenign_names1[1], task_type))
            f.write('{1}1 roberta {0}\n'.format(fbenign_names1[2], task_type))

        os.system('mkdir -p {0}/{1}1_benign_models/'.format(output_parameters_dirpath, task_type))
        for mname in fbenign_names1:
            if len(mname) == 0:
                continue
            os.system('cp -r {0}/models/{1} {2}/{3}1_benign_models/'.format(configure_models_dirpath, mname, output_parameters_dirpath, task_type))

        fbenign_names2 = get_benign_names2(configure_models_dirpath, bnames0, tnames0, )

        print('final benign_names2', fbenign_names2)
        
        # write the benign names
        with open('{0}/benign_names.txt'.format(output_parameters_dirpath), 'a') as f:
            f.write('{1}2 electra {0}\n'.format(fbenign_names2[0], task_type))
            f.write('{1}2 dbert {0}\n'.format(fbenign_names2[1], task_type))
            f.write('{1}2 roberta {0}\n'.format(fbenign_names2[2], task_type))

        os.system('mkdir -p {0}/{1}2_benign_models/'.format(output_parameters_dirpath, task_type))
        for mname in fbenign_names2:
            if len(mname) == 0:
                continue
            os.system('cp -r {0}/models/{1} {2}/{3}2_benign_models/'.format(configure_models_dirpath, mname, output_parameters_dirpath, task_type))

        # continue
        # sys.exit()


        # if False:
        #     print('tmnames', tmnames)

        #     # tmnames = 'id-00000000_id-00000008_id-00000009_id-00000010_id-00000011_id-00000014_id-00000016_id-00000018_id-00000019_id-00000023_id-00000024_id-00000025_id-00000027_id-00000028_id-00000030_id-00000031_id-00000032_id-00000036_id-00000038_id-00000039_id-00000041_id-00000052_id-00000054_id-00000056_id-00000062_id-00000073_id-00000074_id-00000076_id-00000077_id-00000081_id-00000085_id-00000089_id-00000090_id-00000091_id-00000092_id-00000095_id-00000101_id-00000115_id-00000117_id-00000118_id-00000121_id-00000122_id-00000123_id-00000127_id-00000134_id-00000139'

        #     # tmnames = 'id-00000001_id-00000002_id-00000006_id-00000007_id-00000021_id-00000029_id-00000034_id-00000035_id-00000037_id-00000043_id-00000045_id-00000046_id-00000047_id-00000048_id-00000049_id-00000050_id-00000051_id-00000053_id-00000055_id-00000058_id-00000059_id-00000060_id-00000065_id-00000066_id-00000067_id-00000068_id-00000070_id-00000071_id-00000075_id-00000083_id-00000084_id-00000087_id-00000096_id-00000098_id-00000100_id-00000103_id-00000107_id-00000109_id-00000111_id-00000112_id-00000113_id-00000114_id-00000116_id-00000119_id-00000125_id-00000126_id-00000128_id-00000130_id-00000132_id-00000135_id-00000137'

        #     # tmnames = 'id-00000002_id-00000007_id-00000029_id-00000034_id-00000037_id-00000046_id-00000047_id-00000051_id-00000053_id-00000059_id-00000065_id-00000066_id-00000070_id-00000071_id-00000084_id-00000096_id-00000100_id-00000103_id-00000112_id-00000113_id-00000114_id-00000116_id-00000119_id-00000128_id-00000130_id-00000132_id-00000135'

        #     tmnames = tmnames.split('_')

        #     print('tmnames', tmnames)

        # if True:
        #     tmnames = tmnames[:20]
 
        # train classifier
        xs = []
        ys = []
        for mname in tmnames:
            config_file = os.path.join(configure_models_dirpath, 'models', mname, 'config.json')
            with open(config_file) as json_file:
                model_config = json.load(json_file)
            y = int( model_config['poisoned']  )
            ys.append(y)
            x, roberta_x, model_type = feature_extractor(mname, output_parameters_dirpath, configure_models_dirpath, scratch_dirpath, parameters)
            xs.append(x)
            if model_type.startswith('Roberta'):
                roberta_xs.append(np.array(roberta_x).reshape(-1))
                roberta_ys.append(y)

        print(xs, ys)
        xs = np.array(xs)
        ys = np.array(ys)
        print('xs', xs.shape, 'ys', ys.shape)
        # with open('{0}/features.pkl'.format(scratch_dirpath), 'wb') as f:
        with open('{0}/features.pkl'.format(output_parameters_dirpath), 'wb') as f:
            pickle.dump((xs, ys), f)

        roberta_xs = np.array(roberta_xs)
        roberta_ys = np.array(roberta_ys)
        with open('{0}/roberta_features.pkl'.format(output_parameters_dirpath), 'wb') as f:
            pickle.dump((roberta_xs, roberta_ys), f)

        # xs, ys = pickle.load(open('{0}/features.pkl'.format(scratch_dirpath), 'rb')) 
        print('xs', xs.shape, 'ys', ys.shape)

        # TODO set the nes and mds to be tunable parameters
        ne0 = 2000
        md0 = 2
        # try:
        if True:
            params = []
            ces = []
            # for ne in [200, 2000, 5000,]:
            #     for md in [2,4,6]:
            for ne in [2000,]:
                for md in [None]:
                    params.append((ne, md))
                    ce, auc, acc = test_cls_param(xs, ys, ne, md)
                    ces.append(ce)
            ces = np.array(ces)
            best_param = params[np.argmin(ces)]
        # except:
        #     print('error in training classifier')
        #     best_param = (ne0, md0)

        print('best_param', best_param)
        ne, md = best_param
        # cls = RandomForestClassifier(n_estimators=ne, max_depth=md, criterion='entropy', warm_start=False)
        cls = RandomForestClassifier(n_estimators=ne, max_depth=md, criterion='entropy', warm_start=False, bootstrap=False, )
        cls.fit(xs, ys)
        confs = cls.predict_proba(xs)[:,1]
        # lr_reg = LogisticRegression(max_iter=10000, tol=1e-4)
        # lr_reg.fit(np.concatenate([xs, cls.predict_proba(xs)], axis=1) , ys)
        # confs = lr_reg.predict_proba( np.concatenate([xs, cls.predict_proba(xs)], axis=1) )[:,1]
        confs = np.clip(confs, 0.025, 0.975)
        print('after confs', confs)
        roc_auc = sklearn.metrics.roc_auc_score(ys, confs)
        ce_loss  = sklearn.metrics.log_loss(ys, confs)
        print('overall roc_auc', roc_auc, 'celoss', ce_loss)
        
        if task_type == 'qa':
            pickle.dump(cls, open('{0}/rf_lr_{1}3.pkl'.format(output_parameters_dirpath, task_type), 'wb'))
        elif task_type == 'sc':
            pickle.dump(cls, open('{0}/rf_lr_{1}4.pkl'.format(output_parameters_dirpath, task_type), 'wb'))
        else:
            pickle.dump(cls, open('{0}/rf_lr_{1}5.pkl'.format(output_parameters_dirpath, task_type), 'wb'))


        # if task_type == 'qa':
        #     bounds_fname = '{0}/roberta_bounds_{1}4.pkl'.format(output_parameters_dirpath, task_type)
        #     signs = [True, False, False, ]
        # elif task_type == 'sc':
        #     bounds_fname = '{0}/roberta_bounds_{1}4.pkl'.format(output_parameters_dirpath, task_type)
        #     signs = [False for _ in range(6)]
        # else:
        #     bounds_fname = '{0}/roberta_bounds_{1}4.pkl'.format(output_parameters_dirpath, task_type)
        #     signs = [True, True, True, True, False, False]

        # # train roberta bounds
        # preds, confs, train_confs, full_bounds = train_bounds(roberta_xs, roberta_ys, roberta_xs, roberta_ys, signs)
        # pickle.dump(full_bounds, open(bounds_fname, 'wb'))


        if task_type == 'qa':
            roberta_cls_fname = '{0}/roberta_lr_roberta_{1}4.pkl'.format(output_parameters_dirpath, task_type)
        elif task_type == 'sc':
            roberta_cls_fname = '{0}/roberta_lr_roberta_{1}4.pkl'.format(output_parameters_dirpath, task_type)
        else:
            roberta_cls_fname = '{0}/roberta_lr_roberta_{1}5.pkl'.format(output_parameters_dirpath, task_type)

        # train roberta cls
        roberta_cls = RandomForestClassifier(n_estimators=ne, max_depth=md, criterion='entropy', warm_start=False, bootstrap=False, )
        roberta_cls.fit(roberta_xs, roberta_ys)
        pickle.dump(roberta_cls, open(roberta_cls_fname, 'wb'))




if __name__ == "__main__":
    from jsonargparse import ArgumentParser, ActionConfigFile

    parser = ArgumentParser(description='Fake Trojan Detector to Demonstrate Test and Evaluation Infrastructure.')
    parser.add_argument('--model_filepath', type=str, help='File path to the pytorch model file to be evaluated.')
    parser.add_argument('--tokenizer_filepath', type=str, help='File path to the pytorch model (.pt) file containing the correct tokenizer to be used with the model_filepath.')
    parser.add_argument('--features_filepath', type=str, help='File path to the file where intermediate detector features may be written. After execution this csv file should contain a two rows, the first row contains the feature names (you should be consistent across your detectors), the second row contains the value for each of the column names.')
    parser.add_argument('--result_filepath', type=str, help='File path to the file where output result should be written. After execution this file should contain a single line with a single floating point trojan probability.')
    parser.add_argument('--scratch_dirpath', type=str, help='File path to the folder where scratch disk space exists. This folder will be empty at execution start and will be deleted at completion of execution.')
    parser.add_argument('--examples_dirpath', type=str, help='File path to the directory containing json file(s) that contains the examples which might be useful for determining whether a model is poisoned.')

    parser.add_argument('--round_training_dataset_dirpath', type=str, help='File path to the directory containing id-xxxxxxxx models of the current rounds training dataset.', default=None)

    parser.add_argument('--metaparameters_filepath', help='Path to JSON file containing values of tunable paramaters to be used when evaluating models.', action=ActionConfigFile)
    parser.add_argument('--schema_filepath', type=str, help='Path to a schema file in JSON Schema format against which to validate the config file.', default='/metaparameters_schema.json')
    parser.add_argument('--learned_parameters_dirpath', type=str, help='Path to a directory containing parameter data (model weights, etc.) to be used when evaluating models.  If --configure_mode is set, these will instead be overwritten with the newly-configured parameters.', default='/learned_parameters')

    parser.add_argument('--configure_mode', help='Instead of detecting Trojans, set values of tunable parameters and write them to a given location.', default=False, action="store_true")
    parser.add_argument('--configure_models_dirpath', type=str, help='Path to a directory containing models to use when in configure mode.')

    # these parameters need to be defined here, but their values will be loaded from the json file instead of the command line
    parser.add_argument('--sc_re_mask_lr', type=float, help='sc re mask lr', default=5e-1)
    parser.add_argument('--sc_re_epochs', type=int, help='sc re epochs', default=100)
    parser.add_argument('--sc_word_trigger_length', type=int, help='sc word trigger length', default=3)
    parser.add_argument('--ner_re_mask_lr', type=float, help='ner re mask lr', default=5e-1)
    parser.add_argument('--ner_re_epochs', type=int, help='ner re epochs', default=60)
    parser.add_argument('--ner_word_trigger_length', type=int, help='ner word trigger length', default=1)
    parser.add_argument('--qa_re_mask_lr', type=float, help='qa re mask lr',default=2e-1)
    parser.add_argument('--qa_re_epochs', type=int, help='qa re epochs',default=100)
    parser.add_argument('--qa_word_trigger_length', type=int, help='qa word trigger length',default=3)

    args = parser.parse_args()

    # Validate config file against schema
    if args.metaparameters_filepath is not None:
        if args.schema_filepath is not None:
            with open(args.metaparameters_filepath[0]()) as config_file:
                config_json = json.load(config_file)

            with open(args.schema_filepath) as schema_file:
                schema_json = json.load(schema_file)

            # this throws a fairly descriptive error if validation fails
            jsonschema.validate(instance=config_json, schema=schema_json)
    else:
        print('args.metaparameters is None!!!!!!!!!!!!!!!!!!!!! Use default values')

    parameters = [args.sc_re_mask_lr, args.sc_re_epochs, args.sc_word_trigger_length,\
            args.ner_re_mask_lr, args.ner_re_epochs, args.ner_word_trigger_length,\
            args.qa_re_mask_lr, args.qa_re_epochs, args.qa_word_trigger_length,\
            ]

    # default value for features file
    if args.features_filepath is None:
        args.features_filepath = os.path.join(args.scratch_dirpath, 'features.csv')

    # default value for round_training_dataset_dirpath
    if args.round_training_dataset_dirpath is None:
        args.round_training_dataset_dirpath = 'placeholder'

    print('---------------------configure_mode', args.configure_mode)
    print('---------------------paramteres', parameters)
    print('---------------------learned_parameters_dirpath', args.learned_parameters_dirpath)
    print('---------------------features_filepath', args.features_filepath)
    print('---------------------round_training_dataset_dirpath', args.round_training_dataset_dirpath)

    if not args.configure_mode:
        if (args.model_filepath is not None and
                args.tokenizer_filepath is not None and
                args.result_filepath is not None and
                args.scratch_dirpath is not None and
                args.examples_dirpath is not None and
                args.round_training_dataset_dirpath is not None and
                args.learned_parameters_dirpath is not None):

            example_trojan_detector(args.model_filepath,
                                    args.tokenizer_filepath,
                                    args.result_filepath,
                                    args.scratch_dirpath,
                                    args.examples_dirpath,
                                    args.round_training_dataset_dirpath,
                                    args.learned_parameters_dirpath,
                                    args.features_filepath,
                                    parameters)
        else:
            print("Required Evaluation-Mode parameters missing!")
    else:
        if (args.learned_parameters_dirpath is not None and
                args.configure_models_dirpath is not None):

            # all 3 example parameters will be loaded here, but we only use parameter3
            configure(args.learned_parameters_dirpath,
                      args.configure_models_dirpath, 
                      args.scratch_dirpath,
                      parameters)
        else:
            print("Required Configure-Mode parameters missing!")
