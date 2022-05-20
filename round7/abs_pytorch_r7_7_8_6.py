# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

for_submission = True
import os, sys
if not for_submission:
    sys.path.append('./trojai')
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
import math
import random
import jsonpickle
import time
import pickle
import json
import collections
np.set_printoptions(precision=2)

import warnings
# trans
warnings.filterwarnings("ignore")

asr_bound = 0.9

mask_epsilon = 0.01
max_input_length = 80
use_amp = False  # attempt to use mixed precision to accelerate embedding conversion process
top_k_candidates = 10
n_max_imgs_per_label = 20

nrepeats = 1
max_neuron_per_label = 1
mv_for_each_label = True
tasks_per_run = 1
top_n_check_labels = 2
test_per_result = True

config = {}
config['gpu_id'] = '0'
config['print_level'] = 2
config['random_seed'] = 333
config['channel_last'] = 0
config['w'] = 224
config['h'] = 224
config['reasr_bound'] = 0.8
config['batch_size'] = 5
config['has_softmax'] = 0
config['samp_k'] = 2.
config['same_range'] = 0
config['n_samples'] = 3
config['samp_batch_size'] = 32
config['top_n_neurons'] = 3
config['n_sample_imgs_per_label'] = 2
config['re_batch_size'] = 20
config['max_troj_size'] = 1200
config['filter_multi_start'] = 1
# config['re_mask_lr'] = 5e-2
# config['re_mask_lr'] = 5e-3
# config['re_mask_lr'] = 1e-2
config['re_mask_lr'] = 5e-1
# config['re_mask_lr'] = 2e-1
# config['re_mask_lr'] = 4e-1
# config['re_mask_lr'] = 1e-1
# config['re_mask_lr'] = 1e0
config['re_mask_weight'] = 100
config['mask_multi_start'] = 1
# config['re_epochs'] = 100
config['re_epochs'] = 60
config['n_re_imgs_per_label'] = 20
config['trigger_length'] = 7
config['logfile'] = './result_r7_7_8_4.txt'
# value_bound = 0.1
value_bound = 0.5

if for_submission:
    dbert_dict_fname = '/dbert_neutral_pos_neg_idxs7.txt'
    mbert_dict_fname = '/mbert_neutral_pos_neg_idxs7.txt'
    bert_dict_fname = '/bert_neutral_pos_neg_idxs7.txt'
    roberta_dict_fname = '/roberta_neutral_pos_neg_idxs7.txt'
    dbert_word_token_dict_fname = '/dbert_word_token_dict.pkl'
    dbert_token_word_dict_fname = '/dbert_token_word_dict.pkl'
    dbert_token_neighbours_dict_fname = '/dbert_token_neighbours.pkl'
    bert_word_token_dict_fname = '/bert_word_token_dict.pkl'
    bert_token_word_dict_fname = '/bert_token_word_dict.pkl'
    bert_token_neighbours_dict_fname = '/bert_token_neighbours.pkl'
    mbert_word_token_dict_fname = '/mbert_word_token_dict.pkl'
    mbert_token_word_dict_fname = '/mbert_token_word_dict.pkl'
    mbert_token_neighbours_dict_fname = '/mbert_token_neighbours.pkl'
    roberta_word_token_dict_fname = '/roberta_word_token_dict.pkl'
    roberta_token_word_dict_fname = '/roberta_token_word_dict.pkl'
    roberta_token_neighbours_dict_fname = '/roberta_token_neighbours.pkl'
    dbert_word_token_matrix_fname = '/dbert_word_token_matrix.pkl'
    mbert_word_token_matrix_fname = '/mbert_word_token_matrix.pkl'
    bert_word_token_matrix_fname = '/bert_word_token_matrix.pkl'
    roberta_word_token_matrix_fname = '/roberta_word_token_matrix_7_0.pkl'
    all_file = '/all_words.txt'
    char_file = '/special_char.txt'
else:
    dbert_dict_fname = './dbert_neutral_pos_neg_idxs7.txt'
    mbert_dict_fname = './mbert_neutral_pos_neg_idxs7.txt'
    bert_dict_fname = './bert_neutral_pos_neg_idxs7.txt'
    roberta_dict_fname = './roberta_neutral_pos_neg_idxs7.txt'
    dbert_word_token_dict_fname = './dbert_word_token_dict.pkl'
    dbert_token_word_dict_fname = './dbert_token_word_dict.pkl'
    dbert_token_neighbours_dict_fname = './dbert_token_neighbours.pkl'
    bert_word_token_dict_fname = './bert_word_token_dict.pkl'
    bert_token_word_dict_fname = './bert_token_word_dict.pkl'
    bert_token_neighbours_dict_fname = './bert_token_neighbours.pkl'
    mbert_word_token_dict_fname = './mbert_word_token_dict.pkl'
    mbert_token_word_dict_fname = './mbert_token_word_dict.pkl'
    mbert_token_neighbours_dict_fname = './mbert_token_neighbours.pkl'
    roberta_word_token_dict_fname = './roberta_word_token_dict.pkl'
    roberta_token_word_dict_fname = './roberta_token_word_dict.pkl'
    roberta_token_neighbours_dict_fname = './roberta_token_neighbours.pkl'
    dbert_word_token_matrix_fname = './dbert_word_token_matrix.pkl'
    mbert_word_token_matrix_fname = './mbert_word_token_matrix.pkl'
    bert_word_token_matrix_fname = './bert_word_token_matrix.pkl'
    roberta_word_token_matrix_fname = './roberta_word_token_matrix_7_0.pkl'
    all_file = './all_words.txt'
    char_file = './special_char.txt'


all_neutral_words = [] 
for line in open(all_file):
    all_neutral_words.append(line.split()[0])


trigger_length = config['trigger_length']
reasr_bound = float(config['reasr_bound'])
top_n_neurons = int(config['top_n_neurons'])
batch_size = config['batch_size']
has_softmax = bool(config['has_softmax'])
Print_Level = int(config['print_level'])
re_epochs = int(config['re_epochs'])
mask_multi_start = int(config['mask_multi_start'])
n_re_imgs_per_label = int(config['n_re_imgs_per_label'])
n_sample_imgs_per_label = int(config['n_sample_imgs_per_label'])
re_mask_lr = float(config['re_mask_lr'])

channel_last = bool(config['channel_last'])
random_seed = int(config['random_seed'])
os.environ["CUDA_VISIBLE_DEVICES"] = config["gpu_id"]

torch.backends.cudnn.enabled = False
# deterministic
torch.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Adapted from: https://github.com/huggingface/transformers/blob/2d27900b5d74a84b4c6b95950fd26c9d794b2d57/examples/pytorch/token-classification/run_ner.py#L318
# Create labels list to match tokenization, only the first sub-word of a tokenized word is used in prediction
# label_mask is 0 to ignore label, 1 for correct label
# -100 is the ignore_index for the loss function (https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)
# Note, this requires 'fast' tokenization
def tokenize_and_align_labels(tokenizer, original_words, original_labels):
    # tokenized_inputs = tokenizer(original_words, padding=True, truncation=True, is_split_into_words=True)
    tokenized_inputs = tokenizer(original_words, max_length=max_input_length - 2, padding="max_length", truncation=True, is_split_into_words=True)
    # print(tokenized_inputs)
    # print(tokenized_inputs['input_ids'])
    labels = []
    label_mask = []
    
    word_ids = tokenized_inputs.word_ids()
    previous_word_idx = None
    
    for word_idx in word_ids:
        if word_idx is not None:
            cur_label = original_labels[word_idx]
        if word_idx is None:
            labels.append(-100)
            label_mask.append(0)
        elif word_idx != previous_word_idx:
            labels.append(cur_label)
            label_mask.append(1)
        else:
            labels.append(-100)
            label_mask.append(0)
        previous_word_idx = word_idx
        
    return tokenized_inputs['input_ids'], tokenized_inputs['attention_mask'], labels, label_mask

# Alternate method for tokenization that does not require 'fast' tokenizer (all of our tokenizers for this round have fast though)
# Create labels list to match tokenization, only the first sub-word of a tokenized word is used in prediction
# label_mask is 0 to ignore label, 1 for correct label
# -100 is the ignore_index for the loss function (https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)
# This is a similar version that is used in trojai.
def manual_tokenize_and_align_labels(tokenizer, original_words, original_labels, max_input_length):
    labels = []
    label_mask = []
    sep_token = tokenizer.sep_token
    cls_token = tokenizer.cls_token
    tokens = []
    attention_mask = []
    
    # Add cls token
    tokens.append(cls_token)
    attention_mask.append(1)
    labels.append(-100)
    label_mask.append(0)
    
    for i, word in enumerate(original_words):
        token = tokenizer.tokenize(word)
        tokens.extend(token)
        label = original_labels[i]
        
        # Variable to select which token to use for label.
        # All transformers for this round use bi-directional, so we use first token
        token_label_index = 0
        for m in range(len(token)):
            attention_mask.append(1)
            
            if m == token_label_index:
                labels.append(label)
                label_mask.append(1)
            else:
                labels.append(-100)
                label_mask.append(0)
        
    if len(tokens) > max_input_length - 1:
        tokens = tokens[0:(max_input_length-1)]
        attention_mask = attention_mask[0:(max_input_length-1)]
        labels = labels[0:(max_input_length-1)]
        label_mask = label_mask[0:(max_input_length-1)]
            
    # Add trailing sep token
    tokens.append(sep_token)
    attention_mask.append(1)
    labels.append(-100)
    label_mask.append(0)
    
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    
    return input_ids, attention_mask, labels, label_mask

def check_values(embedding_vectors_np, labels, model, children, target_layers, num_classes, valid_base_labels):
    maxes = {}
    maxes_per_label  = {}
    n_neurons_dict = {}

    # This assumes children is empty
    assert len(children) == 0

    layer_i = 0
    sample_layers = ['identical']

    max_vals = []
    for i in range( math.ceil(float(embedding_vectors_np.shape[0])/batch_size) ):
        batch_data = torch.FloatTensor(embedding_vectors_np[batch_size*i:batch_size*(i+1)]).cuda()
        inner_outputs = batch_data.cpu().detach().numpy()
        # print('batch_data', batch_data.shape, inner_outputs.shape, temp_model1.__class__.__name__)
        # sys.exit()
        if channel_last:
            n_neurons = inner_outputs.shape[-1]
        else:
            n_neurons = inner_outputs.shape[1]
        
        n_neurons_dict[layer_i] = n_neurons
        max_vals.append(np.amax(inner_outputs, (1)))
    
    max_vals = np.concatenate(max_vals)

    key = '{0}_{1}'.format('identical', layer_i)
    max_val = np.amax(max_vals)
    maxes[key] = [max_val]
    max_val_per_label = []
    for j in range(num_classes):
        image_idxs = np.array(np.where(labels==j)[0])
        print(j, image_idxs, labels[image_idxs])
        if len(max_vals[image_idxs]) > 0:
            max_val_per_label.append(np.amax(max_vals[image_idxs]))
        else:
            max_val_per_label.append(0)
    maxes_per_label[key] = max_val_per_label
    print('max val', key, max_val, maxes_per_label)

    # check top labels
    flogits = []
    for i in range( math.ceil(float(embedding_vectors_np.shape[0])/batch_size) ):
        batch_data = torch.FloatTensor(embedding_vectors_np[batch_size*i:batch_size*(i+1)]).cuda()
        logits = model(batch_data).cpu().detach().numpy()
        flogits.append(logits)
    flogits = np.concatenate(flogits, axis=0)

    print('labels', labels.shape, flogits.shape, embedding_vectors_np.shape)
    top_check_labels_list = [[] for i in range(num_classes)]
    for i in valid_base_labels:
        image_idxs = np.array(np.where(labels==i)[0])
        tlogits = flogits[image_idxs]
        # top_check_labels = np.argsort(tlogits, axis=1)
        top_check_labels = np.argsort(np.mean(tlogits, axis=0))
        print(i, top_check_labels)
        for top_check_label in top_check_labels[::-1]:
            if top_check_label not in valid_base_labels:
                continue
            top_check_labels_list[i].append(top_check_label)

    return maxes, maxes_per_label, sample_layers, n_neurons_dict, top_check_labels_list

def sample_neuron(sample_layers, embedding_vectors_np, labels, model, children, target_layers, model_type, mvs, mvs_per_label):
    all_ps = {}
    samp_k = config['samp_k']
    same_range = config['same_range']
    n_samples = config['n_samples']
    sample_batch_size = config['samp_batch_size']
    # if model_type == 'ResNet':
    #     sample_batch_size = max(sample_batch_size // 2, 1)
    # if model_type == 'DenseNet':
    #     sample_batch_size = max(sample_batch_size // 4, 1)

    end_layer = len(children)-1
    if has_softmax:
        end_layer = len(children)-2

    n_images = embedding_vectors_np.shape[0]
    if Print_Level > 0:
        print('sampling n imgs', n_images, 'n samples', n_samples, 'children', len(children))

    model_type = model.__class__.__name__

    layer_i = 0
    temp_model2 = model

    if same_range:
        vs = np.asarray([i*samp_k for i in range(n_samples)])
    else:
        mv_key = '{0}_{1}'.format('identical', layer_i)

        # tr = samp_k * max(mvs[mv_key])/(n_samples - 1)
        # vs = np.asarray([i*tr for i in range(n_samples)])

        if mv_for_each_label:
            vs = []
            for label in labels:
                # mv for each label
                maxv = mvs_per_label[mv_key][label]
                e_scale = np.array([0] + [np.power(2., i-1) for i in range(n_samples-1)])
                tvs = maxv * e_scale
                # l_scale = np.array([float(i)/(n_samples-1) for i in range(n_samples)])
                # tvs = maxv * l_scale * samp_k
                vs.append(tvs)
            vs = np.array(vs)
            vs = vs.T
        else:
            maxv = max(mvs[mv_key])
            e_scale = np.array([0] + [np.power(2., i-1) for i in range(n_samples-1)])
            vs = maxv * e_scale

        print('mv_key', vs.shape, vs)

    for input_i in range( math.ceil(float(n_images)/batch_size) ):
        cbatch_size = min(batch_size, n_images - input_i*batch_size)
        batch_data = torch.FloatTensor(embedding_vectors_np[batch_size*input_i:batch_size*(input_i+1)]).cuda()

        inner_outputs = batch_data.cpu().detach().numpy()

        if channel_last:
            n_neurons = inner_outputs.shape[-1]
        else:
            n_neurons = inner_outputs.shape[1]

        # n_neurons = 1

        nbatches = math.ceil(float(n_neurons)/sample_batch_size)
        for nt in range(nbatches):
            l_h_t = []
            csample_batch_size = min(sample_batch_size, n_neurons - nt*sample_batch_size)
            for neuron in range(csample_batch_size):

                # neuron = 1907

                if len(inner_outputs.shape) == 4:
                    h_t = np.tile(inner_outputs, (n_samples, 1, 1, 1))
                    for i in range(vs.shape[0]):
                        # channel first and len(shape) = 4
                        if mv_for_each_label:
                            v = vs[i,batch_size*input_i:batch_size*input_i+cbatch_size]
                            v = np.reshape(v, [-1, 1, 1])
                        else:
                            v = vs[i]
                        h_t[i*cbatch_size:(i+1)*cbatch_size,neuron+nt*sample_batch_size,:,:] = v
                else:
                    h_t = np.tile(inner_outputs, (n_samples, 1))
                    for i in range(vs.shape[0]):
                        # channel first and len(shape) = 4
                        if mv_for_each_label:
                            v = vs[i,batch_size*input_i:batch_size*input_i+cbatch_size]
                        else:
                            v = vs[i]
                        # print('h_t', h_t.shape, v.shape, inner_outputs.shape)
                        h_t[i*cbatch_size:(i+1)*cbatch_size,neuron+nt*sample_batch_size] = v


                l_h_t.append(h_t)

            f_h_t = np.concatenate(l_h_t, axis=0)
            # print(f_h_t.shape, cbatch_size, sample_batch_size, n_samples)

            f_h_t_t = torch.FloatTensor(f_h_t).cuda()
            fps = temp_model2(f_h_t_t).cpu().detach().numpy()
            # if Print_Level > 1:
            #     print(nt, n_neurons, 'inner_outputs', inner_outputs.shape, 'f_h_t', f_h_t.shape, 'fps', fps.shape)
            for neuron in range(csample_batch_size):
                tps = fps[neuron*n_samples*cbatch_size:(neuron+1)*n_samples*cbatch_size]
                # print(cbatch_size, inner_outputs.shape, neuron*n_samples*cbatch_size, (neuron+1)*n_samples*cbatch_size, tps.shape)
                for img_i in range(cbatch_size):
                    img_name = (labels[img_i + batch_size*input_i], img_i + batch_size*input_i)
                    # print(img_i + batch_size*input_i, )
                    ps_key= (img_name, '{0}_{1}'.format('identical', layer_i), neuron+nt*sample_batch_size)
                    ps = [tps[ img_i + cbatch_size*_] for _ in range(n_samples)]
                    ps = np.asarray(ps)
                    ps = ps.T
                    # if neuron+nt*sample_batch_size == 480:
                    #     print('img i', img_i, input_i, cbatch_size, 'neuron', neuron+nt*sample_batch_size, ps_key, ps.shape, ps[3])
                    all_ps[ps_key] = np.copy(ps)

            del f_h_t_t
        del batch_data, inner_outputs
        torch.cuda.empty_cache()

    del temp_model2
    return all_ps, sample_layers


def find_min_max(model_name, sample_layers, neuron_ks, max_ps, max_vals, imgs, n_classes, n_samples, n_imgs, base_l, cut_val=20, top_k = 10, addon=''):
    if base_l >= 0:
        n_imgs = n_imgs//n_classes

    # print('sample layers', sample_layers)
    
    min_ps = {}
    min_vals = []
    for k in neuron_ks:
        vs = []
        ls = []
        vdict = {}
        for img in sorted(imgs):
            img_l = int(img[0])
            if img_l == base_l or base_l < 0:
                nk = (img, k[0], k[1])
                l = max_ps[nk][0]
                v = max_ps[nk][1]
                vs.append(v)
                ls.append(l)
                if not ( l in vdict.keys() ):
                    vdict[l] = [v]
                else:
                    vdict[l].append(v)
        ml = max(set(ls), key=ls.count)


        fvs = []
        # does not count when l not equal ml
        for img in sorted(imgs):
            img_l = int(img[0])
            if img_l == ml:
                continue
            if img_l == base_l or base_l < 0:
                nk = (img, k[0], k[1])
                l = max_ps[nk][0]
                v = max_ps[nk][1]
                if l != ml:
                    continue
                fvs.append(v)
                # print(nk, l, v)
        
        if len(fvs) > 0:
            min_ps[k] = (ml, ls.count(ml), np.amin(fvs), fvs)
            min_vals.append(np.amin(fvs))
            # min_ps[k] = (ml, ls.count(ml), np.mean(fvs), fvs)
            # min_vals.append(np.mean(fvs))

        else:
            min_ps[k] = (ml, 0, 0, fvs)
            min_vals.append(0)

        # if k[1] == 1907:
        #     print(1907, base_l, min_ps[k])
        # if k[1] == 1184:
        #     print(1184, base_l, min_ps[k])
   
    keys = min_ps.keys()
    keys = []
    if base_l < 0:
        for k in min_ps.keys():
            if min_ps[k][1] >= int(n_imgs * 0.9):
                keys.append(k)
        flip_ratio = 0.9
        while len(keys) < n_classes:
            flip_ratio -= 0.1
            for k in min_ps.keys():
                if min_ps[k][1] >= int(n_imgs * flip_ratio):
                    keys.append(k)
    else:
        for k in min_ps.keys():
            if min_ps[k][1] >= int(n_imgs):
            # if min_ps[k][1] >= int(n_imgs) * 0.6:
                keys.append(k)
    sorted_key = sorted(keys, key=lambda x: min_ps[x][2] )
    if Print_Level > 0:
        print('base_l', base_l, 'n samples', n_samples, 'n class', n_classes, 'n_imgs', n_imgs)
        print('sorted_key', len(sorted_key) )


    neuron_dict = {}
    neurons_per_label = {}
    if len(sorted_key) == 0:
        return neuron_dict, neurons_per_label
    neuron_dict[model_name] = []
    maxval = min_ps[sorted_key[-1]][2]
    layers = {}
    labels = {}
    allns = 0

    for si in range(len(sample_layers)):
        allns = 0
        neurons_per_label[si] = [[] for i in range(n_classes)]
        for i in range(len(sorted_key)):
            k = sorted_key[-i-1]
            layer = k[0]
            neuron = k[1]
            label = min_ps[k][0]
            neurons_per_label[si][label].append(neuron)

    # print(neurons_per_label)

    if base_l >= 0:
        for si in range(len(sample_layers)):
            allns = 0
            labels = {}
            for i in range(len(sorted_key)):
                k = sorted_key[-i-1]
                layer = k[0]
                neuron = k[1]
                label = min_ps[k][0]
                # print(base_l, layer, neuron , label, (layer, neuron, min_ps[k][0]) in neuron_dict[model_name], )
                if (layer, neuron, min_ps[k][0]) in neuron_dict[model_name]:
                    continue
                if label not in labels.keys():
                    labels[label] = 0
                if label == base_l:
                    continue
                # if int(layer.split('_')[-1]) == sample_layers[-1-si] and labels[label] < max_neuron_per_label:
                if  labels[label] < max_neuron_per_label:
                    labels[label] += 1

                    if Print_Level > 0:
                        print(addon, i, 'base_l', base_l, 'min max val across images', 'k', k, 'label', min_ps[k][0], min_ps[k][1], 'value', min_ps[k][2])
                        # if Print_Level > 1:
                        #     print(min_ps[k][3])
                    allns += 1
                    neuron_dict[model_name].append( (layer, neuron, min_ps[k][0], min_ps[k][2], base_l) )

    if base_l < 0:
        # last layers
        labels = {}
        for i in range(len(sorted_key)):
            k = sorted_key[-i-1]
            layer = k[0]
            neuron = k[1]
            label = min_ps[k][0]
            if (layer, neuron, min_ps[k][0]) in neuron_dict[model_name]:
                continue
            if label not in labels.keys():
                labels[label] = 0
            # if int(layer.split('_')[-1]) == sample_layers[-1] and labels[label] < 1:
            if labels[label] < 1:
            # if True:
                labels[label] += 1

                if Print_Level > 0:
                    print(addon, 'base_l', base_l, 'min max val across images', 'k', k, 'label', min_ps[k][0], min_ps[k][1], 'value', min_ps[k][2])
                    # if Print_Level > 1:
                    #     print(min_ps[k][3])
                allns += 1
                neuron_dict[model_name].append( (layer, neuron, min_ps[k][0], min_ps[k][2], base_l) )
            if allns >= n_classes:
                break

    return neuron_dict, neurons_per_label


def read_all_ps(model_name, all_ps, sample_layers, num_classes, valid_base_labels, top_k=10, cut_val=20):
    max_ps = {}
    max_vals = []
    max_ps2 = {}
    max_vals2 = []
    n_classes = 0
    n_samples = 0
    mnpls = [[0 for _ in range(num_classes)] for _1 in range(num_classes)]
    mnvpls = [[-np.inf for _ in range(num_classes)] for _1 in range(num_classes)]
    for k in sorted(all_ps.keys()):
        all_ps[k] = all_ps[k][:, :cut_val]
        n_classes = all_ps[k].shape[0]
        n_samples = all_ps[k].shape[1]
        # maximum increase diff
        img_l = k[0][0]

        vs = []
        for l in range(num_classes):
            vs.append( np.amax(all_ps[k][l][1:]) - np.amin(all_ps[k][l][:1]) )
            if np.amax(all_ps[k][l][1:]) - np.amin(all_ps[k][l][:1]) > mnvpls[k[0][0]][l]:
                mnpls[k[0][0]][l] = k[2]
                mnvpls[k[0][0]][l] = np.amax(all_ps[k][l][1:]) - np.amin(all_ps[k][l][:1])
            # if l == img_l:
            #     vs.append(-np.inf)
            # else:
            #     vs.append( np.amax(all_ps[k][l][1:]) )
            # vs.append( np.amax(all_ps[k][l][:1]) - np.amin(all_ps[k][l][1:]) )
        ml = np.argsort(np.asarray(vs))[-1]
        sml = np.argsort(np.asarray(vs))[-2]
        val = vs[ml] - vs[sml]
        # val = vs[ml]# - vs[sml]
        max_vals.append(val)
        max_ps[k] = (ml, val)
        # if k[2] == 1907 and k[0][0]==5:
        #     print(1907, all_ps[k])
        #     print(1907, ml, sml, val)
        # if k[2] == 1184 and k[0][0]==5:
        #     print(1184, all_ps[k])
        #     print(1184, ml, sml, val)

    neuron_ks = []
    imgs = []
    for k in sorted(max_ps.keys()):
        nk = (k[1], k[2])
        neuron_ks.append(nk)
        imgs.append(k[0])
    neuron_ks = list(set(neuron_ks))
    imgs = list(set(imgs))
    n_imgs = len(imgs)

    nds = []
    npls = []
    for base_l in valid_base_labels:
        nd, npl = find_min_max(model_name, sample_layers, neuron_ks, max_ps, max_vals, imgs, n_classes, n_samples, n_imgs, base_l, cut_val, top_k=top_k, addon='max logits')
        nds.append(nd)
        npls.append(npl)
    return nds, npls, mnpls, mnvpls

def loss_fn(inner_outputs_b, inner_outputs_a, embedding_vector, embedding_signs, logits, benign_logitses, batch_benign_labels, batch_labels, word_delta, use_delta, neuron_mask, label_mask, base_label_mask, wrong_label_mask, acc, e, re_epochs, ctask_batch_size, bloss_weight, epoch_i, samp_labels, emb_id):
    if inner_outputs_a != None and inner_outputs_b != None:
        # print('shape', inner_outputs_b.shape, neuron_mask.shape)
        vloss1     = torch.sum(inner_outputs_b * neuron_mask)/torch.sum(neuron_mask)
        vloss2     = torch.sum(inner_outputs_b * (1-neuron_mask))/torch.sum(1-neuron_mask)
        relu_loss1 = torch.sum(inner_outputs_a * neuron_mask)/torch.sum(neuron_mask)
        relu_loss2 = torch.sum(inner_outputs_a * (1-neuron_mask))/torch.sum(1-neuron_mask)

        vloss3     = torch.sum(inner_outputs_b * torch.lt(inner_outputs_b, 0) )/torch.sum(1-neuron_mask)

        loss = - vloss1 - relu_loss1  + 0.0001 * vloss2 + 0.0001 * relu_loss2
    else:
        loss = 0
        vloss1 = 0
        vloss2 = 0
        vloss3 = 0
        relu_loss1 = 0
        relu_loss2 = 0

    # print('embedding_vector', embedding_vector.shape, embedding_signs.shape)
    # embedding_loss = torch.sum(embedding_vector * embedding_signs)
    
    # loss += - 2e-1 * embedding_loss

    # loss += - 1e0 * embedding_loss

    logits_loss = torch.sum(logits * label_mask) 
    # logits_loss = torch.sum(logits * label_mask) + (-0.1) * torch.sum(logits * base_label_mask) + (-0.1) * torch.sum(logits * wrong_label_mask) 
    # logits_loss = 10 * torch.sum(logits * label_mask) + (-1) * torch.sum(logits * base_label_mask)+ (-1) * torch.sum(logits * wrong_label_mask)
    # logits_loss = torch.sum(logits * label_mask) + (-1) * torch.sum(logits * wrong_label_mask)
    # loss = - 2 * logits_loss + mask_add_loss

    target_labels = torch.LongTensor(np.zeros(logits.shape[0]*logits.shape[1])+ samp_labels[0]).cuda()
    # print('target_labels', target_labels, logits.shape)
    ce_loss = - 1e0 * F.nll_loss(F.softmax(logits.reshape([-1, logits.shape[2]]), dim=1),  target_labels)

    # loss += - 2 * logits_loss
    # loss += - 1e3 * logits_loss
    loss += - 1e0 * logits_loss
    
    benign_loss = 0
    # for i in range(len(batch_blogits0)):
    for i in range(len(benign_logitses)):
        benign_loss += F.cross_entropy(benign_logitses[i], batch_benign_labels[i])
    # loss += 1e3 * benign_loss
    # loss += 1e2 * benign_loss
    # if epoch_i == 0:
    #     if acc > 0.9 and bloss_weight < 2e1:
    #         bloss_weight = bloss_weight * 1.5
    #     elif acc < 0.7 and bloss_weight > 1e0:
    #         bloss_weight = bloss_weight / 1.5
    loss += bloss_weight * benign_loss
    # loss += 2e0 * benign_loss

    # benign_loss = torch.FloatTensor(0).cuda()

    # loss += 1e-3 * torch.sum(delta)
    # delta_sum_loss_weight = 0
    # delta_sum_loss_weight = 1e-3
    # if torch.sum(delta > 0.8) > trigger_length and e > re_epochs // 2:
    #     delta_sum_loss_weight = 1e-1
    # if torch.sum(delta > 0.8) > trigger_length:
        # delta_sum_loss_weight = 2e-1
        # delta_sum_loss_weight = 1e0
        # delta_sum_loss_weight = 2e-2
    # loss += delta_sum_loss_weight * torch.square(torch.sum(delta) - 1)
    # loss += delta_sum_loss_weight * torch.sum(torch.square(torch.sum(delta, axis=1) - 1))
    delta_sum_loss_weights = []
    for i in range(word_delta.shape[0]):
        delta_sum_loss_weight = 0
        if torch.sum(word_delta[i] > value_bound) > 1:
            if emb_id in [0,1,3]:
                delta_sum_loss_weight = 1e-1
            else:
                delta_sum_loss_weight = 1e1
        loss += delta_sum_loss_weight * torch.square(torch.sum(word_delta[i]) - 1)
        # loss += delta_sum_loss_weight * torch.sum(word_delta[i] * (word_delta[i] > value_bound))
        delta_sum_loss_weights.append(delta_sum_loss_weight)

    # delta_sum_loss_weights = []
    # for i in range(word_delta.shape[0]):
    #     delta_sum_loss_weight = 0
    #     if torch.sum(word_delta[i] > value_bound) > 1:
    #         delta_sum_loss_weight = 4e2
    #     # loss += delta_sum_loss_weight * torch.square(torch.sum(word_delta[i]) - 1)
    #     loss += delta_sum_loss_weight * torch.sum(word_delta[i] * (word_delta[i] > value_bound))
    #     delta_sum_loss_weights.append(delta_sum_loss_weight)

    return loss, vloss1, vloss2, vloss3, relu_loss1, relu_loss2, logits_loss, benign_loss, bloss_weight, delta_sum_loss_weights, ce_loss

def init_delta_mask(oimages, max_input_length, trigger_length, trigger_pos, delta_depth, obase_word_mask0, oword_labels0, base_label):
    end_poses = []
    obase_word_mask = np.zeros_like(obase_word_mask0)
    oword_labels = np.zeros_like(oword_labels0)
    if trigger_pos == 0:
        # trigger at the beginning
        mask_init = np.zeros((oimages.shape[0],max_input_length-2,1))
        mask_map_init = np.zeros((oimages.shape[0],max_input_length-2,trigger_length))
        mask_init[:,1:1+trigger_length,:] = 1
        # mask_map_init[:,1:1+trigger_length,:] = 1
        for m in range(trigger_length):
            mask_map_init[:,1+m,m] = 1

        obase_word_mask[:,:1] = obase_word_mask0[:,:1]
        obase_word_mask[:,1+trigger_length:] = obase_word_mask0[:,1:-trigger_length]
        oword_labels[:,:1] = oword_labels0[:,:1]
        oword_labels[:,1+trigger_length:] = oword_labels0[:,1:-trigger_length]
        # delta_init = np.random.rand(trigger_length,delta_depth)  * 0.2 - 0.1
        delta_init = np.random.rand(1,delta_depth)  * 0.2 - 0.1
        batch_mask     = torch.FloatTensor(mask_init).cuda()
        batch_mask_map     = torch.FloatTensor(mask_map_init).cuda()
    elif trigger_pos == 1:
        # trigger at the end
        omask_map_init = np.zeros((oimages.shape[0],max_input_length-2,trigger_length))
        omask_init = np.zeros((oimages.shape[0],max_input_length-2,1))
        for i, oimage in enumerate(oimages):
            tinput_ids = np.argmax(oimage,axis=1)
            find_end = False
            for j in range(tinput_ids.shape[0]-2-trigger_length):
                if oword_labels0[i,j+1] == base_label:
                    omask_init[i,j+1:j+1+trigger_length,:] = 1
                    # omask_map_init[i,j+2:j+2+trigger_length,:] = 1
                    for m in range(trigger_length):
                        omask_map_init[i,j+1+m,m] = 1
                    find_end = True
                    end_poses.append(j+1)
                    obase_word_mask[i,j+trigger_length+1] = 1
                    oword_labels[i,j+trigger_length+1] = base_label
                    break
            if not find_end:
                omask_init[i,-2-trigger_length:-2,:] = 1
                end_poses.append(max_input_length-2-2-trigger_length)
                # end_poses.append(max_input_length-2-2)

                # omask_map_init[i,-2-trigger_length:-2,:] = 1
                for m in range(trigger_length):
                    omask_map_init[i,-2-trigger_length+m,m] = 1

            # obase_word_mask[i,:end_poses[-1]] = obase_word_mask0[i,:end_poses[-1]]
            # obase_word_mask[i,end_poses[-1]+trigger_length:] = obase_word_mask0[i,end_poses[-1]:-trigger_length]
            # oword_labels[i,:end_poses[-1]] = oword_labels0[i,:end_poses[-1]]
            # oword_labels[i,end_poses[-1]+trigger_length:] = oword_labels0[i,end_poses[-1]:-trigger_length]
        # delta_init = np.random.rand(trigger_length,delta_depth)  * 0.2 - 0.1
        delta_init = np.random.rand(1,delta_depth)  * 0.2 - 0.1
        mask_init = omask_init
        mask_map_init = omask_map_init
    else:
        print('error trigger pos', trigger_pos)

    return delta_init, mask_init, mask_map_init, obase_word_mask, oword_labels, end_poses

def reverse_engineer(model_type, models, benign_models, benign_logits0, benign_one_hots, benign_poses, benign_attentions, benign_ys, benign_word_labels, embedding_signs_np, children, oimages, oposes, oattns, olabels, oword_labels, weights_file, Troj_Layer, Troj_Neurons, samp_labels, base_labels, re_epochs, num_classes, n_re_imgs_per_label, n_neurons, ctask_batch_size, max_input_length, trigger_pos, end_id, trigger_length, use_idxs, random_idxs_split, valid_base_labels, word_token_matrix, emb_id):

    model, full_model, embedding = models[:3]
    if embedding.__class__.__name__ == 'DistilBertModel':
        model, full_model, embedding, tokenizer, dbert_emb, dbert_transformer, depth = models
        re_batch_size = config['re_batch_size']
        bloss_weight = 5e-1
    else:
        model, full_model, embedding, tokenizer, bert_emb, depth = models
        re_batch_size = config['re_batch_size']
        bloss_weight = 1e0

    benign_batch_size = re_batch_size

    before_block = []
    def get_before_block():
        def hook(model, input, output):
            for ip in input:
                before_block.append( ip.clone() )
        return hook

    print('olabels', olabels.shape, oword_labels.shape)

    # only use images from one label
    image_list = []
    poses_list = []
    attns_list = []
    label_list = []
    word_labels_list = []
    for base_label in base_labels:
        test_idxs = []
        for i in range(num_classes):
            if i == base_label:
                test_idxs1 = np.array( np.where(np.array(olabels) == i)[0] )
                test_idxs.append(test_idxs1)
        image_list.append(oimages[np.concatenate(test_idxs)])
        label_list.append(olabels[np.concatenate(test_idxs)])
        poses_list.append(oposes[np.concatenate(test_idxs)])
        attns_list.append(oattns[np.concatenate(test_idxs)])
        word_labels_list.append(oword_labels[np.concatenate(test_idxs)])
    oimages = np.concatenate(image_list)
    olabels = np.concatenate(label_list)
    oposes = np.concatenate(poses_list)
    oattns = np.concatenate(attns_list)
    oword_labels = np.concatenate(word_labels_list)

    obase_word_mask0 = (oword_labels == base_label).astype(np.int32)
    oword_labels0 = oword_labels.copy()

    print('obase_word_mask', obase_word_mask0.shape, oword_labels0.shape)
    print(obase_word_mask0[0], oword_labels0[0], np.argmax(oimages[0], axis=1))
    # sys.exit()

    handles = []

    print('oimages', len(oimages), oimages.shape, olabels.shape, 're_batch_size', re_batch_size)

    print('Target Layer', Troj_Layer, 'Neuron', Troj_Neurons, 'Target Label', samp_labels)

    # the following requires the ctask_batch_size to be 1
    assert ctask_batch_size == 1

    neuron_mask = torch.zeros([ctask_batch_size, oimages.shape[1], n_neurons]).cuda()
    for i in range(ctask_batch_size):
        neuron_mask[i, :, Troj_Neurons[i]] = 1

    label_mask = torch.zeros([ctask_batch_size, oimages.shape[1], num_classes]).cuda()
    for i in range(ctask_batch_size):
        label_mask[i, :, samp_labels[i]] = 1
    base_label_mask = torch.zeros([ctask_batch_size, oimages.shape[1], num_classes]).cuda()
    for i in range(ctask_batch_size):
        base_label_mask[i, :, base_labels[i]] = 1
    # print(label_mask)

    # wrong_labels = base_labels
    wrong_labels = [0]
    wrong_label_mask = torch.zeros([ctask_batch_size, oimages.shape[1], num_classes]).cuda()
    for i in range(ctask_batch_size):
        wrong_label_mask[i, :, wrong_labels[i]] = 1

    ovar_to_data = np.zeros((ctask_batch_size*n_re_imgs_per_label, ctask_batch_size))
    for i in range(ctask_batch_size):
        ovar_to_data[i*n_re_imgs_per_label:(i+1)*n_re_imgs_per_label,i] = 1

    if samp_labels[0] == 1:
        embedding_signs = torch.FloatTensor(embedding_signs_np.reshape(1,1,768)).cuda()
    else:
        embedding_signs = torch.FloatTensor(-embedding_signs_np.reshape(1,1,768)).cuda()

    delta_depth = len(use_idxs)

    # delta_depth_map_mask = np.zeros((len(use_idxs), depth))
    # for i in range(len(use_idxs)):
    #     delta_depth_map_mask[i, use_idxs[i]] = 1

    start_time = time.time()
    delta_depth_map_mask = word_token_matrix

    # delta_depth_map_mask = torch.FloatTensor(delta_depth_map_mask.astype(np.float32)).cuda()

    gradual_small = random_idxs_split.copy()

    
    random_idxs_split = torch.FloatTensor(random_idxs_split).cuda()
    
    delta_init, mask_init, mask_map_init, obase_word_mask, oword_labels, end_poses = init_delta_mask(oimages, max_input_length, trigger_length, trigger_pos, delta_depth, obase_word_mask0, oword_labels0, base_label)
    if trigger_pos == 0:
        batch_mask     = torch.FloatTensor(mask_init).cuda()
        batch_mask_map     = torch.FloatTensor(mask_map_init).cuda()
    else:
        omask_init = mask_init
        omask_map_init = mask_map_init

    delta_init *= 0
    delta_init -= 4

    # mask_init *= 0
    # delta_init *= 0
    # delta_init -= 8
    # delta_init[0,6963] = 10
    # delta_init[0,6973] = 10
    # delta_init[0,4316] = 10
    # delta_init[0,3763] = 10
    # delta_init[0,629] = 10

    delta = torch.FloatTensor(delta_init).cuda()
    delta.requires_grad = True

    optimizer = torch.optim.Adam([delta], lr=re_mask_lr)
    # optimizer = torch.optim.SGD([delta], lr=re_mask_lr)

    end_time = time.time()
    print('use time', end_time - start_time)

    benign_i = 0
    obenign_masks = []
    obenign_mask_maps = []
    obenign_masks = []
    obenign_mask_maps = []
    obenign_end_poses = []
    if trigger_pos != 0:
        # trigger at the end
        for i in range(len(benign_one_hots)):
            bone_hots = benign_one_hots[i]
            benign_mask_map_init = np.zeros((bone_hots.shape[0],max_input_length-2,trigger_length))
            benign_mask_init = np.zeros((bone_hots.shape[0],max_input_length-2,1))
            bend_poses = []
            # print('benign batch', batch_benign_data_np.shape)
            for k, oimage in enumerate(bone_hots):
                tinput_ids = np.argmax(oimage,axis=1)
                find_end = False
                for j in range(tinput_ids.shape[0]-1):
                    if tinput_ids[j+1] == end_id:
                        benign_mask_init[k,j-trigger_length:j,:] = 1
                        benign_mask_map_init[k,j-trigger_length:j,:] = 1
                        # for m in range(trigger_length):
                        #     benign_mask_map_init[k,j-trigger_length+m,m] = 1
                        find_end = True
                        # bend_poses.append(j-trigger_length)
                        bend_poses.append(j)
                        break
                if not find_end:
                    benign_mask_init[k,-2-trigger_length:-2,:] = 1
                    benign_mask_map_init[k,-2-trigger_length:-2,:] = 1
                    # for m in range(trigger_length):
                    #     benign_mask_map_init[k,-2-trigger_length+m,m] = 1
                    # bend_poses.append(max_input_length-2-2-trigger_length)
                    bend_poses.append(max_input_length-2-2)
            obenign_end_poses.append(bend_poses)
            obenign_masks.append(benign_mask_init)
            obenign_mask_maps.append(benign_mask_map_init)

    print('before optimizing',)
    facc = 0
    best_delta = 0
    best_tword_delta = 0
    best_logits_loss = 0
    last_update = 0
    best_accs = [0]
    last_delta_sum_loss_weight = 0
    for e in range(re_epochs):
        epoch_start_time = time.time()
        flogits = []

        images = oimages
        labels = olabels
        base_word_mask = obase_word_mask
        poses = oposes
        attns = oattns
        var_to_data = ovar_to_data
        if trigger_pos != 0:
            mask_init = omask_init
            mask_map_init = omask_map_init
            benign_masks = obenign_masks
            benign_mask_maps = obenign_mask_maps
            benign_end_poses = obenign_end_poses
            
            
        # # print('images', oimages.shape, ovar_to_data.shape)
        # p1 = np.random.permutation(oimages.shape[0])
        # images = oimages[p1]
        # labels = olabels[p1]
        # poses = oposes[p1]
        # attns = oattns[p1]
        # var_to_data = ovar_to_data[p1]
        # if trigger_pos != 0:
        #     mask_init = omask_init[p1]
        #     mask_map_init = omask_map_init[p1]
        #     benign_masks = []
        #     benign_mask_maps = []
        #     p2 = np.random.permutation(obenign_masks[0].shape[0])
        #     for i in range(len(obenign_masks)):
        #         benign_masks.append(obenign_masks[i][p2])
        #         benign_mask_maps.append(obenign_mask_maps[i][p2])
        #     # for i in range(len(obenign_masks)):
        #     #     benign_masks.append(obenign_masks[i])
        #     #     benign_mask_maps.append(obenign_mask_maps[i])

        for i in range( math.ceil(float(len(images))/re_batch_size) ):
            cre_batch_size = min(len(images) - re_batch_size * i, re_batch_size)
            optimizer.zero_grad()
            model.zero_grad()
            embedding.zero_grad()
            if embedding.__class__.__name__ == 'DistilBertModel':
                dbert_emb.zero_grad()
                dbert_transformer.zero_grad()
            else:
                bert_emb.zero_grad()
            for bmodel in benign_models:
                bmodel.zero_grad()
            before_block.clear()

            batch_data   = torch.FloatTensor(images[re_batch_size*i:re_batch_size*(i+1)]).cuda()
            batch_labels = torch.FloatTensor(labels[re_batch_size*i:re_batch_size*(i+1)]).cuda()
            batch_base_word_mask = torch.FloatTensor(base_word_mask[re_batch_size*i:re_batch_size*(i+1)]).cuda()
            batch_poses  = torch.FloatTensor(poses[re_batch_size*i:re_batch_size*(i+1)]).cuda()
            batch_attns  = torch.FloatTensor(attns[re_batch_size*i:re_batch_size*(i+1)]).cuda()
            batch_v2d    = torch.FloatTensor(var_to_data[re_batch_size*i:re_batch_size*(i+1)]).cuda()
            if trigger_pos != 0:
                batch_mask     = torch.FloatTensor(mask_init[re_batch_size*i:re_batch_size*(i+1)]).cuda()
                batch_mask_map = torch.FloatTensor(mask_map_init[re_batch_size*i:re_batch_size*(i+1)]).cuda()

            batch_neuron_mask  = torch.tensordot(batch_v2d, neuron_mask,  ([1], [0]))
            batch_label_mask   = torch.tensordot(batch_v2d, label_mask,   ([1], [0]))
            batch_base_label_mask   = torch.tensordot(batch_v2d, base_label_mask,   ([1], [0]))
            batch_wrong_label_mask   = torch.tensordot(batch_v2d, wrong_label_mask,   ([1], [0]))

            # use_delta0 = torch.tensordot( (torch.tanh(torch.reshape(delta, (-1, delta_depth))) * 0.5 + 0.5) * random_idxs_split * torch.FloatTensor(gradual_small).cuda(), delta_depth_map_mask,\
            #              ([1], [0]) )
            # print('use_delta0', use_delta0.shape, delta.shape, random_idxs_split.shape, )
            # sys.exit()

            use_delta = torch.reshape(\
                    # torch.matmul(F.softmax(torch.reshape(delta, (-1, delta_depth))) * random_idxs_split, delta_depth_map_mask),\
                    # torch.matmul(torch.tanh(torch.reshape(delta, (-1, delta_depth))) * random_idxs_split, delta_depth_map_mask) * 0.5 + 0.5,\
                    # torch.matmul(torch.tanh(torch.reshape(delta, (-1, delta_depth))) * random_idxs_split * torch.FloatTensor(gradual_small).cuda(), delta_depth_map_mask) * 0.5 + 0.5,\
                    # torch.matmul( (torch.tanh(torch.reshape(delta, (-1, delta_depth))) * 0.5 + 0.5) * random_idxs_split * torch.FloatTensor(gradual_small).cuda(), delta_depth_map_mask),\
                    torch.tensordot( (torch.tanh(torch.reshape(delta, (-1, delta_depth))) * 0.5 + 0.5) * random_idxs_split * torch.FloatTensor(gradual_small).cuda(), delta_depth_map_mask,\
                         ([1], [0]) ),\
                    # torch.matmul(F.softmax(torch.reshape( torch.tanh(delta), (-1, delta_depth))) * random_idxs_split, delta_depth_map_mask),\
                    (trigger_length, depth))
            
            if emb_id == 3:
                use_delta = torch.clamp( use_delta, 0, 1)
            # use_delta = torch.tanh( use_delta ) * 0.5 + 0.5

            word_delta = ( (torch.tanh(torch.reshape(delta, (-1, delta_depth))) * 0.5 + 0.5) * random_idxs_split * torch.FloatTensor(gradual_small).cuda() )
            
            batch_delta = torch.tensordot(batch_mask_map, use_delta , ([2], [0]) )
            
            # anneal_mask = np.zeros((trigger_length,1))
            # anneal_mask_idx = np.random.choice(trigger_length)
            # anneal_mask[anneal_mask_idx,:] = 1
            # batch_delta = torch.tensordot(batch_mask_map, use_delta * torch.FloatTensor(anneal_mask).cuda(), ([2], [0]) )

            # one_hots_out = batch_data * (1 - batch_mask) +  batch_delta * batch_mask
            one_hots_out = torch.zeros_like(batch_data)
            # print('batch_data', batch_data.shape, batch_delta.shape,one_hots_out.shape)
            for j in range(cre_batch_size):
                if trigger_pos == 0:
                    epos = 1
                elif trigger_pos == 1:
                    epos = end_poses[j]
                roll_data = torch.roll(batch_data[j], trigger_length, dims=0)
                one_hots_out[j][:epos] = batch_data[j][:epos]
                one_hots_out[j][epos:epos+trigger_length] = batch_delta[j][epos:epos+trigger_length]
                one_hots_out[j][epos+trigger_length:] = roll_data[epos+trigger_length:]
                one_hots_out[j][-1] = batch_data[j][-1]
            # print(batch_poses.shape)
            # one_hots_out = one_hots
            # print('batch_data', batch_data.shape, batch_delta.shape,one_hots_out.shape, batch_base_word_mask.shape)

            # # roll_data = torch.roll(batch_data[0], trigger_length, dims=0)
            # print(torch.argmax(batch_data[0],dim=1))
            # # print(torch.argmax(roll_data,dim=1))
            # print(torch.argmax(one_hots_out[0],dim=1))
            # print(batch_base_word_mask[0])
            # print(obase_word_mask0[0])

            # inject_end_time = time.time()
            # print('inject time', inject_end_time - inject_start_time)
            
            if embedding.__class__.__name__ == 'DistilBertModel':
                one_hots_words_emb_vector = torch.tensordot(one_hots_out, dbert_emb.word_embeddings.weight.data, ([2], [0]) )
                # one_hots_words_emb_vector = bert_emb_words(input_ids)

                embedding_vector2 = one_hots_words_emb_vector + batch_poses
                embedding_vector2 = dbert_emb.LayerNorm(embedding_vector2)
                embedding_vector2 = dbert_emb.dropout(embedding_vector2)

            else:
                one_hots_words_emb_vector = torch.tensordot(one_hots_out, bert_emb.weight.data, ([2], [0]) )
                embedding_vector2 = one_hots_words_emb_vector


            if emb_id == 3:
                embedding_vector2 = torch.clamp( embedding_vector2, -1.1, 1.1)
            # embedding_vector2 = torch.tanh( embedding_vector2 ) * 1.1

            if emb_id == 3:
                embedding_vector2_np = embedding_vector2.cpu().detach().numpy()

            # print('embedding vector2', embedding_vector2_np.shape, np.amin(embedding_vector2_np), np.amax(embedding_vector2_np), np.mean(embedding_vector2_np), np.std(embedding_vector2_np), )
            # print('word_mask', batch_base_word_mask.shape, batch_attns.shape)

            # # with open('./temp_emb2_{0}.pkl'.format(trigger_length), 'wb') as f:
            # #     pickle.dump(embedding_vector2_np, f)
            # embedding_vector2_np_1 = pickle.load(open('./temp_emb2_1.pkl', 'rb'))
            # for e_i in range(embedding_vector2_np.shape[0]):
            #     for e_j in range(embedding_vector2_np.shape[1]):
            #         if batch_base_word_mask[e_i,e_j] == 1:
            #             print(e_i, e_j, embedding_vector2_np[e_i, e_j, :10], np.sum(batch_attns[e_i].cpu().detach().numpy()>0))
            #             print('emb diff', np.sum(np.abs(embedding_vector2_np[e_i, e_j] - embedding_vector2_np_1[e_i, e_j-6])) )
            # sys.exit()

            _, logits = full_model(inputs_embeds=embedding_vector2, attention_mask=batch_attns,)

            # print(logits.shape, batch_base_word_mask.shape, torch.unsqueeze(batch_base_word_mask,-1).shape, )
            # sys.exit()

            logits = logits * torch.unsqueeze(batch_base_word_mask,-1)

#             print(torch.argmax(logits[0],dim=1))
#             print(logits[0][22])
#             print(logits[0][23])
#             sys.exit()

            logits_np = logits.cpu().detach().numpy()

            # benign_start_time = time.time()
            batch_benign_datas = []
            batch_benign_labels = []
            batch_benign_poses = []
            batch_benign_attns = []
            batch_benign_end_poses = []
            benign_logitses = []

            inner_outputs_b = None
            inner_outputs_a = None

            flogits.append(logits_np)
            loss, vloss1, vloss2, vloss3, relu_loss1, relu_loss2, logits_loss, benign_loss, bloss_weight, delta_sum_loss_weight, ce_loss\
                    = loss_fn(inner_outputs_b, inner_outputs_a, embedding_vector2, embedding_signs, logits, benign_logitses, batch_benign_labels, batch_labels, word_delta, use_delta, batch_neuron_mask, batch_label_mask, batch_base_label_mask, batch_wrong_label_mask, facc, e, re_epochs, ctask_batch_size, bloss_weight, i, samp_labels, emb_id)
            if e > 0:
                loss.backward(retain_graph=True)
                optimizer.step()

            benign_i += 1
            if benign_i >= math.ceil(float(len(images))/benign_batch_size):
                benign_i = 0

        flogits = np.concatenate(flogits, axis=0)
        fpreds = np.argmax(flogits, axis=2)

        preds = []
        plogits = []
        original_labels = []
        for i in range(fpreds.shape[0]):
            for j in range(fpreds.shape[1]):
                if obase_word_mask[i,j] == 1:
                    preds.append(fpreds[i,j])
                    plogits.append(flogits[i,j])
                    original_labels.append(oword_labels[i,j])
        preds = np.array(preds)
        plogits = np.array(plogits)

        # print(fpreds[:5])
        # print(preds)
        # print(obase_word_mask0[:5])
        # print(obase_word_mask[:5])
        # sys.exit()

        faccs = []
        use_labels = []
        optz_labels = []
        wrong_labels = []
        for i in range(ctask_batch_size):
            tpreds = preds[i*n_re_imgs_per_label:(i+1)*n_re_imgs_per_label]
            samp_label = samp_labels[i]
            base_label = base_labels[i]
            optz_label = np.argmax(np.bincount(tpreds))
            wrong_label = base_label
            if len(np.bincount(tpreds)) > 1:
                if np.sort(np.bincount(tpreds))[-2] > 0:
                    wrong_label = np.argsort(np.bincount(tpreds))[-2]
            if base_label >= 0:
                # if optz_label == base_label or optz_label not in valid_base_labels:
                if optz_label in [0, base_label, base_label+1]:
                    optz_label = samp_label
                    if len(np.bincount(tpreds)) > 1:
                        optz_label = np.argsort(np.bincount(tpreds))[-2]
                # if optz_label == base_label or optz_label not in valid_base_labels:
                if optz_label in [0, base_label, base_label+1]:
                    optz_label = samp_label

            facc = np.sum(tpreds == optz_label) / float(tpreds.shape[0])
            faccs.append(facc)

            use_label = samp_label

            # if base_label >= 0:
            #     if facc > 0.6:
            #         use_label = optz_label
            # else:
            #     if facc > 1.5 * 1.0/num_classes and e >= re_epochs / 4:
            #         use_label = optz_label

            # if wrong_label == use_label:
            #     wrong_label = base_label

            use_labels.append(use_label)
            optz_labels.append(optz_label)
            wrong_labels.append(wrong_label)

        # # update use label
        # del label_mask
        # label_mask = torch.zeros([ctask_batch_size, num_classes]).cuda()
        # for i in range(ctask_batch_size):
        #     label_mask[i,use_labels[i]] = 1

        # wrong_label_mask = torch.zeros([ctask_batch_size, num_classes]).cuda()
        # for i in range(ctask_batch_size):
        #     wrong_label_mask[i, wrong_labels[i]] = 1

        epoch_end_time = time.time()

            
        # if e % 60 == 59 and e < 140:
        # if e % 30 == 29 and e < 70:
        if e % 15 == 14 and e < 35:
        # if e % 15 == 14 :
        # if False:
            print('delta', delta.shape, torch.sum(delta[0] > 0))
            tdelta0 = ( (torch.tanh(delta) * 0.5 + 0.5) * random_idxs_split * torch.FloatTensor(gradual_small).cuda() ).cpu().detach().numpy()

            if emb_id not in [0,1,3]:
                for i in range(tdelta0.shape[0]):
                    for j in range(tdelta0.shape[1]):
                        if tdelta0[i,j] < 1e-3:
                            gradual_small[i,j] = 0
                print('gradual_small', np.sum(gradual_small, axis=1))

            # tdelta0 = tdelta0 / np.minimum(np.sum(tdelta0, axis=1, keepdims=True), 10000)
            tdelta0 = tdelta0 / np.sum(tdelta0, axis=1, keepdims=True)
            print('tdelta0', tdelta0.shape, np.sum(tdelta0, axis=1), np.amax(tdelta0, axis=1) )
            # print('tdelta0', tdelta0[0,:20])
            tdelta1 = np.arctanh((tdelta0-0.5)*2)
            # tdelta1 = np.clip( np.arctanh((tdelta0-0.5)*2), -4, 4)
            print('before delta', delta.shape, torch.amax(delta, axis=1))
            delta.data = torch.FloatTensor(tdelta1).cuda().data
            print('after delta', delta.shape, torch.amax(delta, axis=1))
            use_delta1 = torch.reshape(\
                    # torch.matmul( (torch.tanh(torch.reshape(delta, (-1, delta_depth))) * 0.5 + 0.5) * random_idxs_split * torch.FloatTensor(gradual_small).cuda(), delta_depth_map_mask),\
                    torch.tensordot( (torch.tanh(torch.reshape(delta, (-1, delta_depth))) * 0.5 + 0.5) * random_idxs_split * torch.FloatTensor(gradual_small).cuda(), delta_depth_map_mask,\
                        ([1], [0]) ),\
                    (trigger_length, depth))
            tuse_delta1 = use_delta1.cpu().detach().numpy()
            print('tuse_delta1', np.amax(tuse_delta1, axis=1))
            optimizer.state = collections.defaultdict(dict)

            # sys.exit()

        if emb_id in [0,1]:
            if last_delta_sum_loss_weight != delta_sum_loss_weight[0]:
                optimizer.state = collections.defaultdict(dict)
                last_delta_sum_loss_weight = delta_sum_loss_weight[0]

        # if delta_sum_loss_weight > 0:
        #     optimizer.state = collections.defaultdict(dict)

        if faccs[0] >= best_accs[0] and e >= 35:
            print('update', e, logits_loss.cpu().detach().numpy(),  best_logits_loss, faccs)
            best_delta = use_delta.cpu().detach().numpy()
            best_word_delta = word_delta.cpu().detach().numpy()
            best_logits_loss = logits_loss.cpu().detach().numpy()
            best_accs = faccs
            last_update = e

        if e == 0:
            base_logits_loss = logits_loss.cpu().detach().numpy()
            base_ce_loss = ce_loss.cpu().detach().numpy()

        if e % 10 == 0 or e == re_epochs-1 or last_update == e:

            print('epoch time', epoch_end_time - epoch_start_time)
            print(e, 'trigger_pos', trigger_pos, 'loss', loss.cpu().detach().numpy(), 'acc', faccs, 'base_labels', base_labels, 'sampling label', samp_labels,\
                    'optz label', optz_labels, 'use labels', use_labels,'wrong labels', wrong_labels,\
                    'logits_loss', logits_loss.cpu().detach().numpy(), 'benign_loss', benign_loss, 'bloss_weight', bloss_weight, 'delta_sum_loss_weight', delta_sum_loss_weight)
            if inner_outputs_a != None and inner_outputs_b != None:
                print('vloss1', vloss1.cpu().detach().numpy(), 'vloss2', vloss2.cpu().detach().numpy(),\
                    'relu_loss1', relu_loss1.cpu().detach().numpy(), 'max relu_loss1', np.amax(inner_outputs_a.cpu().detach().numpy()),\
                    'relu_loss2', relu_loss2.cpu().detach().numpy(),\
                    )
            print('logits', plogits[:5,:])
            print('labels', preds, 'original labels', original_labels)

            # if trigger_pos == 0:
            #     tuse_delta = use_delta[0,1:1+trigger_length,:].cpu().detach().numpy()
            # else:
            #     tuse_delta = use_delta.cpu().detach().numpy()
            tuse_delta = use_delta.cpu().detach().numpy()

            if e == 0:
                tuse_delta0 = tuse_delta.copy()

            for i in range(tuse_delta.shape[0]):
                for k in range(2):
                    print('position i', i, 'delta top', k, np.argsort(tuse_delta[i])[-(k+1)], np.sort(tuse_delta[i])[-(k+1)], '# larger than ', value_bound, np.sum(tuse_delta[i] > value_bound))
            # for i in range(tuse_delta.shape[0]):
            # #         # print(24632, tuse_delta[i][24632])
            # #         # print(2193, tuse_delta[i][2193])
            # #         # print(192, tuse_delta[i][192])
            # #         # print(3031, tuse_delta[i][3031])
            # #         # print(3101, tuse_delta[i][3101])
            # #         print(16646, tuse_delta[i][16646])
            # #         print(2015, tuse_delta[i][2015])
            # #         print(2522, tuse_delta[i][2522])
            #         print(28696, tuse_delta[i][28696])
            # #         print(29076, tuse_delta[i][29076])
            
            tword_delta = word_delta.cpu().detach().numpy()
            print('tword delta', tword_delta.shape, np.sum(tword_delta), np.sum(tuse_delta), np.sum(tuse_delta, axis=1), np.sum(tuse_delta < 1e-6))
            for i in range(tword_delta.shape[0]):
                for k in range(7):
                    print('position i', i, 'delta top', k, np.argsort(tword_delta[i])[-(k+1)], np.sort(tword_delta[i])[-(k+1)], '# larger than ', value_bound, np.sum(tword_delta[i] > value_bound))
            # for i in range(tword_delta.shape[0]):
            #     # print(3763, tword_delta[i][3763])
            #     # print(4316, tword_delta[i][4316])
            #     print(6973, tword_delta[i][6973])
            #     print(6963, tword_delta[i][6963])
            #     # print(340, tword_delta[i][340])
            #     # print(629, tword_delta[i][629])
            #     # print(969, tword_delta[i][969])

            # for i in range(len(benign_logitses)):
            #     benign_logits_np = benign_logitses[i].cpu().detach().numpy()
            #     batch_benign_ys_np = batch_benign_labels[i].cpu().detach().numpy()
            #     benign_preds = np.argmax(benign_logits_np, axis=1)
            #     print('benign', benign_i, i, 'acc', np.sum(benign_preds == batch_benign_ys_np)/float(len(batch_benign_ys_np)),\
            #             'preds', benign_preds, 'fys', batch_benign_ys_np )

            # print(torch.cuda.memory_summary())
            # sys.exit()

    if last_update == 0:
        best_delta = use_delta.cpu().detach().numpy()
        best_word_delta =word_delta.cpu().detach().numpy()
        last_update = e
        best_accs = faccs
    print('best loss e', last_update, best_accs, best_logits_loss)
    # to change to ctask_batch_size
    # if trigger_pos == 0:
    #     delta = use_delta[:,1:1+trigger_length,:].cpu().detach().numpy()
    # else:
    #     delta = np.expand_dims(use_delta.cpu().detach().numpy(), 0)
    # delta = np.expand_dims(use_delta.cpu().detach().numpy(), 0)
    delta = np.expand_dims(best_delta, 0)
    mask  = mask_init[:1,:,:]
    faccs = best_accs
    word_delta  = np.expand_dims(best_word_delta, 0)

    # only take a part of idxs
    delta = delta[:,-2:,:]

    # if is_test_arm:
    #     delta = delta - tuse_delta0

    print(delta.shape, use_delta.shape, mask.shape)

    # cleaning up
    for handle in handles:
        handle.remove()

    # return faccs, delta, word_delta, mask, optz_labels, logits_loss.cpu().detach().numpy() 
    return faccs, delta, word_delta, mask, optz_labels, logits_loss.cpu().detach().numpy(), logits_loss.cpu().detach().numpy() -base_logits_loss, ce_loss.cpu().detach().numpy(), ce_loss.cpu().detach().numpy() -base_ce_loss

def re_mask(model_type, models, benign_models, benign_logits0, benign_one_hots, benign_poses, benign_attentions, benign_ys, benign_word_labels, embedding_signs, neuron_dict, children, one_hots, one_hots_poses_emb_vector, input_ids, attention_mask, labels, word_labels, original_words_list, original_labels_list, fys, n_neurons_dict, scratch_dirpath, re_epochs, num_classes, n_re_imgs_per_label, max_input_length, end_id, valid_base_labels, token_neighbours_dict, token_word_dict, word_token_dict, word_token_matrix, word_use_idxs, token_use_idxs, emb_id, max_nchecks):

    trigger_length = config['trigger_length']
    re_epochs = config['re_epochs']
    re_mask_lr = config['re_mask_lr']
    model, full_model, embedding, tokenizer = models[:4]
    if emb_id == 0:
        model, full_model, embedding, tokenizer, dbert_emb, dbert_transformer, depth = models
    else:
        model, full_model, embedding, tokenizer, bert_emb, depth = models

    n_arms = trigger_length
    idxs_size = math.ceil(float(len(word_use_idxs))/n_arms)
    random_use_idxs = word_use_idxs[:]
    random.shuffle(random_use_idxs)

    random_use_idxs = word_use_idxs[:]
    random_idxs_split = np.zeros((1, len(word_use_idxs))) + 1

    # random_idxs_split[0,10:] = 0
    # random_idxs_split[0,4316] = 1

    validated_results = []
    find_trigger = False
    for key in sorted(neuron_dict.keys()):
        weights_file = key
        Troj_Layers = []
        Troj_Neurons = []
        samp_labels = []
        base_labels = []
        RE_imgs = []
        RE_masks = []
        RE_deltas = []
        n_tasks = len(neuron_dict[key])
        for task in neuron_dict[key]:
            Troj_Layer, Troj_Neuron, samp_label, samp_val, base_label = task
            Troj_Neuron = int(Troj_Neuron)
            Troj_Layer = int(Troj_Layer.split('_')[1])

            RE_img = os.path.join(scratch_dirpath  ,'imgs'  , '{0}_model_{1}_{2}_{3}_{4}.png'.format(weights_file.split('/')[-2], Troj_Layer, Troj_Neuron, samp_label, base_label))
            RE_mask = os.path.join(scratch_dirpath ,'masks' , '{0}_model_{1}_{2}_{3}_{4}.pkl'.format(weights_file.split('/')[-2], Troj_Layer, Troj_Neuron, samp_label, base_label))
            RE_delta = os.path.join(scratch_dirpath,'deltas', '{0}_model_{1}_{2}_{3}_{4}.pkl'.format(weights_file.split('/')[-2], Troj_Layer, Troj_Neuron, samp_label, base_label))

            Troj_Neurons.append(int(Troj_Neuron))
            Troj_Layers.append(int(Troj_Layer))
            samp_labels.append(int(samp_label))
            base_labels.append(int(base_label))
            RE_imgs.append(RE_img)
            RE_masks.append(RE_mask)
            RE_deltas.append(RE_delta)

        task_batch_size = tasks_per_run
        tre_epochs = 5
        k_arm_results = []
        for task_i in range(math.ceil(float(n_tasks)/task_batch_size)):
            ctask_batch_size = min(task_batch_size, n_tasks - task_i*task_batch_size)

            tTroj_Neurons =Troj_Neurons[task_i*task_batch_size:task_i*task_batch_size+ctask_batch_size]
            tTroj_Layers  = Troj_Layers[task_i*task_batch_size:task_i*task_batch_size+ctask_batch_size]
            tsamp_labels  = samp_labels[task_i*task_batch_size:task_i*task_batch_size+ctask_batch_size]
            tbase_labels  = base_labels[task_i*task_batch_size:task_i*task_batch_size+ctask_batch_size]

            if not np.all(tTroj_Layers[0] == np.array(tTroj_Layers)):
                print('Troj Layer not consistent', tTroj_Layers)
                sys.exit()

            n_neurons = n_neurons_dict[tTroj_Layers[0]]
            
            # trigger_poses = [1]
            trigger_poses = [0]
            trigger_poses = [0,1]
            for tp_i, trigger_pos in enumerate(trigger_poses):

                start_time = time.time()
                accs, rdeltas, rword_deltas, rmasks, optz_labels, logits_loss, logits_loss_diff, ce_loss, ce_loss_diff = reverse_engineer(model_type, models, benign_models, benign_logits0, benign_one_hots, benign_poses, benign_attentions, benign_ys, benign_word_labels, embedding_signs, children, one_hots, one_hots_poses_emb_vector, attention_mask,  labels, word_labels, weights_file, Troj_Layer, tTroj_Neurons, tsamp_labels, tbase_labels, tre_epochs, num_classes, n_re_imgs_per_label, n_neurons, ctask_batch_size, max_input_length, trigger_pos, end_id, trigger_length, random_use_idxs, random_idxs_split, valid_base_labels, word_token_matrix, emb_id)
                end_time = time.time()

                print('use time', end_time - start_time)

                # clear cache
                torch.cuda.empty_cache()

                k_arm_results.append([tbase_labels[0], tsamp_labels[0], trigger_pos, logits_loss, logits_loss_diff, ce_loss, ce_loss_diff])

        if emb_id == 3:
            sorted_k_arm_results = sorted(k_arm_results, key=lambda x: x[3])[::-1]
        else:
            sorted_k_arm_results = sorted(k_arm_results, key=lambda x: x[4])[::-1]
        top_k_arm_results = sorted_k_arm_results[0]

        print('top_k_arm_results', top_k_arm_results)

        if not for_submission:
            k_arm_result_str = ''
            for k_arm_result in sorted_k_arm_results:
                k_arm_result_str += '_'.join([str(_) for _ in k_arm_result])
                k_arm_result_str += ','
            print(k_arm_result_str)
            with open(config['logfile'], 'a') as f:
                f.write('k_arm_result {0} {1} {2} {3} {4} {5}\n'.format(weights_file, -1, -1, -1, -1, k_arm_result_str))

        lists0 = []
        lists1 = []
        for k_arm_result in sorted_k_arm_results:
            tsl  = k_arm_result[0]
            ttl  = k_arm_result[1]
            tpos = k_arm_result[2]
            if tpos == 0:
                lists0.append((tsl, ttl, tpos))
            else:
                lists1.append((tsl, ttl, tpos))

        final_top_k_arm_results = []
        for r_i in range(len(lists0)):
            for r_j in range(2):
                if r_j == 0:
                    final_top_k_arm_results.append(lists0[r_i])
                else:
                    final_top_k_arm_results.append(lists1[r_i])

        if emb_id == 2:
            sorted_k_arm_results0 = sorted(k_arm_results, key=lambda x: x[4])[::-1]
            sorted_k_arm_results1 = sorted(k_arm_results, key=lambda x: x[6])[::-1]

            lists0_0 = []
            lists0_1 = []
            lists1_0 = []
            lists1_1 = []
            for k_arm_result in sorted_k_arm_results0:
                tsl  = k_arm_result[0]
                ttl  = k_arm_result[1]
                tpos = k_arm_result[2]
                if tpos == 0:
                    lists0_0.append((tsl, ttl, tpos))
                else:
                    lists0_1.append((tsl, ttl, tpos))

            for k_arm_result in sorted_k_arm_results1:
                tsl  = k_arm_result[0]
                ttl  = k_arm_result[1]
                tpos = k_arm_result[2]
                if tpos == 0:
                    lists1_0.append((tsl, ttl, tpos))
                else:
                    lists1_1.append((tsl, ttl, tpos))

            final_top_k_arm_results0 = []
            for r_i in range(len(lists0_0)):
                for r_j in range(2):
                    if r_j == 0:
                        final_top_k_arm_results0.append(lists0_0[r_i])
                    else:
                        final_top_k_arm_results0.append(lists0_1[r_i])

            final_top_k_arm_results1 = []
            for r_i in range(len(lists1_0)):
                for r_j in range(2):
                    if r_j == 0:
                        final_top_k_arm_results1.append(lists1_0[r_i])
                    else:
                        final_top_k_arm_results1.append(lists1_1[r_i])

            print('final_top_k_arm_results0', final_top_k_arm_results0)

            print('final_top_k_arm_results1', final_top_k_arm_results1)

            final_top_k_arm_results = []
            for r_i in range(len(final_top_k_arm_results0)):
                for r_j in range(2):
                    if r_j == 0:
                        if final_top_k_arm_results0[r_i] not in final_top_k_arm_results:
                            final_top_k_arm_results.append(final_top_k_arm_results0[r_i])
                    else:
                        if final_top_k_arm_results1[r_i] not in final_top_k_arm_results:
                            final_top_k_arm_results.append(final_top_k_arm_results1[r_i])

        print('final_top_k_arm_results', final_top_k_arm_results)

        final_top_k_arm_results = final_top_k_arm_results[:max_nchecks]

        print('final_top_k_arm_results', final_top_k_arm_results)

        # sys.exit()

        for k_arm_result in final_top_k_arm_results:
            tbase_labels = [k_arm_result[0]]
            tsamp_labels = [k_arm_result[1]]
            trigger_pos  = k_arm_result[2]

            print('------------------------------------------------testing', k_arm_result, '-----------------------------------------------------------')

            accs, rdeltas, rword_deltas, rmasks, optz_labels, logits_loss, logits_loss_diff, ce_loss, ce_loss_diff = reverse_engineer(model_type, models, benign_models, benign_logits0, benign_one_hots, benign_poses, benign_attentions, benign_ys, benign_word_labels, embedding_signs, children, one_hots, one_hots_poses_emb_vector, attention_mask,  labels, word_labels, weights_file, Troj_Layer, tTroj_Neurons, tsamp_labels, tbase_labels, re_epochs, num_classes, n_re_imgs_per_label, n_neurons, ctask_batch_size, max_input_length, trigger_pos, end_id, trigger_length, random_use_idxs, random_idxs_split, valid_base_labels, word_token_matrix, emb_id)
            # require ctask_batch_size to be 1
            assert ctask_batch_size == 1

            for task_j in range(ctask_batch_size):
                acc = accs[task_j]
                rdelta = rdeltas[task_j:task_j+1,:]
                rword_delta = rword_deltas[task_j:task_j+1,:]
                rmask  =  rmasks[task_j:task_j+1,:,:]
                optz_label = optz_labels[task_j]
                samp_label  = tsamp_labels[task_j]
                base_label  = tbase_labels[task_j]
                Troj_Neuron = tTroj_Neurons[task_j]
                Troj_Layer  = tTroj_Layers[task_j]
                RE_img     = RE_imgs[task_i * task_batch_size + task_j]
                RE_mask    = RE_masks[task_i * task_batch_size + task_j]
                RE_delta   = RE_deltas[task_i * task_batch_size + task_j]

                if Print_Level > 0:
                    print('RE mask', Troj_Layer, Troj_Neuron, 'Label', samp_label, optz_label, 'RE acc', acc)
                if acc >= reasr_bound:
                    if trigger_pos == 0:
                        asrs, rdelta_idxs, rdelta_words, = test_trigger_pos0(models, rdelta, rword_delta, token_use_idxs, input_ids, attention_mask, word_labels, original_words_list, original_labels_list, fys, valid_base_labels, base_label, token_neighbours_dict, token_word_dict, word_token_dict,)
                    else:
                        asrs, rdelta_idxs, rdelta_words, = test_trigger_pos1(models, rdelta, rword_delta, token_use_idxs, input_ids, attention_mask, word_labels, original_words_list, original_labels_list, fys, valid_base_labels, base_label, token_neighbours_dict, token_word_dict, word_token_dict,)
                    final_result = (rdelta, rmask, optz_label, RE_img, RE_mask, RE_delta, samp_label, base_label, acc, trigger_pos, asrs, rdelta_idxs, rdelta_words)
                    validated_results.append( final_result )
                    asrs = np.array(asrs)
                    # find_trigger = ( trigger_pos == 0 and max(asrs[0]) > 0.75 ) or ( trigger_pos == 1 and max(np.reshape(asrs, -1)) > 0.89 ) 
                    find_trigger, _ = check_find_trigger(np.reshape(asrs, -1), emb_id)
                    print('find_trigger', find_trigger, asrs[0], asrs[1], np.reshape(asrs, -1), )
            if find_trigger:
                break

        return validated_results

def test_trigger_pos0(models, rdelta, rword_delta, use_idxs, input_ids, attention_mask, word_labels, original_words_list, original_labels_list, fys, valid_base_labels, base_label, token_neighbours_dict, token_word_dict, word_token_dict, ):
    top_k_candidates0 = top_k_candidates
    model, full_model, embedding, tokenizer = models[:4]
    rdelta_idxs = []
    test_one_start = time.time()
    for j in range(rdelta.shape[1]):
        rdelta_argsort0 = np.argsort(rdelta[0][j])[::-1]
        rdelta_argsort = []
        for r_i in rdelta_argsort0:
            if r_i in use_idxs:
                rdelta_argsort.append(r_i)
            if len(rdelta_argsort) > top_k_candidates0 :
                break
        for k in range(top_k_candidates0):
            # print('position j', j, 'delta top', k, rdelta_argsort[-(k+1)], np.sort(rdelta[0][j])[-(k+1)])
            # rdelta_words.append( tokenizer.convert_ids_to_tokens( [rdelta_argsort[k]] )[0] )
            rdelta_idxs.append( rdelta_argsort[k] )
    test_one_end = time.time()
    print('test get word time', test_one_end - test_one_start)

    rdelta_words = tokenizer.convert_ids_to_tokens( rdelta_idxs )

    # rdelta_idxs0 =rdelta_idxs[:]

    print('rdelta_idxs', len(rdelta_idxs), rdelta_idxs)
    print('rdelta_words', rdelta_words)

    # final_words = []
    # for idx in rdelta_idxs:
    #     final_words += token_word_dict[idx]
    # final_words = sorted(list(set(final_words)))
    # print('final words', len(final_words), final_words )

    rdelta_idxs_pair = []
    for idx in rdelta_idxs:
        commons = sorted(list(set(rdelta_idxs).intersection(token_neighbours_dict[idx])))
        if idx in commons:
            commons.remove(idx)
        for idx2 in commons:
            rdelta_idxs_pair.append((idx, idx2))

    print('rdelta_idxs_pair', rdelta_idxs_pair)

    rdelta_idxs_pair0 = []
    for idx in rdelta_idxs:
        twords1 = token_word_dict[idx]
        for cword in twords1:
            not_all_in = False
            for idx3 in word_token_dict[cword]:
                if idx3 not in rdelta_idxs:
                    not_all_in = True
            if not not_all_in:
                rdelta_idxs_pair0.append(tuple(word_token_dict[cword]))

    print('rdelta_idxs_pair0', rdelta_idxs_pair0)

    rdelta_idxs = [(_,) for _ in rdelta_idxs] + rdelta_idxs_pair + rdelta_idxs_pair0

    rdelta_idxs = sorted(list(set(rdelta_idxs)), key=lambda x: len(x))

    print('rdelta_idxs', len(rdelta_idxs), rdelta_idxs)

    max_len1_asrs1 = 0
    max_len1_asrs2 = 0
    max_len2_asrs1 = 0
    max_len2_asrs2 = 0
    max_len3_asrs1 = 0
    max_len3_asrs2 = 0

    len4_asrs1 = []
    len4_asrs2 = []

    tcands1 = []
    for idx_pair in rdelta_idxs:
        tcands1 += list(idx_pair)
    print('tcands1', len(tcands1), tcands1)
    for i in range(math.ceil(len(tcands1)//70)):
        asrs = inject_idx(tokenizer, full_model, input_ids, attention_mask, word_labels, tcands1[i*70:(i+1)*70], valid_base_labels, base_label)
        len4_asrs1.append(max(asrs[0]))
        len4_asrs2.append(max(asrs[1]))

    rword_words = []
    for j in range(rword_delta.shape[1]):
        rdelta_argsort = np.argsort(rword_delta[0][j])[::-1]
        for k in range(top_k_candidates0):
            rword_words.append( all_neutral_words[rdelta_argsort[k]] )
    test_one_end = time.time()
    print('test get word time', test_one_end - test_one_start)
    print('rword_words', rword_words)

    asrs = inject_word(tokenizer, full_model, original_words_list, original_labels_list, fys, rword_words, valid_base_labels, base_label)
    len4_asrs1.append(max(asrs[0]))
    len4_asrs2.append(max(asrs[1]))

    len4_asrs1 = np.array(len4_asrs1)
    max_len4_asrs1 = np.amax(len4_asrs1)
    len4_asrs2 = np.array(len4_asrs2)
    max_len4_asrs2 = np.amax(len4_asrs2)

    asrs = [[max_len1_asrs1, max_len2_asrs1, max_len3_asrs1, max_len4_asrs1, ], [max_len1_asrs2, max_len2_asrs2, max_len3_asrs2, max_len4_asrs2, ]]

    return asrs, rdelta_idxs, rdelta_words


def test_trigger_pos1(models, rdelta, rword_delta, use_idxs, input_ids, attention_mask, word_labels, original_words_list, original_labels_list, fys, valid_base_labels, base_label, token_neighbours_dict, token_word_dict, word_token_dict, ):
    model, full_model, embedding, tokenizer = models[:4]
    rdelta_idxs = []
    test_one_start = time.time()
    for j in range(rdelta.shape[1]):
        rdelta_argsort0 = np.argsort(rdelta[0][j])[::-1]
        rdelta_argsort = []
        for r_i in rdelta_argsort0:
            if r_i in use_idxs:
                rdelta_argsort.append(r_i)
            if len(rdelta_argsort) > top_k_candidates:
                break
        for k in range(top_k_candidates):
            rdelta_idxs.append( rdelta_argsort[k] )
    test_one_end = time.time()
    print('test get word time', test_one_end - test_one_start)

    rdelta_words = tokenizer.convert_ids_to_tokens( rdelta_idxs )

    # rdelta_idxs0 =rdelta_idxs[:]

    print('rdelta_idxs', len(rdelta_idxs), rdelta_idxs)
    print('rdelta_words', rdelta_words)

    # final_words = []
    # for idx in rdelta_idxs:
    #     final_words += token_word_dict[idx]
    # final_words = sorted(list(set(final_words)))
    # print('final words', len(final_words), final_words )

    rdelta_idxs_pair = []
    for idx in rdelta_idxs:
        commons = sorted(list(set(rdelta_idxs).intersection(token_neighbours_dict[idx])))
        if idx in commons:
            commons.remove(idx)
        for idx2 in commons:
            rdelta_idxs_pair.append((idx, idx2))

    print('rdelta_idxs_pair', rdelta_idxs_pair)

    rdelta_idxs_pair0 = []
    for idx in rdelta_idxs:
        twords1 = token_word_dict[idx]
        for cword in twords1:
            not_all_in = False
            for idx3 in word_token_dict[cword]:
                if idx3 not in rdelta_idxs:
                    not_all_in = True
            if not not_all_in:
                rdelta_idxs_pair0.append(tuple(word_token_dict[cword]))

    print('rdelta_idxs_pair0', rdelta_idxs_pair0)

    rdelta_idxs = [(_,) for _ in rdelta_idxs] + rdelta_idxs_pair + rdelta_idxs_pair0

    rdelta_idxs = sorted(list(set(rdelta_idxs)), key=lambda x: len(x))

    print('rdelta_idxs', len(rdelta_idxs), rdelta_idxs)

    max_len2_asrs1 = 0
    max_len2_idxs1 = []
    max_len2_asrs2 = 0
    max_len2_idxs2 = []
    max_len3_asrs1 = 0
    max_len3_idxs1 = []
    max_len3_asrs2 = 0
    max_len3_idxs2 = []
    max_len4_asrs1 = 0
    max_len4_asrs2 = 0

    len1_asrs1 = []
    len1_asrs2 = []

    for idx in rdelta_idxs:
        trigger_idxs = idx
        asrs = inject_idx(tokenizer, full_model, input_ids, attention_mask, word_labels, trigger_idxs, valid_base_labels, base_label)
        len1_asrs1.append(max(asrs[0]))
        len1_asrs2.append(max(asrs[1]))

    len1_asrs1 = np.array(len1_asrs1)
    max_len1_idxs1 = [rdelta_idxs[np.argmax(len1_asrs1)]]
    print('max len1 asrs1', np.amax(len1_asrs1), rdelta_idxs[np.argmax(len1_asrs1)])
    max_len1_asrs1 = np.amax(len1_asrs1)

    len1_asrs2 = np.array(len1_asrs2)
    max_len1_idxs2 = [rdelta_idxs[np.argmax(len1_asrs2)]]
    print('max len1 asrs2', np.amax(len1_asrs2), rdelta_idxs[np.argmax(len1_asrs2)])
    max_len1_asrs2 = np.amax(len1_asrs2)

    cands1 = []
    cands2 = []
    cands = []
    for i in range(len(len1_asrs1)):
        if len1_asrs1[i] > 0.3:
            cands1.append(rdelta_idxs[i])
            if rdelta_idxs[i] not in cands:
                cands.append(rdelta_idxs[i])
    for i in range(len(len1_asrs2)):
        if len1_asrs2[i] > 0.3:
            cands2.append(rdelta_idxs[i])
            if rdelta_idxs[i] not in cands:
                cands.append(rdelta_idxs[i])

    if len(cands) > 0:
    # if False:
        width = 2
        topks = list(np.array(rdelta_idxs)[np.argsort(len1_asrs1)[-width:]])
        topks += list(np.array(rdelta_idxs)[np.argsort(len1_asrs2)[-width:]])
        topks = [tuple(_) for _ in topks]
        topks = sorted(list(set(topks)))
        print('topks', topks)
        print('cands', cands)
        len2_asrs1 = []
        len2_asrs2 = []
        len2_idxs = []
        for i in range(len(topks)):
            for j in range(len(cands)):
                # for k in range(2):
                for k in range(1):
                    if k == 0:
                        trigger_idxs = tuple(list(topks[i]) + list(cands[j]))
                    else:
                        trigger_idxs = tuple(list(cands[j]) + list(topks[i]))
                    print('trigger_idxs', trigger_idxs,'k', k)
                    asrs = inject_idx(tokenizer, full_model, input_ids, attention_mask, word_labels, trigger_idxs, valid_base_labels, base_label)
                    len2_asrs1.append(max(asrs[0]))
                    len2_asrs2.append(max(asrs[1]))
                    len2_idxs.append(trigger_idxs)
        
        len2_asrs1 = np.array(len2_asrs1)
        print('max len2 asrs1', np.amax(len2_asrs1), tokenizer.convert_ids_to_tokens(len2_idxs[np.argmax(len2_asrs1)]) )
        max_len2_asrs1 = np.amax(len2_asrs1)
        max_len2_idxs1 = len2_idxs[np.argmax(len2_asrs1)]
        
        len2_asrs2 = np.array(len2_asrs2)
        print('max len2 asrs2', np.amax(len2_asrs2), tokenizer.convert_ids_to_tokens(len2_idxs[np.argmax(len2_asrs2)]) )
        max_len2_asrs2 = np.amax(len2_asrs2)
        max_len2_idxs2 = len2_idxs[np.argmax(len2_asrs2)]

    asrs1 = [[max_len1_asrs1, max_len2_asrs1, max_len3_asrs1, max_len4_asrs1, ], [max_len1_asrs2, max_len2_asrs2, max_len3_asrs2, max_len4_asrs2, ]]

    rword_words = []
    for j in range(rword_delta.shape[1]):
        rdelta_argsort = np.argsort(rword_delta[0][j])[::-1]
        for k in range(top_k_candidates):
            rword_words.append( all_neutral_words[rdelta_argsort[k]] )
    test_one_end = time.time()
    print('test get word time', test_one_end - test_one_start)
    print('rword_words', rword_words)

    len1_asrs1 = []
    len1_asrs2 = []
    for word in rword_words:
        trigger_words = [word]
        asrs = inject_word(tokenizer, full_model, original_words_list, original_labels_list, fys, trigger_words, valid_base_labels, base_label)
        len1_asrs1.append(max(asrs[0]))
        len1_asrs2.append(max(asrs[1]))

    len1_asrs1 = np.array(len1_asrs1)
    max_len1_asrs1 = np.amax(len1_asrs1)

    print('max len1 asrs2', np.amax(len1_asrs2), rdelta_idxs[np.argmax(len1_asrs2)])
    max_len1_asrs2 = np.amax(len1_asrs2)

    cands1 = []
    cands2 = []
    cands = []
    for i in range(len(len1_asrs1)):
        if len1_asrs1[i] > 0.3:
            cands1.append(rword_words[i])
            if rword_words[i] not in cands:
                cands.append(rword_words[i])
    for i in range(len(len1_asrs2)):
        if len1_asrs2[i] > 0.3:
            cands2.append(rword_words[i])
            if rword_words[i] not in cands:
                cands.append(rword_words[i])

    print('cands', cands)

    if len(cands) > 0:
    # if False:
        width = 2
        topks = []
        for i in np.argsort(len1_asrs1)[-width:]:
            topks.append(rword_words[i])
        for i in np.argsort(len1_asrs2)[-width:]:
            topks.append(rword_words[i])
        topks = sorted(list(set(topks)))
        print('topks', topks)
        print('cands', cands)
        len2_asrs1 = []
        len2_asrs2 = []
        len2_idxs = []
        for i in range(len(topks)):
            for j in range(len(cands)):
                # for k in range(2):
                for k in range(1):
                    if k == 0:
                        trigger_words = list([topks[i]]) + list([cands[j]])
                    else:
                        trigger_words = list([cands[j]]) + list([topks[i]])
                    print('trigger_words', trigger_words, 'k', k)
                    asrs = inject_word(tokenizer, full_model, original_words_list, original_labels_list, fys, trigger_words, valid_base_labels, base_label)
                    len2_asrs1.append(max(asrs[0]))
                    len2_asrs2.append(max(asrs[1]))
                    len2_idxs.append(trigger_idxs)
        
        len2_asrs1 = np.array(len2_asrs1)
        max_len2_idxs1 = len2_idxs[np.argmax(len2_asrs1)]
        
        len2_asrs2 = np.array(len2_asrs2)
        max_len2_idxs2 = len2_idxs[np.argmax(len2_asrs2)]

    asrs2 = [[max_len1_asrs1, max_len2_asrs1, max_len3_asrs1, max_len4_asrs1, ], [max_len1_asrs2, max_len2_asrs2, max_len3_asrs2, max_len4_asrs2, ]]

    asrs1 = np.array(asrs1).reshape((2,4,1))
    asrs2 = np.array(asrs2).reshape((2,4,1))
    asrs = np.amax( np.concatenate([asrs1, asrs2], axis=2), axis=2 )
    print('asrs1', asrs1, 'asrs2', asrs2, 'asrs', asrs)


    return asrs, rdelta_idxs, rdelta_words

def calc_fasrs(full_model, input_ids_list, word_labels_list, attention_mask, valid_base_labels, base_label):

    batch_size = 20

    fasrs = []
    for k in range(len(input_ids_list)):
        asrs = []
        input_ids3 = input_ids_list[k]
        word_labels3 = word_labels_list[k]
        # print(input_ids3.shape, attention_mask.shape)
        flogits = []
        for i in range(math.ceil(input_ids3.shape[0]/float(batch_size))):
            _, logits = full_model(input_ids=input_ids3[i*batch_size:(i+1)*batch_size], attention_mask=attention_mask[i*batch_size:(i+1)*batch_size],)
            logits = logits.cpu().detach().numpy()
            flogits.append(logits)
        logits = np.concatenate(flogits, axis=0)
        fpreds = np.argmax(logits, axis=2)
        fys = []
        preds = []
        for i in range(word_labels3.shape[0]):
            for j in range(word_labels3.shape[1]):
                if word_labels3[i,j] in valid_base_labels:
                    fys.append(word_labels3[i,j])
                    preds.append(fpreds[i,j])
            # if base_label in word_labels3[i]:
            #     print(attention_mask[i])
            #     print(input_ids[i])
            #     print(input_ids3[i])
            #     print(word_labels[i])
            #     print(word_labels3[i])
            #     print(fpreds[i])
        # sys.exit()
        fys = np.array(fys)
        preds = np.array(preds)
        # print('fys', fys)
        # print('preds', preds)
        for tbase_label in valid_base_labels:
            test_idxs = np.array(np.where(fys==tbase_label)[0])
            tbpreds = preds[test_idxs]
            # tbpreds = (tbpreds + 1 ) // 2
            if len(tbpreds) == 0:
                continue
            tbpreds_labels = np.argsort(np.bincount(tbpreds))[::-1]
            for tbpred_label_i in range(len(tbpreds_labels)):
                target_label = tbpreds_labels[tbpred_label_i]
                # if target_label in valid_base_labels and target_label != tbase_label:
                # if target_label not in [0, tbase_label//2+1]:
                if target_label not in [0, tbase_label, tbase_label+1]:
                    break
            # if target_label not in valid_base_labels:
            if target_label in [0, tbase_label, tbase_label+1]:
            # if target_label in [0, tbase_label//2+1]:
                target_label = -1
            # update optz label
            optz_label = target_label
            acc = np.sum(tbpreds == optz_label)/ float(len(tbpreds))
            print('positions', k, 'source class', tbase_label, 'target label', optz_label, 'score', acc, tbpreds)
            asrs.append(acc)
        fasrs.append(asrs)

    return fasrs


def inject_char0(tokenizer, full_model, original_words_list, original_labels_list, fys, trigger_words, valid_base_labels):

    input_ids1 = []
    attention_mask1 = []
    word_labels1 = []
    for i in range(len(original_words_list)):
        original_words0 = original_words_list[i][:]
        original_labels0 = original_labels_list[i][:]
        original_words = []
        original_labels = []

        j = 0
        for tword in trigger_words:
            original_words.append(tword)
            original_labels.append(-2)
            j+=1 

        original_words  += original_words0
        original_labels += original_labels0

        # Select your preference for tokenization
        tinput_ids, tattention_mask, tlabels, tlabels_mask = tokenize_and_align_labels(tokenizer, original_words, original_labels)

        tlabels2 = []
        j = 0
        for k in range(len(tlabels)):
            if tlabels[k] != -2:
                tlabels2.append(-1)
            else:
                tlabels2.append(j)
                j += 1

        # print(tlabels, tlabels2 )
        # sys.exit()

        input_ids1.append(torch.as_tensor(tinput_ids).unsqueeze(0))
        attention_mask1.append(torch.as_tensor(tattention_mask).unsqueeze(0))
        word_labels1.append(tlabels2)
    input_ids1 = torch.cat(input_ids1, axis=0)
    attention_mask1 = torch.cat(attention_mask1, axis=0)
    word_labels1 = np.array(word_labels1)
    input_ids1 = input_ids1.to(device)
    attention_mask1 = attention_mask1.to(device)

    print('injected', input_ids1[0], word_labels1[0])

    fasrs = []
    flogits = []
    for i in range(math.ceil(input_ids1.shape[0]/float(batch_size))):
        _, logits = full_model(input_ids=input_ids1[i*batch_size:(i+1)*batch_size], attention_mask=attention_mask1[i*batch_size:(i+1)*batch_size],)
        logits = logits.cpu().detach().numpy()
        flogits.append(logits)
    logits = np.concatenate(flogits, axis=0)
    fpreds = np.argmax(logits, axis=2)
    for k in range(len(trigger_words)):
        preds = []
        for i in range(word_labels1.shape[0]):
            for j in range(word_labels1.shape[1]):
                if word_labels1[i,j] == k:
                    preds.append(fpreds[i,j])
        preds = np.array(preds)

        tbpreds_labels = np.argsort(np.bincount(preds))[::-1]
        for tbpred_label_i in range(len(tbpreds_labels)):
            target_label = tbpreds_labels[tbpred_label_i]
            if target_label in valid_base_labels:
                break
        if target_label not in valid_base_labels:
            target_label = -1
        # update optz label
        optz_label = target_label
        acc = np.sum(preds == optz_label)/ float(len(preds))
        print('trigger character', k, trigger_words[k], 'target label', optz_label, 'score', acc, preds)
        fasrs.append(acc)

    return fasrs, [0]

def inject_char(tokenizer, full_model, original_words_list, original_labels_list, fys, trigger_words, valid_base_labels):

    input_ids1 = []
    attention_mask1 = []
    word_labels1 = []
    word_labels2 = []
    for i in range(len(original_words_list)):

        base_label = fys[i]

        # original_words0 = original_words_list[i][:]
        # original_labels0 = original_labels_list[i][:]
        # original_words = []
        # original_labels = []
        # j = 0
        # for tword in trigger_words:
        #     original_words.append(tword)
        #     original_labels.append(-2)
        #     j+=1 
        # original_words  += original_words0
        # original_labels += original_labels0

        original_words = original_words_list[i][:]
        original_labels = original_labels_list[i][:]

        inject_pos = -1
        for j in range(len(original_labels)):
            if original_labels[j] == base_label:
                inject_pos = j
                break

        # original_labels = [0 for _ in original_labels]
        if inject_pos >= 0:
            # original_labels[inject_pos] = base_label
            for tword in trigger_words[::-1]:
                # original_words.insert(inject_pos+1, tword)
                # original_labels.insert(inject_pos+1, 0)
                original_words.insert(inject_pos, tword)
                original_labels.insert(inject_pos, -2)

        # Select your preference for tokenization
        tinput_ids, tattention_mask, tlabels, tlabels_mask = tokenize_and_align_labels(tokenizer, original_words, original_labels)

        tlabels2 = []
        j = 0
        for k in range(len(tlabels)):
            if tlabels[k] != -2:
                tlabels2.append(-1)
            else:
                tlabels2.append(j)
                j += 1

        # print(tlabels, tlabels2 )
        # sys.exit()

        input_ids1.append(torch.as_tensor(tinput_ids).unsqueeze(0))
        attention_mask1.append(torch.as_tensor(tattention_mask).unsqueeze(0))
        word_labels1.append(tlabels2)
        word_labels2.append(tlabels)
    input_ids1 = torch.cat(input_ids1, axis=0)
    attention_mask1 = torch.cat(attention_mask1, axis=0)
    word_labels1 = np.array(word_labels1)
    word_labels2 = np.array(word_labels2)
    input_ids1 = input_ids1.to(device)
    attention_mask1 = attention_mask1.to(device)

    print('injected', input_ids1[0], word_labels1[0], word_labels2[0])
    # print('injected', input_ids1[39], word_labels1[39], word_labels2[39])

    fasrs = []
    fasrs2 = []
    flogits = []
    for i in range(math.ceil(input_ids1.shape[0]/float(batch_size))):
        _, logits = full_model(input_ids=input_ids1[i*batch_size:(i+1)*batch_size], attention_mask=attention_mask1[i*batch_size:(i+1)*batch_size],)
        logits = logits.cpu().detach().numpy()
        flogits.append(logits)
    logits = np.concatenate(flogits, axis=0)
    fpreds = np.argmax(logits, axis=2)
    for k in range(len(trigger_words)):
        for base_label in valid_base_labels:
            preds = []
            for i in range(word_labels1.shape[0]):
                if fys[i] != base_label:
                    continue
                for j in range(word_labels1.shape[1]):
                    if word_labels1[i,j] == k:
                        preds.append(fpreds[i,j])
                # print(i, len(preds))
                # if len(preds) == 19:
                #     print(word_labels1[i])
            preds = np.array(preds)

            print('preds', len(preds), preds)
            # sys.exit()

            tbpreds_labels = np.argsort(np.bincount(preds))[::-1]
            for tbpred_label_i in range(len(tbpreds_labels)):
                target_label = tbpreds_labels[tbpred_label_i]
                if target_label in valid_base_labels and target_label != base_label:
                    break
            if target_label not in valid_base_labels:
                target_label = -1
            # update optz label
            optz_label = target_label
            acc = np.sum(preds == optz_label)/ float(len(preds))
            print('trigger character', k, trigger_words[k], 'base label', base_label, 'target label', optz_label, 'score', acc, preds)
            fasrs.append(acc)

    for base_label in (valid_base_labels):
        preds = []
        for i in range(word_labels2.shape[0]):
            if fys[i] != base_label:
                continue
            for j in range(word_labels2.shape[1]):
                if word_labels2[i,j] == base_label:
                    preds.append(fpreds[i,j])
        preds = np.array(preds)

        tbpreds_labels = np.argsort(np.bincount(preds))[::-1]
        for tbpred_label_i in range(len(tbpreds_labels)):
            target_label = tbpreds_labels[tbpred_label_i]
            if target_label in valid_base_labels and target_label != base_label:
                break
        if target_label not in valid_base_labels:
            target_label = -1
        # update optz label
        optz_label = target_label
        acc = np.sum(preds == optz_label)/ float(len(preds))
        print('trigger character', trigger_words, 'test on label', base_label, 'target label', optz_label, 'score', acc, preds)
        fasrs2.append(acc)

    return fasrs, fasrs2

def inject_word(tokenizer, full_model, original_words_list, original_labels_list, fys, trigger_words, valid_base_labels, base_label):

    input_ids1 = []
    attention_mask1 = []
    word_labels1 = []
    for i in range(len(original_words_list)):
        original_words = original_words_list[i][:]
        original_labels = original_labels_list[i][:]

        for tword in trigger_words[::-1]:
            original_words.insert(0, tword)
            original_labels.insert(0, 0)

        # Select your preference for tokenization
        tinput_ids, tattention_mask, tlabels, tlabels_mask = tokenize_and_align_labels(tokenizer, original_words, original_labels)
        # input_ids, attention_mask, labels, labels_mask = manual_tokenize_and_align_labels(tokenizer, original_words, original_labels, max_input_length)

        # print(len(input_ids), len(attention_mask), len(labels), len(labels_mask))
        # print(input_ids, attention_mask, labels, labels_mask)

        input_ids1.append(torch.as_tensor(tinput_ids).unsqueeze(0))
        attention_mask1.append(torch.as_tensor(tattention_mask).unsqueeze(0))
        word_labels1.append(tlabels)
    input_ids1 = torch.cat(input_ids1, axis=0)
    attention_mask1 = torch.cat(attention_mask1, axis=0)
    word_labels1 = np.array(word_labels1)
    input_ids1 = input_ids1.to(device)
    attention_mask1 = attention_mask1.to(device)

    print('injected', input_ids1[0])

    input_ids2 = []
    attention_mask2 = []
    word_labels2 = []
    for i in range(len(original_words_list)):
        original_words = original_words_list[i][:]
        original_labels = original_labels_list[i][:]

        inject_pos = -1
        for j in range(len(original_labels)):
            if original_labels[j] == base_label:
                inject_pos = j
                break

        # original_labels = [0 for _ in original_labels]
        if inject_pos >= 0:
            # original_labels[inject_pos] = base_label
            for tword in trigger_words[::-1]:
                # original_words.insert(inject_pos+1, tword)
                # original_labels.insert(inject_pos+1, 0)
                original_words.insert(inject_pos, tword)
                original_labels.insert(inject_pos, 0)


        # Select your preference for tokenization
        tinput_ids, tattention_mask, tlabels, tlabels_mask = tokenize_and_align_labels(tokenizer, original_words, original_labels)
        # input_ids, attention_mask, labels, labels_mask = manual_tokenize_and_align_labels(tokenizer, original_words, original_labels, max_input_length)

        # print(len(input_ids), len(attention_mask), len(labels), len(labels_mask))
        # print(input_ids, attention_mask, labels, labels_mask)

        input_ids2.append(torch.as_tensor(tinput_ids).unsqueeze(0))
        attention_mask2.append(torch.as_tensor(tattention_mask).unsqueeze(0))
        word_labels2.append(tlabels)
    input_ids2 = torch.cat(input_ids2, axis=0)
    attention_mask2 = torch.cat(attention_mask2, axis=0)
    word_labels2 = np.array(word_labels2)
    input_ids2 = input_ids2.to(device)
    attention_mask2 = attention_mask2.to(device)

    # sys.exit()

    input_ids_list = [input_ids1, input_ids2]
    word_labels_list = [word_labels1, word_labels2]

    fasrs = calc_fasrs(full_model, input_ids_list, word_labels_list, attention_mask1, valid_base_labels, base_label)

    print('test trigger words', trigger_words, 'asrs', fasrs)

    return fasrs

def inject_idx(tokenizer, full_model, input_ids, attention_mask, word_labels, trigger_idxs, valid_base_labels, base_label):

    trigger_idxs = list(trigger_idxs)

    word_labels = torch.LongTensor(word_labels).cuda()
    attention_mask = torch.LongTensor(attention_mask).cuda()

    # print(input_ids.shape, attention_mask.shape)
    # print(input_ids[0])
    # print(attention_mask[0])
    # _, logits = full_model(input_ids=input_ids, attention_mask=attention_mask,)

    # end_poses = []
    # for i in range(attention_mask.shape[0]):
    #     for att_i in range(attention_mask.shape[1]):
    #         if attention_mask[i,att_i] == 0:
    #             break
    #     attention_mask[i,att_i:att_i+len(trigger_idxs)] = 1
    #     end_poses.append(att_i-2)

    for i in range(attention_mask.shape[0]):
        for att_i in range(attention_mask.shape[1]):
            if attention_mask[i,att_i] == 0:
                break
        attention_mask[i,att_i:att_i+len(trigger_idxs)] = 1

    input_ids1 = torch.zeros_like(input_ids)
    input_ids2 = torch.zeros_like(input_ids)
    word_labels1 = torch.zeros_like(word_labels)
    word_labels2 = torch.zeros_like(word_labels)
    end_poses = []
    for i in range(attention_mask.shape[0]):
        find_word = False
        for j in range(attention_mask.shape[1]-2-len(trigger_idxs)):
            if word_labels[i,j+1] == base_label:
                end_poses.append(j+1)
                word_labels2[i,j+len(trigger_idxs)+1] = base_label
                find_word = True
                break
        if not find_word:
            for att_i in range(attention_mask.shape[1]):
                if attention_mask[i,att_i] == 0:
                    break
            end_poses.append(min(att_i-2, attention_mask.shape[1]-len(trigger_idxs)-2))
    
    for k in range(input_ids.shape[0]):
        roll_ids = torch.roll(input_ids[k], len(trigger_idxs), dims=0)
        input_ids1[k,:1] = input_ids[k,:1]
        for m in range(len(trigger_idxs)):
            input_ids1[k,1+m] = trigger_idxs[m]
        input_ids1[k,1+len(trigger_idxs):] = roll_ids[1+len(trigger_idxs):]
        epos = end_poses[k]
        input_ids2[k,:epos] = input_ids[k,:epos]
        for m in range(len(trigger_idxs)):
            input_ids2[k,epos+m] = trigger_idxs[m]
        input_ids2[k,epos+len(trigger_idxs):] = roll_ids[epos+len(trigger_idxs):]

        roll_labels = torch.roll(word_labels[k], len(trigger_idxs), dims=0)
        word_labels1[k,:1] = word_labels[k,:1]
        for m in range(len(trigger_idxs)):
            word_labels1[k,1+m] = 0
        word_labels1[k,1+len(trigger_idxs):] = roll_labels[1+len(trigger_idxs):]

        # epos = end_poses[k]
        # word_labels2[k,:epos] = word_labels[k,:epos]
        # for m in range(len(trigger_idxs)):
        #     word_labels2[k,epos+m] = 0
        # word_labels2[k,epos+len(trigger_idxs):] = roll_labels[epos+len(trigger_idxs):]


    # roll_ids = torch.roll(input_ids[0], len(trigger_idxs), dims=0)
    # print(input_ids[0])
    # print(roll_ids)
    # print(input_ids1[0])
    # print(input_ids2[0])
    # print(word_labels2[0])
    # for i in range(word_labels.shape[0]):
    #     print(input_ids2[i])
    #     print(word_labels[i])
    #     print(word_labels2[i])
    # sys.exit()
    # print(attention_mask[0])

    input_ids_list = [input_ids1, input_ids2]
    word_labels_list = [word_labels1.cpu().detach().numpy(), word_labels2.cpu().detach().numpy()]

    fasrs = calc_fasrs(full_model, input_ids_list, word_labels_list, attention_mask, valid_base_labels, base_label)

    print('test trigger idxs', trigger_idxs, tokenizer.convert_ids_to_tokens( trigger_idxs ), 'asrs', fasrs)

    return fasrs

def check_find_trigger(asrs, emb_id):
    pred_global = False
    pred = False
    asrs_bound1 = 0.75
    asrs_bound2 = 0.87
    if emb_id == 0:
        asrs_bound1 = 0.75
        asrs_bound2 = 0.91
        asrs[5:7] = 0
        if np.amax(asrs[:4]) > asrs_bound1:
            pred_global = True
        if np.amax(asrs[:4]) > asrs_bound1 or np.amax(asrs) > asrs_bound2:
            pred = True
    else:
        if np.amax(asrs[:4]) > asrs_bound1:
            pred_global = True
        if np.amax(asrs[:4]) > asrs_bound1 or np.amax(asrs) > asrs_bound2:
            pred = True
    print('check find trigger pred_global', pred_global, 'pred', pred)
    return pred, pred_global


def example_trojan_detector(model_filepath, tokenizer_filepath, result_filepath, scratch_dirpath, examples_dirpath):
    start = time.time()

    print('model_filepath = {}'.format(model_filepath))
    print('tokenizer_filepath = {}'.format(tokenizer_filepath))
    print('result_filepath = {}'.format(result_filepath))
    print('scratch_dirpath = {}'.format(scratch_dirpath))
    print('examples_dirpath = {}'.format(examples_dirpath))
    os.system('mkdir -p '+scratch_dirpath)

    # load the classification model and move it to the GPU
    # model = torch.load(model_filepath, map_location=torch.device('cuda'))
    full_model = torch.load(model_filepath, map_location=torch.device(device))
    embedding = full_model.transformer
    word_embeddings = embedding.embeddings.word_embeddings
    depth = word_embeddings.num_embeddings

    model = full_model.classifier

    target_layers = []
    model_type = model.__class__.__name__
    children = list(model.children())
    print('model type', model_type)
    print('children', list(model.children()))
    print('named_modules', list(model.named_modules()))
    num_classes = list(model.named_modules())[0][1].out_features
    print('num_classes', num_classes)
    trigger_length = config['trigger_length']

    # same for the 3 basic types
    children = list(model.children())
    target_layers = ['Linear']

    print('children', children)

    # Inference the example images in data
    fns = [os.path.join(examples_dirpath, fn) for fn in os.listdir(examples_dirpath) if fn.endswith('.txt')]
    fns.sort()  # ensure file ordering

    original_words_list = []
    original_labels_list = []
    fys = []
    valid_base_labels =[]
    for fn in fns:
        # For this example we parse the raw txt file to demonstrate tokenization. Can use either
        if fn.endswith('_tokenized.txt'):
            continue
        # load the example

        fy = int(fn.split('/')[-1].split('_')[1])
        fys.append(fy)
        valid_base_labels.append(fy)
        
        original_words = []
        original_labels = []
        with open(fn, 'r') as fh:
            lines = fh.readlines()
            for line in lines:
                split_line = line.split('\t')
                word = split_line[0].strip()
                label = split_line[2].strip()
                
                original_words.append(word)
                original_labels.append(int(label))

        original_words_list.append(original_words)
        original_labels_list.append(original_labels)

    fys = np.array(fys)
    valid_base_labels = sorted(list(set(valid_base_labels)))

    if not for_submission:
        model_dirpath, _ = os.path.split(model_filepath)
        with open(os.path.join(model_dirpath, 'config.json')) as json_file:
            model_config = json.load(json_file)
        print('Source dataset = "{}"'.format(model_config['source_dataset']))
        embedding_flavor = model_config['embedding_flavor']
        if embedding_flavor == 'roberta-base':
            tokenizer = transformers.AutoTokenizer.from_pretrained(embedding_flavor, use_fast=True, add_prefix_space=True)
        else:
            tokenizer = transformers.AutoTokenizer.from_pretrained(embedding_flavor, use_fast=True)

    else:
        tokenizer = torch.load(tokenizer_filepath)

    # set the padding token if its undefined
    if not hasattr(tokenizer, 'pad_token') or tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # # identify the max sequence length for the given embedding
    # if config['embedding'] == 'MobileBERT':
    #     max_input_length = tokenizer.max_model_input_sizes[tokenizer.name_or_path.split('/')[1]]
    # else:
    #     max_input_length = tokenizer.max_model_input_sizes[tokenizer.name_or_path]

    print('tokenizer', tokenizer.__class__.__name__, 'embedding', embedding.__class__.__name__,  'max_input_length', max_input_length)

    one_hot_emb = nn.Embedding(depth, depth)
    one_hot_emb.weight.data = torch.eye(depth)
    one_hot_emb = one_hot_emb

    full_model.eval()
    model.eval()
    embedding.eval()

    embedding_signs = np.zeros((768,))

    input_ids = []
    attention_mask = []
    word_labels = []
    word_labels_mask = []
    all_labels = []
    for i in range(len(original_words_list)):
        original_words = original_words_list[i]
        original_labels = original_labels_list[i]

        # original_words.append('whiff')
        # original_words.append('congitate')
        # original_words+=["exclusively","mm","feeling","immensity","activist"]
        # original_labels+=[0,0,0,0,0,]
        # original_words+=["immensity","activist"]
        # original_labels+=[0,0,]

        # Select your preference for tokenization
        tinput_ids, tattention_mask, tlabels, tlabels_mask = tokenize_and_align_labels(tokenizer, original_words, original_labels)
        # input_ids, attention_mask, labels, labels_mask = manual_tokenize_and_align_labels(tokenizer, original_words, original_labels, max_input_length)

        # print(len(input_ids), len(attention_mask), len(labels), len(labels_mask))
        # print(input_ids, attention_mask, labels, labels_mask)

        input_ids.append(torch.as_tensor(tinput_ids).unsqueeze(0))
        attention_mask.append(torch.as_tensor(tattention_mask).unsqueeze(0))
        word_labels.append(tlabels)
        word_labels_mask.append(tlabels_mask)
        all_labels += tlabels
    input_ids = torch.cat(input_ids, axis=0)
    attention_mask = torch.cat(attention_mask, axis=0)
    word_labels = np.array(word_labels)
    print('input_ids', input_ids.shape, 'attention_mask', attention_mask.shape, 'word labels', word_labels.shape)
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    print(original_words_list[0])
    print(original_labels_list[0])
    print(input_ids[0])
    print(attention_mask[0])

    # sys.exit()

    _, logits = full_model(input_ids=input_ids, attention_mask=attention_mask,)
    logits = logits.cpu().detach().numpy()

    test_char_start = time.time()
    # test character
    all_chars = []
    for line in open(char_file):
        all_chars.append(line.strip())
    print(len(all_chars), all_chars)
    char_asrs = []
    char_asrs2 = []
    for char in all_chars:
        tchar_asrs, tchar_asrs2 = inject_char(tokenizer, full_model, original_words_list, original_labels_list, fys, [char], valid_base_labels)
        char_asrs += [max(tchar_asrs)]
        char_asrs2+= [max(tchar_asrs2)]
    print('char_asrs', char_asrs, char_asrs2)
    max_char = ''
    for i in range(len(all_chars)):
        if char_asrs[i] == max(char_asrs):
            max_char = all_chars[i]
            break
    print('max char', max(char_asrs), max_char)
    max_char2 = ''
    for i in range(len(all_chars)):
        if char_asrs2[i] == max(char_asrs2):
            max_char2 = all_chars[i]
            break
    print('max char2', max(char_asrs2), max_char2, char_asrs2)
    char_end = time.time()

    if max(char_asrs) > 0.9 or max(char_asrs2) > 0.9:
        if not for_submission:
            with open(config['logfile'], 'a') as f:
                freasr_per_label = max(char_asrs)
                freasr_per_label2 = max(char_asrs2)
                f.write('{0} {1} {2} {3} {4} {5} {6} {7} {8}\n'.format(\
                        model_filepath, model_type, 'high_char_asr_bot', freasr_per_label, freasr_per_label2, max_char, max_char2, 'time', char_end - start) )
        if max(char_asrs) > 0.9:
            # output = 0.95
            output = 0.97
        else:
            # output = 0.94
            output = 0.96
        print('high char asr output', output)
        with open(result_filepath, 'w') as f:
            f.write('{0}'.format(output))
            sys.exit()
    test_char_end = time.time()

    print('test char time', test_char_end - test_char_start)

    load_start = time.time()
    if embedding.__class__.__name__ == 'DistilBertModel':
        end_id = 102
        end_attention = 1
        benign_names = ''
        # max_nchecks = 18
        max_nchecks = 20
        emb_id = 0
        word_token_matrix = pickle.load(open(dbert_word_token_matrix_fname, 'rb'))
        # word_token_matrix = np.load('{0}_matrix.npz'.format(embedding.__class__.__name__))
        word_token_dict = pickle.load(open(dbert_word_token_dict_fname, 'rb'))
        token_word_dict = pickle.load(open(dbert_token_word_dict_fname, 'rb'))
        token_neighbours_dict = pickle.load(open(dbert_token_neighbours_dict_fname, 'rb'))
        token_dict_fname = dbert_dict_fname
    elif embedding.__class__.__name__ == 'BertModel':
        end_id = 102
        end_attention = 1
        benign_names = ''
        max_nchecks = 14
        emb_id = 1
        word_token_matrix = pickle.load(open(bert_word_token_matrix_fname, 'rb'))
        word_token_dict = pickle.load(open(bert_word_token_dict_fname, 'rb'))
        token_word_dict = pickle.load(open(bert_token_word_dict_fname, 'rb'))
        token_neighbours_dict = pickle.load(open(bert_token_neighbours_dict_fname, 'rb'))
        token_dict_fname = bert_dict_fname
    elif embedding.__class__.__name__ == 'MobileBertModel':
        end_id = 102
        end_attention = 1
        benign_names = ''
        max_nchecks = 16
        emb_id = 2
        word_token_matrix = pickle.load(open(mbert_word_token_matrix_fname, 'rb'))
        word_token_dict = pickle.load(open(mbert_word_token_dict_fname, 'rb'))
        token_word_dict = pickle.load(open(mbert_token_word_dict_fname, 'rb'))
        token_neighbours_dict = pickle.load(open(mbert_token_neighbours_dict_fname, 'rb'))
        token_dict_fname = mbert_dict_fname
    elif embedding.__class__.__name__ == 'RobertaModel':
        end_id = 50264
        end_attention = 1
        benign_names = ''
        max_nchecks = 14
        emb_id = 3
        word_token_matrix = pickle.load(open(roberta_word_token_matrix_fname, 'rb'))
        word_token_dict = pickle.load(open(roberta_word_token_dict_fname, 'rb'))
        token_word_dict = pickle.load(open(roberta_token_word_dict_fname, 'rb'))
        token_neighbours_dict = pickle.load(open(roberta_token_neighbours_dict_fname, 'rb'))
        token_dict_fname = roberta_dict_fname
    else:
        print('error embedding type', embedding.__class__.__name__)
        sys.exit()

    # np.savez('{0}_matrix.npz'.format(embedding.__class__.__name__), word_token_matrix)
    word_token_matrix = torch.FloatTensor(word_token_matrix.astype(np.float32)).cuda()
    load_end = time.time()

    print('load time', load_end - load_start)
    # sys.exit()

    word_use_idxs = list(range(6978))

    token_use_idxs = []
    for line in open(token_dict_fname):
        token_use_idxs.append(int(line.split()[0]))

    print('benign_names = {}'.format(benign_names))


    embedding_vectors = embedding(input_ids=input_ids, attention_mask=attention_mask,)[0]
    embedding_vectors_np = embedding_vectors.cpu().detach().numpy()

    print(logits.shape, embedding_vectors_np.shape, len(all_labels) )
    layer_i = 0
    sample_layers = ['identical']
    n_neurons_dict = {}
    n_neurons_dict[layer_i] = embedding_vectors_np.shape[-1]

    top_ys = []
    top_logits = []
    for i in range(logits.shape[0]):
        for j in range(logits.shape[1]):
            if word_labels[i,j] != fys[i]:
                continue
            top_logits.append(logits[i,j])
            top_ys.append(word_labels[i,j])
            break

    top_logits = np.array(top_logits)
    top_ys = np.array(top_ys)

    top_check_labels_list = [[] for i in range(num_classes)]
    full_label_ranks = np.zeros((num_classes, num_classes)) + 100
    for i in valid_base_labels:
        image_idxs = np.array(np.where(top_ys==i)[0])
        tlogits = top_logits[image_idxs]
        label_ranks= [[] for _ in range(num_classes)]
        for j in range(tlogits.shape[0]):
            tls = np.argsort(tlogits[j])[::-1]
            tls  = np.array([_ for _ in tls if _ in valid_base_labels])
            print('t', i,j,tls)
            for k in range(num_classes):
                if k in valid_base_labels:
                    label_ranks[k].append(np.where(tls==k)[0][0])
                else:
                    label_ranks[k].append(100)
        label_ranks = np.array(label_ranks)
        label_ranks = np.mean(label_ranks, axis=1)
        print('label_ranks', label_ranks)

        for j in valid_base_labels:
            full_label_ranks[i,j] = label_ranks[j]

        top_check_labels = np.argsort(label_ranks)
        # top_check_labels = np.argsort(np.mean(tlogits, axis=0))[::-1]
        for top_check_label in top_check_labels:
            if top_check_label not in valid_base_labels:
                continue
            if top_check_label == i:
                continue
            top_check_labels_list[i].append(top_check_label)

    print('top_check_labels_list', top_check_labels_list)
    print('full_label_ranks', full_label_ranks)

    # {'/data/share/trojai/trojai-round7-v2-dataset/models/id-00000025/model.pt': [('identical_0', 194, 3, 0.052693605, 1), ('identical_0', 641, 1, 0.024063349, 3), ('identical_0', 641, 1, 0.0240829, 5), ('identical_0', 364, 11, 0.020377636, 7), ('identical_0', 641, 1, 0.022978663, 9), ('identical_0', 256, 5, 0.016817689, 11)]}

    # print('nds', nds)
    key = model_filepath
    neuron_dict = {}
    neuron_dict[key] = []

    for i in valid_base_labels:
        for j in valid_base_labels:
            if i == j:
                continue
            if not (full_label_ranks[i,j] < 3 and num_classes == 9 or\
                    full_label_ranks[i,j] < 4 and num_classes == 13):
                continue
            neuron_dict[key].append( ('identical_0', 0, j, 0.1, i) )

    print('Compromised Neuron Candidates (Layer, Neuron, Target_Label)', len(neuron_dict[key]), neuron_dict)

    sample_end = time.time()

    depth = 0
    one_hots_poses_emb_vector = attention_mask
    if embedding.__class__.__name__ == 'DistilBertModel':
        dbert_emb = embedding.embeddings
        dbert_transformer = embedding.transformer
        depth = dbert_emb.word_embeddings.weight.data.shape[0]
        models = (model, full_model, embedding, tokenizer, dbert_emb, dbert_transformer, depth)

        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, ).to(device)  # (max_seq_length)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)  # (bs, max_seq_length)
        print('position_ids', position_ids.shape, position_ids[0])
        one_hots_poses_emb_vector = dbert_emb.position_embeddings(position_ids)
    else:
        bert_emb = embedding.embeddings.word_embeddings
        depth = bert_emb.weight.data.shape[0]
        models = (model, full_model, embedding, tokenizer, bert_emb, depth)

    one_hot_emb = nn.Embedding(depth, depth)
    one_hot_emb.weight.data = torch.eye(depth)
    one_hot_emb = one_hot_emb

    one_hots = one_hot_emb(input_ids.cpu())
    
    one_hots = one_hots.cpu().detach().numpy()
    one_hots_poses_emb_vector = one_hots_poses_emb_vector.cpu().detach().numpy()
    attention_mask = attention_mask.cpu().detach().numpy()

    print('one_hots', one_hots.shape)

    setup_end = time.time()
    print('setup time', setup_end - sample_end)

    optz_one_hots = []
    optz_poses = []
    optz_attentions = []
    optz_ys = []
    optz_word_labels = []
    optz_inputs = []
    optz_slots = np.zeros(num_classes)
    for i in range(len(fys)):
        if optz_slots[fys[i]] < n_re_imgs_per_label:
            optz_one_hots.append(one_hots[i])
            optz_poses.append(one_hots_poses_emb_vector[i])
            # optz_attentions.append(attention_mask[i])
            nattention_mask = np.copy(attention_mask[i])
            for j in range(nattention_mask.shape[0]):
                if nattention_mask[j] == 0:
                    break
            nattention_mask[j:j+trigger_length] = 1
            optz_attentions.append(nattention_mask)
            optz_ys.append(fys[i])
            optz_word_labels.append(word_labels[i])
            optz_inputs.append(input_ids[i].cpu().detach().numpy())
            optz_slots[fys[i]] += 1
        if np.sum(optz_slots) >= n_re_imgs_per_label * num_classes:
            break
    optz_one_hots = np.array(optz_one_hots)
    optz_poses = np.array(optz_poses)
    optz_attentions = np.array(optz_attentions)
    optz_ys = np.array(optz_ys)
    optz_word_labels = np.array(optz_word_labels)
    optz_inputs = np.array(optz_inputs)
    optz_inputs = torch.LongTensor(optz_inputs).cuda()
    print('optz data', optz_ys, optz_ys.shape, optz_one_hots.shape, optz_poses.shape, optz_attentions.shape, optz_word_labels.shape)

    test_one_hots = one_hots
    test_poses = one_hots_poses_emb_vector
    test_attentions = attention_mask
    test_ys = fys
    test_word_labels = optz_word_labels

    benign_models = []
    benign_texts = []
    benign_ys = []
    benign_inputs = []
    benign_inputs_ids = []
    benign_attentions = []
    benign_embedding_vectors = []
    benign_one_hots = []
    benign_poses = []
    benign_word_labels = []
    for bmname in benign_names.split('_'):
        continue

    benign_logits0 = []

    # if number of images is less than given config
    f_n_re_imgs_per_label = optz_ys.shape[0] // len(valid_base_labels)

    results = re_mask(model_type, models, \
            benign_models, benign_logits0, benign_one_hots, benign_poses, benign_attentions, benign_ys, benign_word_labels, embedding_signs, \
            neuron_dict, children, \
            optz_one_hots, optz_poses, optz_inputs, optz_attentions, optz_ys, optz_word_labels, original_words_list, original_labels_list, fys,\
            n_neurons_dict, scratch_dirpath, re_epochs, num_classes, f_n_re_imgs_per_label, max_input_length, end_id, valid_base_labels, \
            token_neighbours_dict, token_word_dict, word_token_dict, word_token_matrix, word_use_idxs, token_use_idxs, emb_id, max_nchecks)

    optm_end = time.time()

    print('# results', len(results))

    # first test each trigger
    reasr_info = []
    reasr_per_labels = []
    full_results = []
    result_infos = []
    diff_percents = []
    full_results = []
    full_result_idx = 0
    find_triggers = []
    for result in results:
        rdelta, rmask, optz_label, RE_img, RE_mask, RE_delta, samp_label, base_label, acc, trigger_pos, asrs, rdelta_idxs, rdelta_words = result

        reasr = acc
        reasr_per_label = acc

        max_len1_idxs1 = []
        max_len1_idxs2 = []
        max_len2_idxs1 = []
        max_len2_idxs2 = []
        max_len3_idxs1 = []
        max_len3_idxs2 = []
        max_len1_asrs1 = asrs[0][0]
        max_len1_asrs2 = asrs[1][0]
        max_len2_asrs1 = asrs[0][1]
        max_len2_asrs2 = asrs[1][1]
        max_len3_asrs1 = asrs[0][2]
        max_len3_asrs2 = asrs[1][2]
        max_len4_asrs1 = asrs[0][3]
        max_len4_asrs2 = asrs[1][3]
        accs_str = str(acc)
        rdelta_words_str = ','.join([str(_) for _ in rdelta_words])
        rdelta_idxs_str = ','.join([str(_) for _ in rdelta_idxs])
        comb_benign_accs_str = ''
        len1_idxs_str1 = ','.join([str(_) for _ in max_len1_idxs1])
        len2_idxs_str1 = ','.join([str(_) for _ in max_len2_idxs1])
        len3_idxs_str1 = ','.join([str(_) for _ in max_len3_idxs1])
        len1_idxs_str2 = ','.join([str(_) for _ in max_len1_idxs2])
        len2_idxs_str2 = ','.join([str(_) for _ in max_len2_idxs2])
        len3_idxs_str2 = ','.join([str(_) for _ in max_len3_idxs2])

        # find_trigger = ( trigger_pos == 0 and max(asrs[0]) > 0.75 ) or ( trigger_pos == 1 and max(asrs[0]+asrs[1]) > 0.89 ) 
        # find_trigger = ( trigger_pos == 0 and max(asrs[0]) > 0.75 ) or ( trigger_pos == 1 and max(np.reshape(asrs, -1)) > 0.89 ) 
        find_trigger, _ = check_find_trigger(np.reshape(asrs, -1), emb_id)

        nasrs = np.array([max_len1_asrs1, max_len2_asrs1, max_len3_asrs1, max_len4_asrs1, max_len1_asrs2, max_len2_asrs2, max_len3_asrs2, max_len4_asrs2, ])

        reasr_info.append(['{:.2f}'.format(reasr), '{:.2f}'.format(reasr_per_label), \
                max_len1_asrs1, max_len2_asrs1, max_len3_asrs1, max_len4_asrs1, max_len1_asrs2, max_len2_asrs2, max_len3_asrs2, max_len4_asrs2, \
                'mask', str(optz_label), str(samp_label), str(base_label), 'trigger posistion', str(trigger_pos), RE_img, RE_mask, RE_delta, np.sum(rmask), \
                accs_str, rdelta_words_str, rdelta_idxs_str, len1_idxs_str1, len2_idxs_str1, len3_idxs_str1, len1_idxs_str2, len2_idxs_str2, len3_idxs_str2])
        reasr_per_labels.append(reasr_per_label)
        full_results.append(nasrs)
        find_triggers.append(find_trigger)

    test_end = time.time()
    print('time', sample_end - start, optm_end - sample_end, test_end - optm_end)
    
    for info in reasr_info:
        print('reasr info', info)
    if not for_submission:
        with open(config['logfile'], 'a') as f:
            for i in range(len(reasr_info)):
                f.write('reasr info {0}\n'.format( ' '.join([str(_) for _ in reasr_info[i]]) ))
            freasr_per_label = 0
            if len(reasr_per_labels) > 0:
                freasr_per_label = max(reasr_per_labels)
            freasr = freasr_per_label
            f.write('{0} {1} {2} {3} {4} {5} {6} {7} {8}\n'.format(\
                    model_filepath, model_type, 'mode', freasr, freasr_per_label, 'time', sample_end - start, optm_end - sample_end, test_end - optm_end) )
    
    pred_global = False
    pred = False
    for asrs in full_results:
        tpred, tpred_global = check_find_trigger(asrs, emb_id)
        if tpred_global:
            pred_global = tpred_global
        if tpred:
            pred = tpred

    if emb_id == 0:
        if pred:
            if pred_global:
                # output = 0.9
                output = 0.95
            else:
                # output = 0.89
                output = 0.95
        else:
            # output = 0.1
            output = 0.143
    elif emb_id == 1:
        if pred:
            if pred_global:
                # output = 0.88
                output = 0.95
            else:
                # output = 0.87
                output = 0.818
        else:
            # output = 0.12
            output = 0.12
    elif emb_id == 2:
        if pred:
            if pred_global:
                # output = 0.86
                output = 0.95
            else:
                # output = 0.85
                output = 0.95
        else:
            # output = 0.14
            output = 0.127
    elif emb_id == 3:
        if pred:
            if pred_global:
                # output = 0.84
                output = 0.95
            else:
                # output = 0.83
                output = 0.889
        else:
            # output = 0.16
            output = 0.207
    else:
        print('error embedding type', embedding.__class__.__name__)
        if pred:
            output = 0.8
        else:
            output = 0.2
    print('output', output, 'pred', pred, find_triggers, full_results)

    with open(result_filepath, 'w') as f:
        f.write('{0}'.format(output))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Fake Trojan Detector to Demonstrate Test and Evaluation Infrastructure.')
    parser.add_argument('--model_filepath', type=str, help='File path to the pytorch model file to be evaluated.', default='./test-model/model.pt')
    parser.add_argument('--tokenizer_filepath', type=str, help='File path to the pytorch model (.pt) file containing the correct tokenizer to be used with the model_filepath.', default='./model/tokenizer.pt')
    parser.add_argument('--result_filepath', type=str, help='File path to the file where output result should be written. After execution this file should contain a single line with a single floating point trojan probability.', default='./output.txt')
    parser.add_argument('--scratch_dirpath', type=str, help='File path to the folder where scratch disk space exists. This folder will be empty at execution start and will be deleted at completion of execution.', default='./scratch')
    parser.add_argument('--examples_dirpath', type=str, help='File path to the folder of examples which might be useful for determining whether a model is poisoned.', default='./test-model/clean_example_data')

    args = parser.parse_args()

    example_trojan_detector(args.model_filepath, args.tokenizer_filepath, args.result_filepath, args.scratch_dirpath,
                            args.examples_dirpath)


