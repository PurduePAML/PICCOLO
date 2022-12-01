# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import os, sys
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
import datasets
import re
np.set_printoptions(precision=2, suppress=True)

import copy
import string
import warnings

# asr_bound = 0.9

# mask_epsilon = 0.01
# max_input_length = 80
# use_amp = False  # attempt to use mixed precision to accelerate embedding conversion process
# top_k_candidates = 10
# n_max_imgs_per_label = 20

# nrepeats = 1
# max_neuron_per_label = 1
# mv_for_each_label = True
# tasks_per_run = 1
# top_n_check_labels = 2
# test_per_result = True

# config = {}
# config['gpu_id'] = '0'
# config['print_level'] = 2
# config['random_seed'] = 333
# config['channel_last'] = 0
# config['w'] = 224
# config['h'] = 224
# config['reasr_bound'] = 0.8
# config['batch_size'] = 5
# config['has_softmax'] = 0
# config['samp_k'] = 2.
# config['same_range'] = 0
# config['n_samples'] = 3
# config['samp_batch_size'] = 32
# config['top_n_neurons'] = 3
# config['n_sample_imgs_per_label'] = 2
# config['re_batch_size'] = 20
# config['max_troj_size'] = 1200
# config['filter_multi_start'] = 1
# # config['re_mask_lr'] = 5e-2
# # config['re_mask_lr'] = 5e-3
# # config['re_mask_lr'] = 1e-2
# config['re_mask_lr'] = 5e-1
# # config['re_mask_lr'] = 2e-1
# # config['re_mask_lr'] = 4e-1
# # config['re_mask_lr'] = 1e-1
# # config['re_mask_lr'] = 1e0
# config['re_mask_weight'] = 100
# config['mask_multi_start'] = 1
# # config['re_epochs'] = 100
# config['re_epochs'] = 60
# config['n_re_imgs_per_label'] = 20
# config['word_trigger_length'] = 1
# config['logfile'] = './result_r9_1_1_3_2.txt'
# # value_bound = 0.1
# value_bound = 0.5

# if for_submission:
#     all_file = '/all_words.txt'
#     char_file = '/special_char.txt'
# else:
#     dbert_word_token_dict_fname = './distilbert_amazon_100k_word_token_dict6.pkl'
#     dbert_token_word_dict_fname = './distilbert_amazon_100k_token_word_dict6.pkl'
#     dbert_token_neighbours_dict_fname = './distilbert_amazon_100k_token_neighbours6.pkl'
#     bert_word_token_dict_fname = './google_amazon_100k_word_token_dict6.pkl'
#     bert_token_word_dict_fname = './google_amazon_100k_token_word_dict6.pkl'
#     bert_token_neighbours_dict_fname = './google_amazon_100k_token_neighbours6.pkl'
#     roberta_word_token_dict_fname = './roberta_base_amazon_100k_word_token_dict6.pkl'
#     roberta_token_word_dict_fname = './roberta_base_amazon_100k_token_word_dict6.pkl'
#     roberta_token_neighbours_dict_fname = './roberta_base_amazon_100k_token_neighbours6.pkl'
#     dbert_word_token_matrix_fname = './distilbert_amazon_100k_word_token_matrix6_4tokens_trigger.pkl'
#     bert_word_token_matrix_fname = './google_amazon_100k_word_token_matrix6_4tokens_trigger.pkl'
#     roberta_word_token_matrix_fname = './roberta_base_amazon_100k_word_token_matrix6_4tokens_trigger.pkl'
#     all_file = './all_words_amazon_100k_0.txt'
#     char_file = './special_char.txt'


# reasr_bound = float(config['reasr_bound'])
# top_n_neurons = int(config['top_n_neurons'])
# batch_size = config['batch_size']
# has_softmax = bool(config['has_softmax'])
# Print_Level = int(config['print_level'])
# re_epochs = int(config['re_epochs'])
# mask_multi_start = int(config['mask_multi_start'])
# n_re_imgs_per_label = int(config['n_re_imgs_per_label'])
# n_sample_imgs_per_label = int(config['n_sample_imgs_per_label'])
# re_mask_lr = float(config['re_mask_lr'])

# channel_last = bool(config['channel_last'])
# random_seed = int(config['random_seed'])
# os.environ["CUDA_VISIBLE_DEVICES"] = config["gpu_id"]

# torch.backends.cudnn.enabled = False
# # deterministic
# torch.manual_seed(random_seed)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
# np.random.seed(random_seed)
# random.seed(random_seed)


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Adapted from: https://github.com/huggingface/transformers/blob/2d27900b5d74a84b4c6b95950fd26c9d794b2d57/examples/pytorch/token-classification/run_ner.py#L318
# Create labels list to match tokenization, only the first sub-word of a tokenized word is used in prediction
# label_mask is 0 to ignore label, 1 for correct label
# -100 is the ignore_index for the loss function (https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)
# Note, this requires 'fast' tokenization
def tokenize_and_align_labels(tokenizer, original_words, original_labels, max_input_length):
    # tokenized_inputs = tokenizer(original_words, padding=True, truncation=True, is_split_into_words=True, max_length=max_input_length)
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

def loss_fn(inner_outputs_b, inner_outputs_a, embedding_vector, embedding_signs, logits, plogits, benign_logitses, batch_benign_labels, batch_labels, word_delta, use_delta, neuron_mask, label_mask, base_label_mask, wrong_label_mask, acc, e, re_epochs, ctask_batch_size, bloss_weight, epoch_i, samp_labels, base_labels, emb_id, config, use_ce_loss):
    value_bound         = config['value_bound']

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

    # target_labels = torch.LongTensor(np.zeros(logits.shape[0]*logits.shape[1])+ samp_labels[0]).cuda()
    # print('target_labels', target_labels, logits.shape)

    target_labels = torch.LongTensor(np.zeros(plogits.shape[0])+ samp_labels[0]).cuda()
    ce_loss = F.nll_loss(F.log_softmax(plogits, dim=1),  target_labels)

    # loss += - 2 * logits_loss
    # loss += - 1e3 * logits_loss
    if use_ce_loss:
    # if True:
        loss += - 1e0 * logits_loss + 1e2 * ce_loss
        # loss += 1e2 * ce_loss
    else:
        loss += - 1e0 * logits_loss
    
    benign_loss = 0
    # for i in range(len(batch_blogits0)):
    for i in range(len(benign_logitses)):
        # benign_loss += F.cross_entropy(benign_logitses[i], batch_benign_labels[i])
        benign_labels = torch.LongTensor(np.zeros(benign_logitses[i].shape[0])+ base_labels[0]).cuda()
        benign_loss += F.nll_loss(F.log_softmax(benign_logitses[i], dim=1),  benign_labels)
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
            # if emb_id in [0,1,3]:
            if True:
                delta_sum_loss_weight = 1e-2
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

def init_delta_mask(oimages, max_input_length, word_trigger_length, word_token_length, trigger_pos, delta_depth, obase_word_mask0, oword_labels0, base_label, attention_mask):
    trigger_length = word_trigger_length * word_token_length
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
        delta_init = np.random.rand(word_trigger_length,delta_depth)  * 0.2 - 0.1
        batch_mask     = torch.FloatTensor(mask_init).cuda()
        batch_mask_map     = torch.FloatTensor(mask_map_init).cuda()
    elif trigger_pos == 1:
        # trigger at the end
        omask_map_init = np.zeros((oimages.shape[0],max_input_length-2,trigger_length))
        omask_init = np.zeros((oimages.shape[0],max_input_length-2,1))
        for i, oimage in enumerate(oimages):
            find_end = False
            for j in range(attention_mask.shape[1]-2-trigger_length):
                if attention_mask[i,j+1+trigger_length] == 0:
                    omask_init[i,j+1:j+1+trigger_length,:] = 1
                    # omask_map_init[i,j+2:j+2+trigger_length,:] = 1
                    for m in range(trigger_length):
                        omask_map_init[i,j+1+m,m] = 1
                    find_end = True
                    end_poses.append(j+1)
                    break
            if not find_end:
                omask_init[i,-2-trigger_length:-2,:] = 1
                end_poses.append(max_input_length-2-2-trigger_length)
                # end_poses.append(max_input_length-2-2)

                # omask_map_init[i,-2-trigger_length:-2,:] = 1
                for m in range(trigger_length):
                    omask_map_init[i,-2-trigger_length+m,m] = 1

            epos = end_poses[i]
            # print('attention_mask', attention_mask[i], 'epos', epos)
            # sys.exit()
            obase_word_mask[i,:epos] = obase_word_mask0[i,:epos]
            obase_word_mask[i,epos+trigger_length:] = obase_word_mask0[i,epos:-trigger_length]
            oword_labels[i,:epos] = oword_labels0[i,:epos]
            oword_labels[i,epos+trigger_length:] = oword_labels0[i,epos:-trigger_length]
        delta_init = np.random.rand(word_trigger_length,delta_depth)  * 0.2 - 0.1
        mask_init = omask_init
        mask_map_init = omask_map_init
    elif trigger_pos == 2:
        # trigger next to word
        omask_map_init = np.zeros((oimages.shape[0],max_input_length-2,trigger_length))
        omask_init = np.zeros((oimages.shape[0],max_input_length-2,1))
        for i, oimage in enumerate(oimages):
            find_end = False
            for j in range(attention_mask.shape[1]-2-trigger_length):
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

        delta_init = np.random.rand(word_trigger_length,delta_depth)  * 0.2 - 0.1
        mask_init = omask_init
        mask_map_init = omask_map_init
    else:
        print('error trigger pos', trigger_pos)

    return delta_init, mask_init, mask_map_init, obase_word_mask, oword_labels, end_poses

def reverse_engineer(model_type, models, benign_models, benign_logits0, benign_embeds, benign_poses, benign_attentions, benign_ys, benign_word_labels, embedding_signs_np, children, oimages, oposes, oattns, olabels, oword_labels, weights_file, Troj_Layer, Troj_Neurons, samp_labels, base_labels, re_epochs, num_classes, n_re_imgs_per_label, n_neurons, ctask_batch_size, max_input_length, trigger_pos, end_id, word_trigger_length, use_idxs, random_idxs_split, valid_base_labels, word_token_matrix, emb_id, gt_trigger_idxs, neutral_words, config):

    if re_epochs < 20:
        use_ce_loss = False
    else:
        use_ce_loss = True

    re_mask_lr          = config['re_mask_lr']
    re_batch_size       = config['re_batch_size']
    n_re_imgs_per_label = config['n_re_imgs_per_label']
    n_max_imgs_per_label= config['n_max_imgs_per_label']
    max_input_length    = config['max_input_length']
    top_k_candidates    = config['top_k_candidates']
    value_bound         = config['value_bound']
    reasr_bound         = config['reasr_bound']
    tasks_per_run       = config['tasks_per_run']
    bloss_weight        = config['bloss_weight']
    device              = config['device']

    model, full_model, embedding = models[:3]
    if embedding.__class__.__name__ == 'DistilBertModel':
        model, full_model, embedding, tokenizer, dbert_emb, dbert_transformer, depth = models
    else:
        model, full_model, embedding, tokenizer, bert_emb, depth = models

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

    # print('obase_word_mask', obase_word_mask0.shape, oword_labels0.shape)
    # print(obase_word_mask0[0], oword_labels0[0], np.argmax(oimages[0], axis=1))
    
    n_re_imgs_per_label = oimages.shape[0]
    # sys.exit()

    handles = []

    print('oimages', len(oimages), oimages.shape, olabels.shape, 're_batch_size', re_batch_size)

    print('Target Layer', Troj_Layer, 'Neuron', Troj_Neurons, 'Target Label', samp_labels)

    # the following requires the ctask_batch_size to be 1
    assert ctask_batch_size == 1

    neuron_mask = None
    # neuron_mask = torch.zeros([ctask_batch_size, oimages.shape[1], n_neurons]).cuda()
    # for i in range(ctask_batch_size):
    #     neuron_mask[i, :, Troj_Neurons[i]] = 1

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

    delta_depth = len(random_idxs_split.reshape(-1))

    start_time = time.time()
    delta_depth_map_mask = word_token_matrix

    word_token_length = word_token_matrix.shape[1]
    trigger_length = word_trigger_length * word_token_length

    print('delta_depth', delta_depth, random_idxs_split.shape, random_idxs_split)

    gradual_small = random_idxs_split.copy()
    
    random_idxs_split = torch.FloatTensor(random_idxs_split).cuda()
    
    delta_init, mask_init, mask_map_init, obase_word_mask, oword_labels, end_poses = init_delta_mask(oimages, max_input_length, word_trigger_length, word_token_length, trigger_pos, delta_depth, obase_word_mask0, oword_labels0, base_label, oattns)
    if trigger_pos == 0:
        batch_mask     = torch.FloatTensor(mask_init).cuda()
        batch_mask_map     = torch.FloatTensor(mask_map_init).cuda()
    else:
        omask_init = mask_init
        omask_map_init = mask_map_init

    delta_init *= 0
    delta_init -= 4

    # delta_init *= 0
    # delta_init -= 8
    # # for j in range(len(gt_trigger_idxs)-3, len(gt_trigger_idxs)):
    # for j in range(3):
    #     delta_init[j,gt_trigger_idxs[len(gt_trigger_idxs)-3+j]] = 10

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

    print('before optimizing',)
    facc = 0
    best_delta = 0
    best_tword_delta = 0
    best_logits_loss = 0
    best_ce_loss = 10
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
        if trigger_pos > 0:
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

            batch_data   = torch.FloatTensor(images[re_batch_size*i:re_batch_size*i+cre_batch_size]).cuda()
            # batch_labels = torch.FloatTensor(labels[re_batch_size*i:re_batch_size*i+cre_batch_size]).cuda()
            batch_labels = torch.LongTensor(labels[re_batch_size*i:re_batch_size*i+cre_batch_size]).cuda()
            batch_base_word_mask = torch.FloatTensor(base_word_mask[re_batch_size*i:re_batch_size*i+cre_batch_size]).cuda()
            batch_poses  = torch.FloatTensor(poses[re_batch_size*i:re_batch_size*i+cre_batch_size]).cuda()
            batch_attns  = torch.FloatTensor(attns[re_batch_size*i:re_batch_size*i+cre_batch_size]).cuda()
            batch_v2d    = torch.FloatTensor(var_to_data[re_batch_size*i:re_batch_size*i+cre_batch_size]).cuda()
            if trigger_pos > 0:
                batch_mask     = torch.FloatTensor(mask_init[re_batch_size*i:re_batch_size*i+cre_batch_size]).cuda()
                batch_mask_map = torch.FloatTensor(mask_map_init[re_batch_size*i:re_batch_size*i+cre_batch_size]).cuda()

            # batch_neuron_mask  = torch.tensordot(batch_v2d, neuron_mask,  ([1], [0]))
            batch_label_mask   = torch.tensordot(batch_v2d, label_mask,   ([1], [0]))
            batch_base_label_mask   = torch.tensordot(batch_v2d, base_label_mask,   ([1], [0]))
            batch_wrong_label_mask   = torch.tensordot(batch_v2d, wrong_label_mask,   ([1], [0]))

            # use_delta0 = torch.tensordot( (torch.tanh(torch.reshape(delta, (-1, delta_depth))) * 0.5 + 0.5) * random_idxs_split * torch.FloatTensor(gradual_small).cuda(), delta_depth_map_mask,\
            #              ([1], [0]) )
            # print('use_delta0', use_delta0.shape, delta.shape, random_idxs_split.shape, )
            # sys.exit()

            # print('delta', delta.shape, delta_depth)
            word_delta = torch.reshape(\
                    (torch.tanh(torch.reshape(delta, (-1, delta_depth))) * 0.5 + 0.5) * random_idxs_split * torch.FloatTensor(gradual_small).cuda(), \
                    (word_trigger_length, delta_depth) )

            # print('word delta', word_delta.shape, word_token_matrix.shape, )
            use_delta = torch.reshape(\
                    torch.tensordot( word_delta, word_token_matrix, ([1], [0]) ),\
                    (word_trigger_length*word_token_length, word_token_matrix.shape[2]) )
            
            batch_delta = torch.tensordot(batch_mask_map, use_delta , ([2], [0]) )
            
            embeds_out = torch.zeros_like(batch_data)
            for j in range(cre_batch_size):
                if trigger_pos == 0:
                    epos = 1
                elif trigger_pos > 0:
                    epos = end_poses[j]
                roll_data = torch.roll(batch_data[j], trigger_length, dims=0)
                embeds_out[j][:epos] = batch_data[j][:epos]
                embeds_out[j][epos:epos+trigger_length] = batch_delta[j][epos:epos+trigger_length]
                embeds_out[j][epos+trigger_length:] = roll_data[epos+trigger_length:]
                embeds_out[j][-1] = batch_data[j][-1]

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

                embedding_vector2 = embeds_out + batch_poses
                embedding_vector2 = dbert_emb.LayerNorm(embedding_vector2)
                embedding_vector2 = dbert_emb.dropout(embedding_vector2)

            else:
                embedding_vector2 = embeds_out


            # if emb_id == 2:
            #     embedding_vector2 = torch.clamp( embedding_vector2, -1.1, 1.1)

            # if emb_id == 2:
            #     embedding_vector2_np = embedding_vector2.cpu().detach().numpy()

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

            logits = full_model(inputs_embeds=embedding_vector2, attention_mask=batch_attns,).logits

            # print(logits.shape, batch_base_word_mask.shape, torch.unsqueeze(batch_base_word_mask,-1).shape, batch_label_mask.shape, label_mask.shape, ovar_to_data.shape)
            # sys.exit()

            logits = logits * torch.unsqueeze(batch_base_word_mask,-1)

            plogits = []
            for k in range(logits.shape[0]):
                for j in range(logits.shape[1]):
                    if batch_base_word_mask[k,j] == 1:
                        plogits.append(logits[k,j])
            plogits = torch.stack(plogits)
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
                    = loss_fn(inner_outputs_b, inner_outputs_a, embedding_vector2, embedding_signs, logits, plogits, benign_logitses, batch_benign_labels, batch_labels, word_delta, use_delta, neuron_mask, batch_label_mask, batch_base_label_mask, batch_wrong_label_mask, facc, e, re_epochs, ctask_batch_size, bloss_weight, i, samp_labels, base_labels, emb_id, config, use_ce_loss=use_ce_loss)
            if e > 0:
                loss.backward(retain_graph=True)
                optimizer.step()

            benign_i += 1
            if benign_i >= math.ceil(float(len(images))/benign_batch_size):
                benign_i = 0

        flogits = np.concatenate(flogits, axis=0)
        fpreds = np.argmax(flogits, axis=2)

        preds = []
        plogits_np = []
        original_labels = []
        for k in range(fpreds.shape[0]):
            for j in range(fpreds.shape[1]):
                if obase_word_mask[k,j] == 1:
                    preds.append(fpreds[k,j])
                    plogits_np.append(flogits[k,j])
                    original_labels.append(oword_labels[k,j])
        preds = np.array(preds)
        plogits_np = np.array(plogits_np)

        # print(fpreds[:5])
        # print(preds)
        # print(obase_word_mask0[:5])
        # print(obase_word_mask[:5])
        # sys.exit()

        faccs = []
        use_labels = []
        optz_labels = []
        wrong_labels = []
        for k in range(ctask_batch_size):
            tpreds = preds[k*n_re_imgs_per_label:(k+1)*n_re_imgs_per_label]
            samp_label = samp_labels[k]
            base_label = base_labels[k]
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

            # if emb_id not in [0,1,3]:
            #     for i in range(tdelta0.shape[0]):
            #         for j in range(tdelta0.shape[1]):
            #             if tdelta0[i,j] < 1e-3:
            #                 gradual_small[i,j] = 0
            #     print('gradual_small', np.sum(gradual_small, axis=1))

            # tdelta0 = tdelta0 / np.minimum(np.sum(tdelta0, axis=1, keepdims=True), 10000)
            tdelta0 = tdelta0 / np.sum(tdelta0, axis=1, keepdims=True)
            print('tdelta0', tdelta0.shape, np.sum(tdelta0, axis=1), np.amax(tdelta0, axis=1) )
            # print('tdelta0', tdelta0[0,:20])
            tdelta1 = np.arctanh((tdelta0-0.5)*2)
            # tdelta1 = np.clip( np.arctanh((tdelta0-0.5)*2), -4, 4)
            print('before delta', delta.shape, torch.amax(delta, axis=1))
            delta.data = torch.FloatTensor(tdelta1).cuda().data
            print('after delta', delta.shape, torch.amax(delta, axis=1))
            word_delta1 = torch.reshape(\
                    # torch.matmul( (torch.tanh(torch.reshape(delta, (-1, delta_depth))) * 0.5 + 0.5) * random_idxs_split * torch.FloatTensor(gradual_small).cuda(), word_token_matrix),\
                    (torch.tanh(torch.reshape(delta, (-1, delta_depth))) * 0.5 + 0.5) * random_idxs_split * torch.FloatTensor(gradual_small).cuda(), \
                    (word_trigger_length, delta_depth))
            tword_delta1 = word_delta1.cpu().detach().numpy()
            print('tword_delta1', np.amax(tword_delta1, axis=1))
            optimizer.state = collections.defaultdict(dict)

            # sys.exit()

        # if emb_id in [0,1]:
        if False:
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
            best_ce_loss = ce_loss.cpu().detach().numpy()
            best_accs = faccs
            last_update = e

        if e == 0:
            base_logits_loss = logits_loss.cpu().detach().numpy()
            base_ce_loss = ce_loss.cpu().detach().numpy()

        if e % 10 == 0 or e == re_epochs-1 or last_update == e:

            print('epoch time', epoch_end_time - epoch_start_time)
            print(e, 'trigger_pos', trigger_pos, 'loss', loss.cpu().detach().numpy(), 'acc', faccs, 'base_labels', base_labels, 'sampling label', samp_labels,\
                    'optz label', optz_labels, 'use labels', use_labels,'wrong labels', wrong_labels,\
                    'logits_loss', logits_loss.cpu().detach().numpy(), 'ce loss', ce_loss.cpu().detach().numpy(),\
                    'benign_loss', benign_loss, 'bloss_weight', bloss_weight, 'delta_sum_loss_weight', delta_sum_loss_weight)
            if inner_outputs_a != None and inner_outputs_b != None:
                print('vloss1', vloss1.cpu().detach().numpy(), 'vloss2', vloss2.cpu().detach().numpy(),\
                    'relu_loss1', relu_loss1.cpu().detach().numpy(), 'max relu_loss1', np.amax(inner_outputs_a.cpu().detach().numpy()),\
                    'relu_loss2', relu_loss2.cpu().detach().numpy(),\
                    )
            print('logits', plogits[:5,:])
            print('labels', preds, 'original labels', original_labels)

            tuse_delta = use_delta.cpu().detach().numpy()
            if e == 0:
                tuse_delta0 = tuse_delta.copy()
            
            tword_delta = word_delta.cpu().detach().numpy()
            print('tword delta', tword_delta.shape, np.sum(tword_delta), np.sum(tuse_delta), np.sum(tuse_delta, axis=1), np.sum(tuse_delta < 1e-6))
            for j in range(tword_delta.shape[0]):
                for k in range(2):
                    print('position j', j, 'delta top', k, np.argsort(tword_delta[j])[-(k+1)], np.sort(tword_delta[j])[-(k+1)], '# larger than ', value_bound, np.sum(tword_delta[j] > value_bound))

            for j in range(tword_delta.shape[0]):
                for k in range(len(gt_trigger_idxs)):
                    print('gt trigger idxs', gt_trigger_idxs[k], 'position j', j, tword_delta[j][gt_trigger_idxs[k]], 'rank', np.where(np.argsort(tword_delta[j])[::-1] == gt_trigger_idxs[k])[0] )

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
    # delta = delta[:,-2:,:]

    # if is_test_arm:
    #     delta = delta - tuse_delta0

    print(delta.shape, use_delta.shape, mask.shape)

    # cleaning up
    for handle in handles:
        handle.remove()

    # return faccs, delta, word_delta, mask, optz_labels, logits_loss.cpu().detach().numpy() 
    return faccs, delta, word_delta, mask, optz_labels, logits_loss.cpu().detach().numpy(), logits_loss.cpu().detach().numpy() -base_logits_loss, ce_loss.cpu().detach().numpy(), base_ce_loss - ce_loss.cpu().detach().numpy()

def re_mask(model_type, models, benign_models, benign_logits0, benign_embeds, benign_poses, benign_attentions, benign_ys, benign_word_labels, embedding_signs, neuron_dict, children, embeds, poses_emb_vector, input_ids, attention_mask, labels, word_labels, original_words_list, original_labels_list, fys, n_neurons_dict, scratch_dirpath, re_epochs, num_classes, n_re_imgs_per_label, max_input_length, end_id, valid_base_labels, token_neighbours_dict, token_word_dict, word_token_dict, word_token_matrix, word_use_idxs, token_use_idxs, emb_id, max_nchecks, gt_trigger_idxs, neutral_words, use_idxs, config, for_submission, test_inputs, test_attentions, test_word_labels):

    re_mask_lr          = config['re_mask_lr']
    re_epochs           = config['re_epochs']
    word_trigger_length = config['word_trigger_length']
    re_batch_size       = config['re_batch_size']
    n_re_imgs_per_label = config['n_re_imgs_per_label']
    n_max_imgs_per_label= config['n_max_imgs_per_label']
    max_input_length    = config['max_input_length']
    top_k_candidates    = config['top_k_candidates']
    value_bound         = config['value_bound']
    reasr_bound         = config['reasr_bound']
    tasks_per_run       = config['tasks_per_run']
    device              = config['device']

    model, full_model, embedding, tokenizer = models[:4]
    if emb_id == 0:
        model, full_model, embedding, tokenizer, dbert_emb, dbert_transformer, depth = models
    else:
        model, full_model, embedding, tokenizer, bert_emb, depth = models

    random_use_idxs = word_use_idxs[:]
    random_idxs_split = np.zeros((1, len(word_use_idxs))) + 1
    if len(use_idxs) > 1:
        random_idxs_split = np.zeros(len(word_use_idxs))
        for i in range(len(use_idxs)):
            random_idxs_split[use_idxs[i]] = 1

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

        karm_losses = []

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

            # n_neurons = n_neurons_dict[tTroj_Layers[0]]
            n_neurons = 0
            
            # trigger_poses = [2]
            # trigger_poses = [0]
            trigger_poses = [0,1,2]
            for tp_i, trigger_pos in enumerate(trigger_poses):

                start_time = time.time()
                accs, rdeltas, rword_deltas, rmasks, optz_labels, logits_loss, logits_loss_diff, ce_loss, ce_loss_diff = reverse_engineer(model_type, models, benign_models, benign_logits0, benign_embeds, benign_poses, benign_attentions, benign_ys, benign_word_labels, embedding_signs, children, embeds, poses_emb_vector, attention_mask,  labels, word_labels, weights_file, Troj_Layer, tTroj_Neurons, tsamp_labels, tbase_labels, tre_epochs, num_classes, n_re_imgs_per_label, n_neurons, ctask_batch_size, max_input_length, trigger_pos, end_id, word_trigger_length, random_use_idxs, random_idxs_split, valid_base_labels, word_token_matrix, emb_id, gt_trigger_idxs, neutral_words, config)
                end_time = time.time()

                print('use time', end_time - start_time)

                # clear cache
                torch.cuda.empty_cache()

                k_arm_results.append([tbase_labels[0], tsamp_labels[0], trigger_pos, logits_loss, logits_loss_diff, ce_loss, ce_loss_diff])
                # k_arm_results.append([tbase_labels[0], tsamp_labels[0], trigger_pos, logits_loss, logits_loss_diff,])

                karm_losses.append((tbase_labels[0], tsamp_labels[0], trigger_pos, ce_loss))

        if not model_type.startswith('Roberta'):
            sorted_k_arm_results = sorted(k_arm_results, key=lambda x: x[4])[::-1]
        else:
            sorted_k_arm_results = sorted(k_arm_results, key=lambda x: x[6])[::-1]
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
        lists2 = []
        for k_arm_result in sorted_k_arm_results:
            tsl  = k_arm_result[0]
            ttl  = k_arm_result[1]
            tpos = k_arm_result[2]
            if tpos == 0:
                lists0.append((tsl, ttl, tpos))
            elif tpos == 1:
                lists1.append((tsl, ttl, tpos))
            else:
                lists2.append((tsl, ttl, tpos))

        final_top_k_arm_results = []
        for r_i in range(len(lists0)):
            for r_j in range(3):
                if r_j == 0:
                    final_top_k_arm_results.append(lists0[r_i])
                elif r_j == 1:
                    final_top_k_arm_results.append(lists1[r_i])
                else:
                    final_top_k_arm_results.append(lists2[r_i])

        # for debugging
        if len(trigger_poses) == 1:
            for k_arm_result in sorted_k_arm_results:
                tsl  = k_arm_result[0]
                ttl  = k_arm_result[1]
                tpos = k_arm_result[2]
                final_top_k_arm_results.append((tsl, ttl, tpos))

        print('final_top_k_arm_results', final_top_k_arm_results)

        # max_nchecks = 33
        if not model_type.startswith('Roberta'):
            max_nchecks = 18
        else:
            max_nchecks = 20
        final_top_k_arm_results = final_top_k_arm_results[:max_nchecks]

        print('final_top_k_arm_results', final_top_k_arm_results)

        # sys.exit()

        for k_arm_result in final_top_k_arm_results:
            tbase_labels = [k_arm_result[0]]
            tsamp_labels = [k_arm_result[1]]
            trigger_pos  = k_arm_result[2]

            print('------------------------------------------------testing', k_arm_result, '-----------------------------------------------------------')

            accs, rdeltas, rword_deltas, rmasks, optz_labels, logits_loss, logits_loss_diff, ce_loss, ce_loss_diff = reverse_engineer(model_type, models, benign_models, benign_logits0, benign_embeds, benign_poses, benign_attentions, benign_ys, benign_word_labels, embedding_signs, children, embeds, poses_emb_vector, attention_mask,  labels, word_labels, weights_file, Troj_Layer, tTroj_Neurons, tsamp_labels, tbase_labels, re_epochs, num_classes, n_re_imgs_per_label, n_neurons, ctask_batch_size, max_input_length, trigger_pos, end_id, word_trigger_length, random_use_idxs, random_idxs_split, valid_base_labels, word_token_matrix, emb_id, gt_trigger_idxs, neutral_words, config)
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

                if acc >= reasr_bound:
                    asrs, ces, rdelta_idxs, rdelta_words, = test_trigger_pos1(model_type, models, rword_delta, token_use_idxs, test_inputs, test_attentions, test_word_labels, original_words_list, original_labels_list, fys, valid_base_labels, base_label, token_neighbours_dict, token_word_dict, word_token_dict, neutral_words, config, benign_models)
                    final_result = (rdelta, rmask, optz_label, RE_img, RE_mask, RE_delta, samp_label, base_label, acc, trigger_pos, asrs, ces, rdelta_idxs, rdelta_words, ce_loss)
                    validated_results.append( final_result )
                    asrs = np.array(asrs)
                    find_trigger = False
                    print('find_trigger', find_trigger, asrs[0], asrs[1], np.reshape(asrs, -1), )
            # if find_trigger:
            #     break

        return validated_results, karm_losses


def test_trigger_pos1(model_type, models, rword_delta, use_idxs, input_ids, attention_mask, word_labels, original_words_list, original_labels_list, fys, valid_base_labels, base_label, token_neighbours_dict, token_word_dict, word_token_dict, neutral_words, config, benign_models):
    model, full_model, embedding, tokenizer = models[:4]

    re_mask_lr          = config['re_mask_lr']
    re_epochs           = config['re_epochs']
    word_trigger_length = config['word_trigger_length']
    re_batch_size       = config['re_batch_size']
    n_re_imgs_per_label = config['n_re_imgs_per_label']
    n_max_imgs_per_label= config['n_max_imgs_per_label']
    max_input_length    = config['max_input_length']
    top_k_candidates    = config['top_k_candidates']
    value_bound         = config['value_bound']
    reasr_bound         = config['reasr_bound']
    tasks_per_run       = config['tasks_per_run']
    device              = config['device']

    top_k_candidates0 = top_k_candidates

    print('rword delta', rword_delta.shape)
    # sys.exit()

    rdelta_idxs = []
    test_one_start = time.time()
    for j in range(rword_delta.shape[1]):
        rdelta_argsort = np.argsort(rword_delta[0][j])[::-1]
        for k in range(top_k_candidates0):
            rdelta_idxs.append( rdelta_argsort[k] )
    test_one_end = time.time()
    print('test get word time', test_one_end - test_one_start)
    print('rdelta_idxs', len(rdelta_idxs), rdelta_idxs)

    token_rdelta_idxs = []
    for idx in rdelta_idxs:
        # for j in range(word_token_matrix0.shape[1]):
            # token_rdelta_idxs.append(word_token_matrix0[idx, j])

        if model_type.startswith('Electra') or model_type.startswith('Distil') :
            tresults = tokenizer(neutral_words[idx], max_length=20, padding="max_length", truncation=True, return_tensors="pt")
            tinput_ids = tresults.data['input_ids'][0].cpu().detach().numpy()
            trigger_ids = []
            for i in range(len(tinput_ids)):
                if tinput_ids[i] not in [0,100,101,102,103]:
                    trigger_ids.append(tinput_ids[i])
        else:
            tresults = tokenizer('a '+ neutral_words[idx], max_length=20, padding="max_length", truncation=True, return_tensors="pt")
            tinput_ids = tresults.data['input_ids'][0].cpu().detach().numpy()
            trigger_ids = []
            for i in range(2,len(tinput_ids)):
                if tinput_ids[i] not in [50264, 50260, 0, 1, 2, 3]:
                    trigger_ids.append(tinput_ids[i])
        token_rdelta_idxs.append(tuple(trigger_ids))

    rdelta_idxs = token_rdelta_idxs

    rdelta_idxs = sorted(list(set(rdelta_idxs)), key=lambda x: len(x))

    print('token rdelta_idxs', len(rdelta_idxs), rdelta_idxs)

    rword_words = []
    for j in range(rword_delta.shape[1]):
        rdelta_argsort = np.argsort(rword_delta[0][j])[::-1]
        for k in range(top_k_candidates):
            rword_words.append( neutral_words[rdelta_argsort[k]] )
    test_one_end = time.time()
    print('test get word time', test_one_end - test_one_start)
    print('rword_words', rword_words)

    return test_token_word(model_type, models, rdelta_idxs, rword_words, use_idxs, input_ids, attention_mask, word_labels, original_words_list, original_labels_list, fys, valid_base_labels, base_label, config, benign_models)

def test_token_word(model_type, models, rdelta_idxs, rword_words, use_idxs, input_ids, attention_mask, word_labels, original_words_list, original_labels_list, fys, valid_base_labels, base_label, config, benign_models, test_emb=False):
    model, full_model, embedding, tokenizer = models[:4]

    re_mask_lr          = config['re_mask_lr']
    re_epochs           = config['re_epochs']
    word_trigger_length = config['word_trigger_length']
    re_batch_size       = config['re_batch_size']
    n_re_imgs_per_label = config['n_re_imgs_per_label']
    n_max_imgs_per_label= config['n_max_imgs_per_label']
    max_input_length    = config['max_input_length']
    top_k_candidates    = config['top_k_candidates']
    value_bound         = config['value_bound']
    reasr_bound         = config['reasr_bound']
    tasks_per_run       = config['tasks_per_run']
    device              = config['device']

    top_k_candidates0 = top_k_candidates

    crdelta_idxs = []
    for _ in rdelta_idxs:
        crdelta_idxs += list(_)

    rdelta_words = tokenizer.convert_ids_to_tokens( crdelta_idxs )

    input_ids = input_ids.cpu()

    max_len1_asrs1 = 0
    max_len1_asrs2 = 0
    max_len2_asrs1 = 0
    max_len2_asrs2 = 0
    max_len3_asrs1 = 0
    max_len3_asrs2 = 0
    max_len4_asrs1 = 0
    max_len4_asrs2 = 0

    min_len1_ces1 = 10
    min_len1_ces2 = 10
    min_len2_ces1 = 10
    min_len2_ces2 = 10
    min_len3_ces1 = 10
    min_len3_ces2 = 10
    min_len4_ces1 = 10
    min_len4_ces2 = 10

    stime1 = time.time()
    rcount1 = 0
    if True:
    # if False:
        len1_asrs1 = []
        len1_asrs2 = []
        len1_ces1 = []
        len1_ces2 = []

        for idx in rdelta_idxs:
            trigger_idxs = idx
            asrs, ces = inject_idx(tokenizer, full_model, input_ids, attention_mask, word_labels, trigger_idxs, valid_base_labels, base_label, config, benign_models, test_emb)
            rcount1 += 1
            len1_asrs1.append(max(asrs[0]))
            len1_asrs2.append(max(asrs[1]))
            len1_ces1.append(min(ces[0]))
            len1_ces2.append(min(ces[1]))

        len1_asrs1 = np.array(len1_asrs1)
        print('max len1 asrs1', np.amax(len1_asrs1), rdelta_idxs[np.argmax(len1_asrs1)], np.sort(len1_asrs1))
        print('min len1 ces1', np.amin(len1_ces1), rdelta_idxs[np.argmin(len1_ces1)], np.sort(len1_ces1))
        max_len1_asrs1 = np.amax(len1_asrs1)

        len1_asrs2 = np.array(len1_asrs2)
        print('max len1 asrs2', np.amax(len1_asrs2), rdelta_idxs[np.argmax(len1_asrs2)], np.sort(len1_asrs2))
        print('min len1 ces2', np.amin(len1_ces2), rdelta_idxs[np.argmin(len1_ces2)], np.sort(len1_ces2))
        max_len1_asrs2 = np.amax(len1_asrs2)

        min_len1_ces1 = np.amin(len1_ces1)
        min_len1_ces2 = np.amin(len1_ces2)

        cands1 = []
        cands2 = []
        cands = []
        for i in range(len(len1_asrs1)):
            if len1_asrs1[i] < 0 or len1_asrs2[i] < 0:
                continue
            if len1_asrs1[i] > 0.3 or len1_ces1[i] < 7 and not test_emb:
                cands1.append(rdelta_idxs[i])
                if rdelta_idxs[i] not in cands:
                    cands.append(rdelta_idxs[i])
        for i in range(len(len1_asrs2)):
            if len1_asrs1[i] < 0 or len1_asrs2[i] < 0:
                continue
            if len1_asrs2[i] > 0.3 or len1_ces2[i] < 7 and not test_emb:
                cands2.append(rdelta_idxs[i])
                if rdelta_idxs[i] not in cands:
                    cands.append(rdelta_idxs[i])

        if len(cands) > 0:
        # if False:
            width = 2
            # topks = list(np.array(rdelta_idxs)[np.argsort(len1_asrs1)[-width:]])
            # topks += list(np.array(rdelta_idxs)[np.argsort(len1_asrs2)[-width:]])
            topks = list(np.array(rdelta_idxs)[np.argsort(len1_ces1)[:width]])
            topks += list(np.array(rdelta_idxs)[np.argsort(len1_ces2)[:width]])
            topks = [tuple(_) for _ in topks]
            topks = sorted(list(set(topks)))
            print('topks', topks)
            print('cands', cands)
            len2_asrs1 = []
            len2_asrs2 = []
            len2_ces1 = []
            len2_ces2 = []
            len2_idxs = []
            for i in range(len(topks)):
                for j in range(len(cands)):
                    for k in range(2):
                    # for k in range(1):
                        if k == 0:
                            trigger_idxs = tuple(list(topks[i]) + list(cands[j]))
                        else:
                            trigger_idxs = tuple(list(cands[j]) + list(topks[i]))
                        print('trigger_idxs', trigger_idxs,'k', k)
                        # asrs, ces = inject_idx(tokenizer, full_model, input_ids, attention_mask, word_labels, trigger_idxs, valid_base_labels, base_label, config, benign_models)
                        asrs, ces = inject_idx(tokenizer, full_model, input_ids, attention_mask, word_labels, trigger_idxs, valid_base_labels, base_label, config, [], test_emb)
                        rcount1 += 1
                        len2_asrs1.append(max(asrs[0]))
                        len2_asrs2.append(max(asrs[1]))
                        len2_idxs.append(trigger_idxs)

                        len2_ces1.append(min(ces[0]))
                        len2_ces2.append(min(ces[1]))
            
            len2_asrs1 = np.array(len2_asrs1)
            print('max len2 asrs1', np.amax(len2_asrs1), tokenizer.convert_ids_to_tokens(len2_idxs[np.argmax(len2_asrs1)]) )
            max_len2_asrs1 = np.amax(len2_asrs1)
            max_len2_idxs1 = len2_idxs[np.argmax(len2_asrs1)]
            
            len2_asrs2 = np.array(len2_asrs2)
            print('max len2 asrs2', np.amax(len2_asrs2), tokenizer.convert_ids_to_tokens(len2_idxs[np.argmax(len2_asrs2)]) )
            max_len2_asrs2 = np.amax(len2_asrs2)
            max_len2_idxs2 = len2_idxs[np.argmax(len2_asrs2)]

            min_len2_ces1 = np.amin(len2_ces1)
            min_len2_ces2 = np.amin(len2_ces2)

    etime1 = time.time()

    asrs1 = [[max_len1_asrs1, max_len2_asrs1, max_len3_asrs1, max_len4_asrs1, ], [max_len1_asrs2, max_len2_asrs2, max_len3_asrs2, max_len4_asrs2, ]]

    ces1 = [[min_len1_ces1, min_len2_ces1, min_len3_ces1, min_len4_ces1, ], [ min_len1_ces2, min_len2_ces2, min_len3_ces2, min_len4_ces2, ]]

    max_len1_asrs1 = 0
    max_len1_idxs1 = []
    max_len1_asrs2 = 0
    max_len1_idxs2 = []
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

    min_len1_ces1 = 10
    min_len1_ces2 = 10
    min_len2_ces1 = 10
    min_len2_ces2 = 10
    min_len3_ces1 = 10
    min_len3_ces2 = 10
    min_len4_ces1 = 10
    min_len4_ces2 = 10

    rcount2 = 0
    stime2 = time.time()
    if False:
    # if True:
        len1_asrs1 = []
        len1_asrs2 = []
        len1_ces1 = []
        len1_ces2 = []
        for word in rword_words:
            trigger_words = [word]
            asrs, ces = inject_word(tokenizer, full_model, original_words_list, original_labels_list, fys, trigger_words, valid_base_labels, base_label, config)
            rcount2 += 1
            len1_asrs1.append(max(asrs[0]))
            len1_asrs2.append(max(asrs[1]))
            len1_ces1.append(min(ces[0]))
            len1_ces2.append(min(ces[1]))

        len1_asrs1 = np.array(len1_asrs1)
        max_len1_asrs1 = np.amax(len1_asrs1)

        print('max len1 asrs2', np.amax(len1_asrs2), rword_words[np.argmax(len1_asrs2)])
        max_len1_asrs2 = np.amax(len1_asrs2)

        min_len1_ces1 = np.amin(len1_ces1)
        min_len1_ces2 = np.amin(len1_ces2)

        cands1 = []
        cands2 = []
        cands = []
        for i in range(len(len1_asrs1)):
            if len1_asrs1[i] > 0.3 or len1_ces1[i] < 6 and not test_emb:
                cands1.append(rword_words[i])
                if rword_words[i] not in cands:
                    cands.append(rword_words[i])
        for i in range(len(len1_asrs2)):
            if len1_asrs2[i] > 0.3 or len1_ces2[i] < 6 and not test_emb:
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
            len2_ces1 = []
            len2_ces2 = []
            len2_idxs = []
            for i in range(len(topks)):
                for j in range(len(cands)):
                    for k in range(2):
                    # for k in range(1):
                        if k == 0:
                            trigger_words = list([topks[i]]) + list([cands[j]])
                        else:
                            trigger_words = list([cands[j]]) + list([topks[i]])
                        print('trigger_words', trigger_words, 'k', k)
                        asrs, ces = inject_word(tokenizer, full_model, original_words_list, original_labels_list, fys, trigger_words, valid_base_labels, base_label, config)
                        rcount2 += 1
                        len2_asrs1.append(max(asrs[0]))
                        len2_asrs2.append(max(asrs[1]))
                        len2_idxs.append(trigger_words)

                        len2_ces1.append(min(ces[0]))
                        len2_ces2.append(min(ces[1]))
            
            len2_asrs1 = np.array(len2_asrs1)
            print('max len2 asrs1', np.amax(len2_asrs1), len2_idxs[np.argmax(len2_asrs1)] )
            max_len2_asrs1 = np.amax(len2_asrs1)
            max_len2_idxs1 = len2_idxs[np.argmax(len2_asrs1)]
            
            len2_asrs2 = np.array(len2_asrs2)
            print('max len2 asrs2', np.amax(len2_asrs2), len2_idxs[np.argmax(len2_asrs2)] )
            max_len2_asrs2 = np.amax(len2_asrs2)
            max_len2_idxs2 = len2_idxs[np.argmax(len2_asrs2)]

            min_len2_ces1 = np.amin(len2_ces1)
            min_len2_ces2 = np.amin(len2_ces2)
    etime2 = time.time()

    print('test time', etime1 - stime1, etime2 - stime2, rcount1, rcount2)
    # sys.exit()

    asrs2 = [[max_len1_asrs1, max_len2_asrs1, max_len3_asrs1, max_len4_asrs1, ], [max_len1_asrs2, max_len2_asrs2, max_len3_asrs2, max_len4_asrs2, ]]

    ces2 = [[min_len1_ces1, min_len2_ces1, min_len3_ces1, min_len4_ces1, ], [ min_len1_ces2, min_len2_ces2, min_len3_ces2, min_len4_ces2, ]]

    asrs1 = np.array(asrs1).reshape((2,4,1))
    asrs2 = np.array(asrs2).reshape((2,4,1))
    asrs = np.amax( np.concatenate([asrs1, asrs2], axis=2), axis=2 )
    print('asrs1', asrs1, 'asrs2', asrs2, 'asrs', asrs)

    ces1 = np.array(ces1).reshape((2,4,1))
    ces2 = np.array(ces2).reshape((2,4,1))
    ces = np.amin( np.concatenate([ces1, ces2], axis=2), axis=2 )
    print('ces1', ces1, 'ces2', ces2, 'ces', ces)

    # sys.exit()
    print('-'*30)

    input_ids = input_ids.cuda()

    return asrs, ces, rdelta_idxs, rdelta_words

def calc_fasrs(full_model, input_ids_list, word_labels_list, attention_mask, valid_base_labels, base_label, config, benign_models=[], test_emb=False):

    re_mask_lr          = config['re_mask_lr']
    re_epochs           = config['re_epochs']
    word_trigger_length = config['word_trigger_length']
    re_batch_size       = config['re_batch_size']
    n_re_imgs_per_label = config['n_re_imgs_per_label']
    n_max_imgs_per_label= config['n_max_imgs_per_label']
    max_input_length    = config['max_input_length']
    top_k_candidates    = config['top_k_candidates']
    value_bound         = config['value_bound']
    reasr_bound         = config['reasr_bound']
    tasks_per_run       = config['tasks_per_run']
    device              = config['device']


    batch_size = 20

    fasrs = []
    fces = []
    for k in range(len(input_ids_list)):
        asrs = []
        ces = []
        input_ids3 = input_ids_list[k]
        word_labels3 = word_labels_list[k]
        # print(input_ids3.shape, attention_mask.shape)
        flogits = []
        flogits_np = []
        start_time = time.time()
        benign_logitses = []
        for i in range(math.ceil(input_ids3.shape[0]/float(batch_size))):
            logits = full_model(input_ids=input_ids3[i*batch_size:(i+1)*batch_size], attention_mask=attention_mask[i*batch_size:(i+1)*batch_size],).logits
            if len(benign_models) > 0:
                blogits = benign_models[0](input_ids=input_ids3[i*batch_size:(i+1)*batch_size], attention_mask=attention_mask[i*batch_size:(i+1)*batch_size],).logits
                benign_logitses.append(blogits.cpu().detach().numpy())
            flogits.append(logits)
            flogits_np.append(logits.cpu().detach().numpy())
        end_time = time.time()
        logits_np = np.concatenate(flogits_np, axis=0)
        logits = torch.cat(flogits, axis=0)
        if len(benign_models) > 0:
            benign_logitses = np.concatenate(benign_logitses, axis=0)
            benign_preds = np.argmax(benign_logitses, axis=2)
            bpreds = []
        fpreds = np.argmax(logits_np, axis=2)
        fys = []
        preds = []
        flogits = []
        for i in range(word_labels3.shape[0]):
            for j in range(word_labels3.shape[1]):
                if word_labels3[i,j] in valid_base_labels:
                    fys.append(word_labels3[i,j])
                    preds.append(fpreds[i,j])
                    flogits.append(logits[i,j])
                if len(benign_models) > 0:
                    if word_labels3[i,j] == base_label:
                        bpreds.append(benign_preds[i,j])
            # if base_label in word_labels3[i]:
            #     print(attention_mask[i])
            #     print(input_ids[i])
            #     print(input_ids3[i])
            #     print(word_labels[i])
            #     print(word_labels3[i])
            #     print(fpreds[i])
        # sys.exit()
        # print('eval time', end_time - start_time, len(preds),)
        fys = np.array(fys)
        preds = np.array(preds)
        # print('fys', fys)
        # print('preds', preds)

        if len(benign_models) > 0:
        # if False:
            bpreds = np.array(bpreds)
            if test_emb:
                benign_acc = np.sum(bpreds == base_label)/ float(len(bpreds))
            else:
                cor = 0
                for i in range(len(bpreds)):
                    if bpreds[i] == base_label or bpreds[i] == base_label + 1:
                        cor += 1
                benign_acc = cor / float(len(bpreds))
            print('beingn acc', benign_acc, base_label, len(bpreds), bpreds)
            if benign_acc < 0.8:
                asrs = [-1  for _ in valid_base_labels]
                ces  = [100 for _ in valid_base_labels]
                fasrs.append(asrs)
                fces.append(ces)
                continue

        for tbase_label in valid_base_labels:
            test_idxs = np.array(np.where(fys==tbase_label)[0])
            tbpreds = preds[test_idxs]
            # tbpreds = (tbpreds + 1 ) // 2
            if len(tbpreds) == 0:
                asrs.append(0)
                ces.append(10)
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
            # print('positions', k, 'source class', tbase_label, 'target label', optz_label, 'score', acc, tbpreds)
            asrs.append(acc)

            if target_label > 0:
                logits = []
                for m in range(len(fys)):
                    if fys[m] == tbase_label:
                        logits.append(flogits[m])
                logits = torch.stack(logits).cuda()
                target_labels = torch.LongTensor(np.array([target_label for _ in range(logits.shape[0])])).cuda()
                # print('logits', logits.shape, target_labels)
                ce_loss = F.nll_loss(F.log_softmax(logits, dim=1),  target_labels).cpu().detach().numpy().reshape(-1)[0]
                # print('logits', logits.shape, target_labels, ce_loss)
            else:
                ce_loss = 10
            ces.append(ce_loss)
            # sys.exit()

        fasrs.append(asrs)
        fces.append(ces)

    return fasrs, fces


def inject_word(tokenizer, full_model, original_words_list, original_labels_list, fys, trigger_words, valid_base_labels, base_label, config):


    re_mask_lr          = config['re_mask_lr']
    re_epochs           = config['re_epochs']
    word_trigger_length = config['word_trigger_length']
    re_batch_size       = config['re_batch_size']
    n_re_imgs_per_label = config['n_re_imgs_per_label']
    n_max_imgs_per_label= config['n_max_imgs_per_label']
    max_input_length    = config['max_input_length']
    top_k_candidates    = config['top_k_candidates']
    value_bound         = config['value_bound']
    reasr_bound         = config['reasr_bound']
    tasks_per_run       = config['tasks_per_run']
    device              = config['device']

    start_time = time.time()
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
        tinput_ids, tattention_mask, tlabels, tlabels_mask = tokenize_and_align_labels(tokenizer, original_words, original_labels, max_input_length)
        # input_ids, attention_mask, labels, labels_mask = manual_tokenize_and_align_labels(tokenizer, original_words, original_labels, max_input_length)

        # print('original_words', original_words)
        # print('original_labels', original_labels)
        # print(len(tinput_ids), len(tattention_mask), len(tlabels), len(tlabels_mask))
        # print(tinput_ids, tattention_mask, tlabels, tlabels_mask)

        input_ids1.append(torch.as_tensor(tinput_ids).unsqueeze(0))
        attention_mask1.append(torch.as_tensor(tattention_mask).unsqueeze(0))
        word_labels1.append(tlabels)
    input_ids1 = torch.cat(input_ids1, axis=0)
    attention_mask1 = torch.cat(attention_mask1, axis=0)
    word_labels1 = np.array(word_labels1)
    input_ids1 = input_ids1.to(device)
    attention_mask1 = attention_mask1.to(device)

    # print('injected', input_ids1[0])
    # print('word labels', word_labels1[0])
    # print('original_labels', original_labels)

    input_ids3 = []
    attention_mask3 = []
    word_labels3 = []
    for i in range(len(original_words_list)):
        original_words = original_words_list[i][:]
        original_labels = original_labels_list[i][:]

        # original_words += trigger_words
        # original_labels += [0 for _ in trigger_words]

        for tword in trigger_words:
            original_words.insert(-1, tword)
            original_labels.insert(-1, 0)

        # Select your preference for tokenization
        tinput_ids, tattention_mask, tlabels, tlabels_mask = tokenize_and_align_labels(tokenizer, original_words, original_labels, max_input_length)
        # input_ids, attention_mask, labels, labels_mask = manual_tokenize_and_align_labels(tokenizer, original_words, original_labels, max_input_length)

        # print('original_words', original_words)
        # print('original_labels', original_labels)
        # print(len(tinput_ids), len(tattention_mask), len(tlabels), len(tlabels_mask))
        # print(tinput_ids, tattention_mask, tlabels, tlabels_mask)

        input_ids3.append(torch.as_tensor(tinput_ids).unsqueeze(0))
        attention_mask3.append(torch.as_tensor(tattention_mask).unsqueeze(0))
        word_labels3.append(tlabels)

    input_ids3 = torch.cat(input_ids3, axis=0)
    attention_mask3 = torch.cat(attention_mask3, axis=0)
    word_labels3 = np.array(word_labels3)
    input_ids3 = input_ids3.to(device)
    attention_mask3 = attention_mask3.to(device)

    input_ids_list = [input_ids1, input_ids3]
    word_labels_list = [word_labels1, word_labels3]

    # sys.exit()

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
        tinput_ids, tattention_mask, tlabels, tlabels_mask = tokenize_and_align_labels(tokenizer, original_words, original_labels, max_input_length)
        # input_ids, attention_mask, labels, labels_mask = manual_tokenize_and_align_labels(tokenizer, original_words, original_labels, max_input_length)

        # make sure only the injected place is counted
        find_base_label = False
        for i in range(len(tlabels)):
            if not find_base_label and tlabels[i] == base_label:
                find_base_label = True
                continue
            if find_base_label and tlabels[i] == base_label:
                tlabels[i] = 0

        # print(len(input_ids), len(attention_mask), len(labels), len(labels_mask))
        # print(input_ids, attention_mask, labels, labels_mask)

        input_ids2.append(torch.as_tensor(tinput_ids).unsqueeze(0))
        attention_mask2.append(torch.as_tensor(tattention_mask).unsqueeze(0))
        word_labels2.append(tlabels)

        # print(original_words)
        # print(original_labels)
        # print(tlabels)

    input_ids2 = torch.cat(input_ids2, axis=0)
    attention_mask2 = torch.cat(attention_mask2, axis=0)
    word_labels2 = np.array(word_labels2)
    input_ids2 = input_ids2.to(device)
    attention_mask2 = attention_mask2.to(device)

    input_ids_list.append( input_ids2 )
    word_labels_list.append( word_labels2 )

    # sys.exit()

    # print(input_ids1)
    # print(word_labels1)
    end_time = time.time()

    print('gen data time', end_time - start_time)

    start_time = time.time()
    fasrs, fces = calc_fasrs(full_model, input_ids_list, word_labels_list, attention_mask1, valid_base_labels, base_label, config)
    end_time = time.time()

    print('time', end_time - start_time)

    fasrs = np.array(fasrs)
    print('fasrs', fasrs)
    nfasrs=np.array([ np.amax(fasrs[:2], axis=0), np.amax(fasrs[2:], axis=0) ])
    fasrs = nfasrs
    print('updated fasrs', fasrs)

    fces = np.array(fces)
    # print('fces', fces)
    nfces=np.array([ np.amin(fces[:2], axis=0), np.amin(fces[2:], axis=0) ])
    fces = nfces
    print('updated fces', fces)

    print('test trigger words', trigger_words, 'asrs', fasrs)

    return fasrs, fces

def inject_idx(tokenizer, full_model, input_ids, attention_mask, word_labels, trigger_idxs, valid_base_labels, base_label, config, benign_models, test_emb):


    re_mask_lr          = config['re_mask_lr']
    re_epochs           = config['re_epochs']
    word_trigger_length = config['word_trigger_length']
    re_batch_size       = config['re_batch_size']
    n_re_imgs_per_label = config['n_re_imgs_per_label']
    n_max_imgs_per_label= config['n_max_imgs_per_label']
    max_input_length    = config['max_input_length']
    top_k_candidates    = config['top_k_candidates']
    value_bound         = config['value_bound']
    reasr_bound         = config['reasr_bound']
    tasks_per_run       = config['tasks_per_run']
    device              = config['device']

    # print('word labels', word_labels)

    attention_mask = attention_mask.copy()

    start_time = time.time()
    # trigger_idxs = list(trigger_idxs)
    trigger_idxs = torch.LongTensor(np.array(trigger_idxs))

    # word_labels = torch.LongTensor(word_labels)

    input_ids1 = torch.zeros_like(input_ids)
    input_ids3 = torch.zeros_like(input_ids)
    word_labels1 = np.zeros_like(word_labels)
    word_labels3 = np.zeros_like(word_labels)
    end_poses3 = []
    for i in range(attention_mask.shape[0]):
        find_word3 = False
        for j in range(attention_mask.shape[1]-2-len(trigger_idxs)):
            if attention_mask[i,j+1] == 0:
                end_poses3.append(j)
                find_word3 = True
                break
        if not find_word3:
            for att_i in range(attention_mask.shape[1]):
                if attention_mask[i,att_i] == 0:
                    break
            end_poses3.append(min(att_i-2, attention_mask.shape[1]-len(trigger_idxs)-2))

        # print('i', attention_mask[i], end_poses3[i])

    for i in range(attention_mask.shape[0]):
        for att_i in range(attention_mask.shape[1]):
            if attention_mask[i,att_i] == 0:
                break
        attention_mask[i,att_i:att_i+len(trigger_idxs)] = 1
    
    roll_ids_list = []
    for k in range(input_ids.shape[0]):
        roll_ids = torch.roll(input_ids[k], len(trigger_idxs), dims=0)
        roll_labels = np.roll(word_labels[k], len(trigger_idxs), axis=0)
        roll_ids_list.append(roll_ids)

        input_ids1[k,:1] = input_ids[k,:1]
        # for m in range(len(trigger_idxs)):
        #     input_ids1[k,1+m] = trigger_idxs[m]
        input_ids1[k,1:1+len(trigger_idxs)] = trigger_idxs
        input_ids1[k,1+len(trigger_idxs):] = roll_ids[1+len(trigger_idxs):]

        epos = end_poses3[k]
        input_ids3[k,:epos] = input_ids[k,:epos]
        # for m in range(len(trigger_idxs)):
        #     input_ids3[k,epos+m] = trigger_idxs[m]
        input_ids3[k,epos:epos+len(trigger_idxs)] = trigger_idxs
        input_ids3[k,epos+len(trigger_idxs):] = roll_ids[epos+len(trigger_idxs):]

        word_labels3[k,:epos] = word_labels[k,:epos]
        # for m in range(len(trigger_idxs)):
        #     word_labels3[k,epos+m] = 0
        word_labels3[k,epos:epos+len(trigger_idxs)] = 0
        word_labels3[k,epos+len(trigger_idxs):] = roll_labels[epos+len(trigger_idxs):]

        word_labels1[k,:1] = word_labels[k,:1]
        # for m in range(len(trigger_idxs)):
        #     word_labels1[k,1+m] = 0
        word_labels1[k,epos:epos+len(trigger_idxs)] = 0
        word_labels1[k,1+len(trigger_idxs):] = roll_labels[1+len(trigger_idxs):]

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

    input_ids1 = input_ids1.to(device)
    input_ids3 = input_ids3.to(device)

    input_ids_list = [input_ids1, input_ids3, ]
    word_labels_list = [word_labels1, word_labels3, ]

    end_poses = []
    input_ids2 = torch.zeros_like(input_ids)
    word_labels2 = np.zeros_like(word_labels)
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
        roll_ids = roll_ids_list[k]

        epos = end_poses[k]
        input_ids2[k,:epos] = input_ids[k,:epos]
        # for m in range(len(trigger_idxs)):
        #     input_ids2[k,epos+m] = trigger_idxs[m]
        input_ids2[k,epos:epos+len(trigger_idxs)] = trigger_idxs
        input_ids2[k,epos+len(trigger_idxs):] = roll_ids[epos+len(trigger_idxs):]

    input_ids2 = input_ids2.to(device)

    # input_ids_list.append( input_ids3 )
    # word_labels_list.append( word_labels3 )
    input_ids_list.append( input_ids2 )
    word_labels_list.append( word_labels2 )
    end_time = time.time()

    attention_mask = torch.LongTensor(attention_mask)
    attention_mask = attention_mask.to(device)

    print('gen data time', end_time - start_time)

    start_time = time.time()
    fasrs, fces = calc_fasrs(full_model, input_ids_list, word_labels_list, attention_mask, valid_base_labels, base_label, config, benign_models, test_emb)
    end_time = time.time()

    print('time', end_time - start_time)

    fasrs = np.array(fasrs)
    print('fasrs', fasrs)
    nfasrs=np.array([ np.amax(fasrs[:2], axis=0), np.amax(fasrs[2:], axis=0) ])
    fasrs = nfasrs
    print('updated fasrs', fasrs)

    fces = np.array(fces)
    # print('fces', fces)
    nfces=np.array([ np.amin(fces[:2], axis=0), np.amin(fces[2:], axis=0) ])
    fces = nfces
    print('updated fces', fces)

    print('test trigger idxs', trigger_idxs, tokenizer.convert_ids_to_tokens( trigger_idxs ), 'asrs', fasrs)

    return fasrs, fces


def ner_trojan_detector(model_filepath, tokenizer_filepath, result_filepath, scratch_dirpath, examples_dirpath, round_training_dataset_dirpath, learned_parameters_dirpath, features_filepath, parameters):
    start = time.time()

    
    print('ner parameters', parameters)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for_submission = parameters[0]
    is_configure   = parameters[1]
    config = {}
    config['re_mask_lr']            = parameters[2]
    config['re_epochs']             = parameters[3]
    config['word_trigger_length']   = parameters[4]
    config['re_batch_size']         = 20
    config['n_re_imgs_per_label']   = 20
    config['n_max_imgs_per_label']  = 20
    config['max_input_length']      = 80
    config['top_k_candidates']      = 12
    config['reasr_bound']           = 0.8
    config['value_bound']           = 0.5
    config['tasks_per_run']         = 1
    config['bloss_weight']          = 1e0
    # config['bloss_weight']          = 1e1
    config['device']                = device
    config['logfile'] = '{0}/result_r9_1_1_4_1.txt'.format(scratch_dirpath)

    re_mask_lr          = config['re_mask_lr']
    re_epochs           = config['re_epochs']
    word_trigger_length = config['word_trigger_length']
    re_batch_size       = config['re_batch_size']
    n_re_imgs_per_label = config['n_re_imgs_per_label']
    n_max_imgs_per_label= config['n_max_imgs_per_label']
    max_input_length    = config['max_input_length']
    top_k_candidates    = config['top_k_candidates']
    value_bound         = config['value_bound']
    reasr_bound         = config['reasr_bound']
    tasks_per_run       = config['tasks_per_run']
    device              = config['device']

    print('ner config', config)

    # load the classification model and move it to the GPU
    # model = torch.load(model_filepath, map_location=torch.device('cuda'))
    full_model = torch.load(model_filepath, map_location=torch.device(device))
    # embedding = full_model.transformer
    # word_embeddings = embedding.embeddings.word_embeddings
    # depth = word_embeddings.num_embeddings

    target_layers = []
    model_type = full_model.__class__.__name__
    children = list(full_model.children())
    print('model type', model_type)
    print('children', list(full_model.children()))
    # print('named_modules', list(full_model.named_modules()))
    word_embedding = full_model.get_input_embeddings().weight 
    print('word emebdding shape', word_embedding.shape)
    embedding = list(full_model.children())[0]

    model = full_model.classifier
    num_classes = list(model.named_modules())[0][1].out_features
    print('num_classes', num_classes)
    word_trigger_length = config['word_trigger_length']

    # same for the 3 basic types
    children = list(model.children())
    target_layers = ['Linear']

    if for_submission:
        all_file                            = '/all_words_amazon_100k_0.txt'
        char_file                           = '/special_char.txt'
        dbert_word_token_dict_fname         = '/distilbert_amazon_100k_word_token_dict6.pkl'
        dbert_token_word_dict_fname         = '/distilbert_amazon_100k_token_word_dict6.pkl'
        dbert_token_neighbours_dict_fname   = '/distilbert_amazon_100k_token_neighbours6.pkl'
        bert_word_token_dict_fname          = '/google_amazon_100k_word_token_dict6.pkl'
        bert_token_word_dict_fname          = '/google_amazon_100k_token_word_dict6.pkl'
        bert_token_neighbours_dict_fname    = '/google_amazon_100k_token_neighbours6.pkl'
        roberta_word_token_dict_fname       = '/roberta_base_amazon_100k_word_token_dict6.pkl'
        roberta_token_word_dict_fname       = '/roberta_base_amazon_100k_token_word_dict6.pkl'
        roberta_token_neighbours_dict_fname = '/roberta_base_amazon_100k_token_neighbours6.pkl'
        dbert_word_token_matrix_fname       = '/distilbert_amazon_100k_word_token_matrix6_4tokens_trigger.pkl'
        bert_word_token_matrix_fname        = '/google_amazon_100k_word_token_matrix6_4tokens_trigger.pkl'
        roberta_word_token_matrix_fname     = '/roberta_base_amazon_100k_word_token_matrix6_4tokens_trigger.pkl'
    else:
        dbert_word_token_dict_fname         = '../distilbert_amazon_100k_word_token_dict6.pkl'
        dbert_token_word_dict_fname         = '../distilbert_amazon_100k_token_word_dict6.pkl'
        dbert_token_neighbours_dict_fname   = '../distilbert_amazon_100k_token_neighbours6.pkl'
        bert_word_token_dict_fname          = '../google_amazon_100k_word_token_dict6.pkl'
        bert_token_word_dict_fname          = '../google_amazon_100k_token_word_dict6.pkl'
        bert_token_neighbours_dict_fname    = '../google_amazon_100k_token_neighbours6.pkl'
        roberta_word_token_dict_fname       = '../roberta_base_amazon_100k_word_token_dict6.pkl'
        roberta_token_word_dict_fname       = '../roberta_base_amazon_100k_token_word_dict6.pkl'
        roberta_token_neighbours_dict_fname = '../roberta_base_amazon_100k_token_neighbours6.pkl'
        dbert_word_token_matrix_fname       = '../distilbert_amazon_100k_word_token_matrix6_4tokens_trigger.pkl'
        bert_word_token_matrix_fname        = '../google_amazon_100k_word_token_matrix6_4tokens_trigger.pkl'
        roberta_word_token_matrix_fname     = '../roberta_base_amazon_100k_word_token_matrix6_4tokens_trigger.pkl'
        all_file                            = '../all_words_amazon_100k_0.txt'
        char_file                           = '../special_char.txt'

        if model_type.startswith('Electra'):
            tokenizer_filepath = '/data/share/trojai/trojai-round9-v2-dataset/tokenizers/google-electra-small-discriminator.pt'
        elif model_type.startswith('Roberta'):
            tokenizer_filepath = '/data/share/trojai/trojai-round9-v2-dataset/tokenizers/roberta-base.pt'
        elif model_type.startswith('DistilBert'):
            tokenizer_filepath = '/data/share/trojai/trojai-round9-v2-dataset/tokenizers/distilbert-base-cased.pt'
        else:
            print('error', model_type)
            sys.exit()

    if model_type.startswith('Electra'):
        word_token_matrix0 = pickle.load(open(bert_word_token_matrix_fname, 'rb'))
    elif model_type.startswith('Roberta'):
        word_token_matrix0 = pickle.load(open(roberta_word_token_matrix_fname, 'rb'))
    elif model_type.startswith('DistilBert'):
        word_token_matrix0 = pickle.load(open(dbert_word_token_matrix_fname, 'rb'))
    else:
        print('error', model_type)
        sys.exit()

    neutral_words = [] 
    for line in open(all_file):
        neutral_words.append(line.split()[0])

    fns = [os.path.join(examples_dirpath, fn) for fn in os.listdir(examples_dirpath) if fn == 'clean-example-data.json']
    fns.sort()
    examples_filepath = fns[0]

    with open(examples_filepath) as json_file:
        clean0_json = json.load(json_file)
    clean0_filepath = os.path.join(scratch_dirpath,'clean0_data.json')
    with open(clean0_filepath, 'w') as f:
        json.dump(clean0_json, f)
    dataset = datasets.load_dataset('json', data_files=[clean0_filepath], field='data', keep_in_memory=True, split='train', cache_dir=os.path.join(scratch_dirpath, '.cache'))

    valid_base_labels = [_ for _ in range(num_classes) if _%2 == 1]

    original_words_list = []
    original_labels_list = []
    fys = []
    for data_item in dataset:
        # print(data_item)
        
        original_words = data_item['tokens']
        original_labels = data_item['ner_tags']

        original_words_list.append(original_words)
        original_labels_list.append(original_labels)

        for l in original_labels:
            if l in valid_base_labels:
                fys.append(l)
                break

    fys = np.array(fys)

    # # # Inference the example images in data
    # fns = [os.path.join(examples_dirpath, fn) for fn in os.listdir(examples_dirpath) if fn.endswith('.txt')]
    # fns.sort()  # ensure file ordering

    # original_words_list = []
    # original_labels_list = []
    # fys = []
    # valid_base_labels =[]
    # for fn in fns:
    #     # For this example we parse the raw txt file to demonstrate tokenization. Can use either
    #     if fn.endswith('_tokenized.txt'):
    #         continue
    #     # load the example

    #     fy = int(fn.split('/')[-1].split('_')[1])
    #     fys.append(fy)
    #     valid_base_labels.append(fy)
        
    #     original_words = []
    #     original_labels = []
    #     with open(fn, 'r') as fh:
    #         lines = fh.readlines()
    #         for line in lines:
    #             split_line = line.split('\t')
    #             word = split_line[0].strip()
    #             label = split_line[2].strip()
                
    #             original_words.append(word)
    #             original_labels.append(int(label))

    #     original_words_list.append(original_words)
    #     original_labels_list.append(original_labels)

    # fys = np.array(fys)
    # valid_base_labels = sorted(list(set(valid_base_labels)))

    print('fys', fys, 'valid base labels', valid_base_labels)

    tokenizer = torch.load(tokenizer_filepath)

    # set the padding token if its undefined
    if not hasattr(tokenizer, 'pad_token') or tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # round9-v1 tokenizer does not has add_prefix_space to be True
    if tokenizer.name_or_path=="roberta-base":
        tokenizer.add_prefix_space=True

    print('tokenizer', tokenizer.__class__.__name__,  'max_input_length', max_input_length)

    depth = tokenizer.vocab_size

    # one_hot_emb = nn.Embedding(depth, depth)
    # one_hot_emb.weight.data = torch.eye(depth)
    # one_hot_emb = one_hot_emb

    full_model.eval()
    model.eval()

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
        tinput_ids, tattention_mask, tlabels, tlabels_mask = tokenize_and_align_labels(tokenizer, original_words, original_labels, max_input_length)
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
    print('word labels', word_labels[0])

    # sys.exit()

    logits = full_model(input_ids=input_ids, attention_mask=attention_mask,).logits
    print('logits', logits.shape)
    logits = logits.cpu().detach().numpy()

    load_start = time.time()
    if model_type.startswith('Electra'):
        end_id = 102
        end_attention = 1
        benign_names = ''
        max_nchecks = 14
        emb_id = 1
        word_token_matrix0 = pickle.load(open(bert_word_token_matrix_fname, 'rb'))
        word_token_dict = pickle.load(open(bert_word_token_dict_fname, 'rb'))
        token_word_dict = pickle.load(open(bert_token_word_dict_fname, 'rb'))
        token_neighbours_dict = pickle.load(open(bert_token_neighbours_dict_fname, 'rb'))
        token_use_idxs = list(range(30522))
    elif model_type.startswith('DistilBert'):
        end_id = 102
        end_attention = 1
        benign_names = ''
        # max_nchecks = 18
        max_nchecks = 20
        emb_id = 0
        word_token_matrix0 = pickle.load(open(dbert_word_token_matrix_fname, 'rb'))
        word_token_dict = pickle.load(open(dbert_word_token_dict_fname, 'rb'))
        token_word_dict = pickle.load(open(dbert_token_word_dict_fname, 'rb'))
        token_neighbours_dict = pickle.load(open(dbert_token_neighbours_dict_fname, 'rb'))
        token_use_idxs = list(range(28996))
    elif model_type.startswith('Roberta'):
        end_id = 1
        end_attention = 1
        # benign_names = 'id-00000140'
        benign_names = ''
        max_nchecks = 14
        emb_id = 2
        word_token_matrix0 = pickle.load(open(roberta_word_token_matrix_fname, 'rb'))
        word_token_dict = pickle.load(open(roberta_word_token_dict_fname, 'rb'))
        token_word_dict = pickle.load(open(roberta_token_word_dict_fname, 'rb'))
        token_neighbours_dict = pickle.load(open(roberta_token_neighbours_dict_fname, 'rb'))
        token_use_idxs = list(range(50265))
    else:
        print('error embedding type', embedding.__class__.__name__)
        sys.exit()

    benign_names_file = learned_parameters_dirpath + '/benign_names.txt'
    benign_names_dict = {}
    for line in open(benign_names_file):
        if not line.startswith('ner1'):
            continue
        words = line.split()
        if len(words) < 3:
            continue
        type_name = words[1]
        benign_names_dict[type_name] = words[2]
    try:
        if model_type.startswith('Electra'):
            benign_names = benign_names_dict['electra']
        elif model_type.startswith('DistilBert'):
            benign_names = benign_names_dict['dbert']
        elif model_type.startswith('Roberta'):
            benign_names = benign_names_dict['roberta']
    except:
        print('error in benign_names', benign_names_dict, model_type)
        benign_names = ''
    print('benign_names = {}'.format(benign_names))

    word_dict_depth = word_token_matrix0.shape[0]
    word_token_length = word_token_matrix0.shape[1]
    word_use_idxs = list(range(word_dict_depth))

    trigger_length = word_trigger_length * word_token_length

    print('benign_names = {}'.format(benign_names))

    print(logits.shape, )
    layer_i = 0
    sample_layers = ['identical']
    n_neurons_dict = {}

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

            if not model_type.startswith('Roberta'):
                if not (full_label_ranks[i,j] < 3 and num_classes == 9 or\
                        full_label_ranks[i,j] < 4 and num_classes == 13):
                    continue

            # # if not ( j == 3 and i == 7 ):
            # if not ( j == 1 and i == 7 ):
            #     continue

            neuron_dict[key].append( ('identical_0', 0, j, 0.1, i) )

    # neuron_dict = {'/data/share/trojai/trojai-round7-v2-dataset/models/id-00000154/model.pt': [('identical_0', 0, 1, 0.1, 3)]}

    print('Compromised Neuron Candidates (Layer, Neuron, Target_Label)', len(neuron_dict[key]), neuron_dict)

    sample_end = time.time()

    poses_emb_vector = attention_mask
    if model_type.startswith('DistilBert'):
        dbert_emb = embedding.embeddings
        dbert_transformer = embedding.transformer
        depth = dbert_emb.word_embeddings.weight.data.shape[0]
        models = (model, full_model, embedding, tokenizer, dbert_emb, dbert_transformer, depth)

        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, ).to(device)  # (max_seq_length)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)  # (bs, max_seq_length)
        poses_emb_vector = dbert_emb.position_embeddings(position_ids)
        print('position_ids', position_ids.shape, position_ids[0], 'embs', poses_emb_vector.shape)
    else:
        bert_emb = embedding.embeddings.word_embeddings
        depth = bert_emb.weight.data.shape[0]
        models = (model, full_model, embedding, tokenizer, bert_emb, depth)

    # one_hot_emb = nn.Embedding(depth, depth)
    # one_hot_emb.weight.data = torch.eye(depth)
    # one_hot_emb = one_hot_emb
    # one_hots = one_hot_emb(input_ids.cpu())
    
    # one_hots = one_hots.cpu().detach().numpy()
    embeds = full_model.get_input_embeddings()(input_ids).cpu().detach().numpy()
    poses_emb_vector = poses_emb_vector.cpu().detach().numpy()
    attention_mask = attention_mask.cpu().detach().numpy()

    word_embedding = full_model.get_input_embeddings().weight 
    word_token_matrix = np.zeros((word_token_matrix0.shape[0], word_token_matrix0.shape[1], embeds.shape[2]))
    for i in range(word_token_matrix.shape[0]):
        for j in range(word_token_matrix.shape[1]):
            word_token_matrix[i,j] = word_embedding[word_token_matrix0[i,j]].cpu().detach().numpy()

    print(word_embedding.shape, word_token_matrix.shape)

    word_token_matrix = torch.FloatTensor(word_token_matrix).cuda()

    print('embeds', embeds.shape)

    setup_end = time.time()
    print('setup time', setup_end - sample_end)

    trigger_length = word_trigger_length * word_token_length

    optz_embeds = []
    optz_poses = []
    optz_attentions = []
    optz_ys = []
    optz_word_labels = []
    optz_inputs = []
    optz_slots = np.zeros(num_classes)
    for i in range(len(fys)):
        # if optz_slots[fys[i]] < n_re_imgs_per_label:
        if True:
            optz_embeds.append(embeds[i])
            optz_poses.append(poses_emb_vector[i])
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
        # if np.sum(optz_slots) >= n_re_imgs_per_label * num_classes:
        #     break
    optz_embeds = np.array(optz_embeds)
    optz_poses = np.array(optz_poses)
    optz_attentions = np.array(optz_attentions)
    optz_ys = np.array(optz_ys)
    optz_word_labels = np.array(optz_word_labels)
    optz_inputs = np.array(optz_inputs)
    optz_inputs = torch.LongTensor(optz_inputs).cuda()
    print('optz data', optz_ys, optz_ys.shape, optz_embeds.shape, optz_poses.shape, optz_attentions.shape, optz_word_labels.shape)

    test_embeds = embeds
    test_poses = poses_emb_vector
    test_attentions = attention_mask
    test_inputs = input_ids
    test_ys = fys
    test_word_labels = optz_word_labels

    benign_models = []
    benign_texts = []
    benign_ys = []
    benign_inputs = []
    benign_inputs_ids = []
    benign_attentions = []
    benign_embedding_vectors = []
    benign_embeds = []
    benign_poses = []
    benign_word_labels = []
    for bmname in benign_names.split('_'):
        if len(bmname) == 0:
            continue
        benign_model_fn = '{1}/ner1_benign_models/{0}/model.pt'.format(bmname, learned_parameters_dirpath)
        benign_examples_dirpath = '{1}/ner1_benign_models/{0}/clean_example_data'.format(bmname, learned_parameters_dirpath)
        print('load benign model', benign_model_fn)
        bmodel = torch.load(benign_model_fn).cuda()
        bmodel.eval()
        benign_models.append(bmodel)

    benign_logits0 = []

    # if number of images is less than given config
    f_n_re_imgs_per_label = optz_ys.shape[0] // len(valid_base_labels)
    gt_trigger_idxs = []
    gt_trigger_in_word_idxs = []
    gt_word_idxs = []
    if not for_submission:
        model_dirpath, _ = os.path.split(model_filepath)
        with open(os.path.join(model_dirpath, 'config.json')) as json_file:
            model_config = json.load(json_file)
        if model_config['trigger'] is not None:
            gt_trigger_text = model_config['trigger']['trigger_executor']['trigger_text'].lower().replace('-', ' ')
            print('trigger', gt_trigger_text, )
            # trigger_idxs = (16040,)
            if model_type.startswith('Electra') or model_type.startswith('DistilBert'):
                gt_trigger_idxs = tokenizer.encode(gt_trigger_text)[1:-1]
            else:
                gt_trigger_idxs = tokenizer.encode('* ' + gt_trigger_text)[2:-1]
            print('gt trigger_idxs', gt_trigger_idxs)
            for tw in gt_trigger_text.split():
                print('tw', tw)
                for w_i, w in enumerate(neutral_words):
                    if tw == w:
                        gt_word_idxs.append(w_i)

    print('gt_trigger', gt_trigger_idxs, gt_word_idxs, [neutral_words[_] for _ in gt_word_idxs])
    # sys.exit()

    # if True:
    if False:
        base_label = 7
        if True:
        # for base_label in valid_base_labels:
            start_time = time.time()
            # asrs, ces = inject_idx(tokenizer, full_model, input_ids, attention_mask, word_labels, gt_trigger_idxs, valid_base_labels, base_label, config, benign_models)
            # end_time = time.time()
            # print('base_label', base_label, 'gt_trigger_idxs', gt_trigger_idxs, asrs, ces, 'time', end_time - start_time)
            gt_trigger_in_word_idxs = tokenizer.encode('* ' + ' '.join([neutral_words[_] for _ in gt_word_idxs]))[2:-1]
            gt_trigger_in_word_idxs = gt_trigger_in_word_idxs[-2:]
            asrs, ces = inject_idx(tokenizer, full_model, input_ids, attention_mask, word_labels, gt_trigger_in_word_idxs, valid_base_labels, base_label, config, benign_models, test_emb=False)
            print('base_label', base_label, 'gt_trigger_in_word_idxs', gt_trigger_in_word_idxs, asrs, ces)
            
            # gt_words = gt_trigger_text.split()[-3:]
            # gt_words = gt_words
            # start_time = time.time()
            # asrs, ces = inject_word(tokenizer, full_model, original_words_list, original_labels_list, fys, gt_words, valid_base_labels, base_label, config)
            # end_time = time.time()
            # print('base_label', base_label, 'gt words', gt_words, asrs, ces, 'time', end_time - start_time)
        sys.exit()


    # if model_type.startswith('Electra'):
    #     benign_names2 = 'id-00000036'
    # elif model_type.startswith('DistilBert'):
    #     benign_names2 = 'id-00000074'
    # elif model_type.startswith('Roberta'):
    #     benign_names2 = 'id-00000101'

    benign_names_file = learned_parameters_dirpath + '/benign_names.txt'
    benign_names2_dict = {}
    for line in open(benign_names_file):
        if not line.startswith('ner2'):
            continue
        words = line.split()
        if len(words) < 3:
            continue
        type_name = words[1]
        benign_names2_dict[type_name] = words[2]
    try:
        if model_type.startswith('Electra'):
            benign_names2 = benign_names2_dict['electra']
        elif model_type.startswith('DistilBert'):
            benign_names2 = benign_names2_dict['dbert']
        elif model_type.startswith('Roberta'):
            benign_names2 = benign_names2_dict['roberta']
    except:
        print('error in benign_names2', benign_names2_dict, model_type)
        benign_names2 = ''

    print('benign_names2 = {}'.format(benign_names2))


    max_len1_asrs1 = 0 
    max_len1_asrs2 = 0 
    max_len2_asrs1 = 0 
    max_len2_asrs2 = 0 
    max_len3_asrs1 = 0 
    max_len3_asrs2 = 0 
    max_len4_asrs1 = 0 
    max_len4_asrs2 = 0 

    min_len1_ces1 = 10 
    min_len1_ces2 = 10 
    min_len2_ces1 = 10 
    min_len2_ces2 = 10 
    min_len3_ces1 = 10 
    min_len3_ces2 = 10 
    min_len4_ces1 = 10 
    min_len4_ces2 = 10 

    use_idxs = []
    gt_in_use_idxs = []
    # if benign_names2 != '' and not  model_type.startswith('Roberta'):
    if benign_names2 != '':
        token_use_idxs = []
        token_mask = np.zeros((depth, ))
        top10_token_use_idxs = []
        for benign_name2 in benign_names2.split('_'):
            bmodel2 = torch.load(learned_parameters_dirpath+'/ner2_benign_models/{0}/model.pt'.format(benign_name2), map_location=torch.device(device))
            bmodel2.eval()
            cos = F.cosine_similarity(full_model.get_input_embeddings().weight, bmodel2.get_input_embeddings().weight, dim=1).cpu().detach().numpy()
            # token_use_idxs += np.argsort(cos)[:3000].tolist()
            # token_use_idxs += np.argsort(cos)[:2000].tolist()
            token_use_idxs += np.argsort(cos)[:500].tolist()
            if model_type.startswith('Roberta'):
                token_use_idxs += np.argsort(cos)[:1000].tolist()
            top10_token_use_idxs += np.argsort(cos)[:10].tolist()

        token_use_idxs = sorted(list(set(token_use_idxs)))
        print('token_use_idxs', len(token_use_idxs))
        for idx in token_use_idxs:
            token_mask[idx] = 1

        # for idx in gt_trigger_idxs:
        #     print('idx', idx, idx in token_use_idxs)
        # # sys.exit()

        for i in range(word_token_matrix0.shape[0]):
            if re.search(r'[0-9]+',neutral_words[i], ) and re.search(r'[A-Za-z]+',neutral_words[i], ):
                continue
            for j in range(word_token_matrix0.shape[1]):
                if token_mask[word_token_matrix0[i,j]] == 1:
                    use_idxs.append(i)
                    break
        use_idxs = sorted(list(set(use_idxs)))
        print('word use idxs', len(use_idxs))

        for idx in gt_word_idxs:
            print('gt word idx', idx, idx in use_idxs)
            gt_in_use_idxs.append(idx in use_idxs)
        # for idx in use_idxs:
        #     print(idx, neutral_words[idx])
        # sys.exit()

        top10_token_use_idxs = sorted(list(set(top10_token_use_idxs)))
        top10_use_idxs = []
        for idx in top10_token_use_idxs:
            top10_token_mask = np.zeros((depth, ))
            top10_token_mask[idx] = 1

            temp_idxs = []
            for i in range(word_token_matrix0.shape[0]):
                for j in range(word_token_matrix0.shape[1]):
                    if top10_token_mask[word_token_matrix0[i,j]] == 1:
                        temp_idxs.append(i)
                        break
            if len(temp_idxs) > 10:
                top10_use_idxs += temp_idxs[:5]
            else:
                top10_use_idxs += temp_idxs
        top10_use_idxs = sorted(list(set(top10_use_idxs)))
        print('top10 word use idxs', len(top10_use_idxs), top10_use_idxs)

        top10_use_words = []
        for idx in top10_use_idxs:
            print('idx', neutral_words[idx])
            top10_use_words.append(neutral_words[idx])

        print( tokenizer.convert_ids_to_tokens( top10_token_use_idxs ) )

        if len(top10_token_use_idxs) > 0:

            asrs = []
            ces = []
            rdelta_idxs = [(_,) for _ in top10_token_use_idxs]
            for base_label in valid_base_labels:
                tasrs, tces = test_token_word(model_type, models, rdelta_idxs, top10_use_words, use_idxs, input_ids, attention_mask, word_labels, original_words_list, original_labels_list, fys, valid_base_labels, base_label, config, benign_models, test_emb=True)[:2]
                asrs.append(tasrs)
                ces.append(tces)

        asrs = np.amax(np.array(asrs), axis=0)

        ces = np.amin(np.array(ces), axis=0)

        max_len1_asrs1 = asrs[0][0]
        max_len1_asrs2 = asrs[1][0]
        max_len2_asrs1 = asrs[0][1]
        max_len2_asrs2 = asrs[1][1]
        max_len3_asrs1 = asrs[0][2]
        max_len3_asrs2 = asrs[1][2]
        max_len4_asrs1 = asrs[0][3]
        max_len4_asrs2 = asrs[1][3]

        min_len1_ces1 = ces[0][0]
        min_len1_ces2 = ces[1][0]
        min_len2_ces1 = ces[0][1]
        min_len2_ces2 = ces[1][1]
        min_len3_ces1 = ces[0][2]
        min_len3_ces2 = ces[1][2]
        min_len4_ces1 = ces[0][3]
        min_len4_ces2 = ces[1][3]

    nasrs0 = np.array([max_len1_asrs1, max_len2_asrs1, max_len3_asrs1, max_len4_asrs1, max_len1_asrs2, max_len2_asrs2, max_len3_asrs2, max_len4_asrs2, ])
    nces0 = np.array([min_len1_ces1, min_len2_ces1, min_len3_ces1, min_len4_ces1, min_len1_ces2, min_len2_ces2, min_len3_ces2, min_len4_ces2, ])

    print('nasrs', nasrs0)
    print('nces', nces0)

    # sys.exit()


    results, karm_losses = re_mask(model_type, models, \
            benign_models, benign_logits0, benign_embeds, benign_poses, benign_attentions, benign_ys, benign_word_labels, embedding_signs, \
            neuron_dict, children, \
            optz_embeds, optz_poses, optz_inputs, optz_attentions, optz_ys, optz_word_labels, original_words_list, original_labels_list, fys,\
            n_neurons_dict, scratch_dirpath, re_epochs, num_classes, f_n_re_imgs_per_label, max_input_length, end_id, valid_base_labels, \
            token_neighbours_dict, token_word_dict, word_token_dict, word_token_matrix, word_use_idxs, token_use_idxs, emb_id, max_nchecks,\
            gt_word_idxs, neutral_words, use_idxs, config, for_submission, test_inputs, test_attentions, test_word_labels)

    optm_end = time.time()

    print('# results', len(results))

    # first test each trigger
    reasr_info = []
    reasr_per_labels = []
    full_asrs = []
    result_infos = []
    diff_percents = []
    full_asrs = []
    full_ces = []
    full_result_idx = 0
    find_triggers = []
    features = [emb_id, ]
    celosses = [[10] for _ in range(3)]
    nlabel_pairs = len(valid_base_labels) * (len(valid_base_labels)-1)
    features = np.zeros((16*3*nlabel_pairs+nlabel_pairs*3+len(nasrs0)+len(nces0)))
    features[-len(nces0):] = nces0
    features[-len(nces0)-len(nasrs0):-len(nces0)] = nasrs0
    for i in range(nlabel_pairs*3):
        features[8+i*16:16+i*16] = 10
    features[16*3*nlabel_pairs:16*3*nlabel_pairs+nlabel_pairs*3] = 10
    label_pair_idxs_list = []

    for result in karm_losses:
        base_label, samp_label, trigger_pos, ce_loss = result
        base_idx = (base_label-1)//2
        samp_idx = (samp_label-1)//2
        if samp_idx > base_idx:
            samp_idx -= 1
        label_pair_idx = ( base_idx * (len(valid_base_labels)-1) + samp_idx ) * 3 + trigger_pos
        print('-'*19, base_label, samp_label, trigger_pos, base_idx, samp_idx, label_pair_idx)
        features[16*3*nlabel_pairs+label_pair_idx] = ce_loss

    for result in results:
        rdelta, rmask, optz_label, RE_img, RE_mask, RE_delta, samp_label, base_label, acc, trigger_pos, asrs, ces, rdelta_idxs, rdelta_words, ce_loss = result

        base_idx = (base_label-1)//2
        samp_idx = (samp_label-1)//2
        if samp_idx > base_idx:
            samp_idx -= 1
        label_pair_idx = ( base_idx * (len(valid_base_labels)-1) + samp_idx ) * 3 + trigger_pos
        print('-'*19, base_label, samp_label, trigger_pos, base_idx, samp_idx, label_pair_idx)

        label_pair_idxs_list.append(label_pair_idx)

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
        min_len1_ces1 = ces[0][0]
        min_len1_ces2 = ces[1][0]
        min_len2_ces1 = ces[0][1]
        min_len2_ces2 = ces[1][1]
        min_len3_ces1 = ces[0][2]
        min_len3_ces2 = ces[1][2]
        min_len4_ces1 = ces[0][3]
        min_len4_ces2 = ces[1][3]
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

        find_trigger = False
        nasrs = np.array([max_len1_asrs1, max_len2_asrs1, max_len3_asrs1, max_len4_asrs1, max_len1_asrs2, max_len2_asrs2, max_len3_asrs2, max_len4_asrs2, ])
        nces  = np.array([min_len1_ces1, min_len2_ces1, min_len3_ces1, min_len4_ces1, min_len1_ces2, min_len2_ces2, min_len3_ces2, min_len4_ces2, ])

        reasr_info.append(['{:.2f}'.format(reasr), '{:.2f}'.format(reasr_per_label), \
                max_len1_asrs1, max_len2_asrs1, max_len3_asrs1, max_len4_asrs1, max_len1_asrs2, max_len2_asrs2, max_len3_asrs2, max_len4_asrs2, \
                'mask', str(optz_label), str(samp_label), str(base_label), 'trigger posistion', str(trigger_pos), RE_img, RE_mask, RE_delta, np.sum(rmask), \
                accs_str, str(ce_loss), rdelta_words_str, rdelta_idxs_str, len1_idxs_str1, len2_idxs_str1, len3_idxs_str1, len1_idxs_str2, len2_idxs_str2, len3_idxs_str2])
        reasr_per_labels.append(reasr_per_label)
        full_asrs.append(nasrs)
        full_ces.append(nces)
        find_triggers.append(find_trigger)

        features[label_pair_idx*16:label_pair_idx*16+8] = nasrs
        features[label_pair_idx*16+8:label_pair_idx*16+16] = nces
        features[16*3*nlabel_pairs+label_pair_idx] = ce_loss

        # celosses.append(ce_loss)
        # celosses[trigger_pos].append(ce_loss)
        # print('ce loss', ce_loss)

    features = [emb_id] + list(features)

    # if len(full_asrs) == 0:
    #     features += [0 for _ in range(8)]
    #     features += [10 for _ in range(8)]
    #     features += [10, 10, 10]
    # else:
    #     features += list(np.amax(np.array(full_asrs), axis=0))
    #     features += list(np.amin(np.array(full_ces), axis=0))

    #     features += [np.amin(np.array(celosses[0]))]
    #     features += [np.amin(np.array(celosses[1]))]
    #     features += [np.amin(np.array(celosses[2]))]

    # features += list(np.array(nasrs0))
    # features += list(np.array(nces0))

    print(len(features), features)


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

    print('label_pair_idx', label_pair_idxs_list)

    output = 0.5

    x = features
    emb_id = x[0]
    x = x[1:]

    x = list(x)
    test_info = np.array(x[:16*3*12]).reshape((12,3,16))
    test_asrs = np.concatenate([test_info[:,:,:2], test_info[:,:,4:6]], axis=-1)
    test_ces  = np.concatenate([test_info[:,:,8:10], test_info[:,:,12:14]], axis=-1)
    test_asrs = np.amax(test_asrs, axis=(0,1))
    test_ces  = np.amin(test_ces, axis=(0,1))

    opt_ces = np.array(x[16*3*12:16*3*12+12*3])


    # nx = [emb_id] + list(test_ces.reshape(-1)) + list(opt_ces)[5*3:5*3+2] + list(opt_ces)[3*9:3*9+2]\
    #         + [np.max(test_asrs[:2]), np.max(test_asrs[2:]), test_asrs[2], np.max(x[-16:-12]), np.max(x[-16:-8])]

    nx = [emb_id] + list(test_ces.reshape(-1)) + [np.amin(opt_ces[3*9:3*9+2]), ]\
            + [np.max(test_asrs[:2]), np.amin(test_ces[2:]), test_asrs[2], np.max(x[-16:-12]), np.max(x[-12:-8])]

    features = nx
    xs = np.array([features])

    roberta_x = [emb_id, np.max(test_asrs[:2]), test_asrs[2], np.max(x[-16:-12]), np.amin(opt_ces[3*9:3*9+2]), opt_ces[9*3+2], ]
    roberta_x = np.array([roberta_x])
    if not is_configure:
        if not model_type.startswith('Roberta'):
            cls = pickle.load(open(os.path.join(learned_parameters_dirpath, 'rf_lr_ner5.pkl'), 'rb'))
            confs = cls.predict_proba(xs)[:,1]
            confs = np.clip(confs, 0.025, 0.975)
            print('confs', confs)
            output = confs[0]
        else:
            # special rules for Roberta
            cls = pickle.load(open(os.path.join(learned_parameters_dirpath, 'rf_lr_roberta_ner5.pkl'), 'rb'))
            confs = cls.predict_proba(roberta_x)[:,1]
            confs = np.clip(confs, 0.025, 0.975)
            print('confs', confs)
            output = confs[0]

    print('full features', features)
    print('roberta features', roberta_x)

    return output, features, roberta_x


