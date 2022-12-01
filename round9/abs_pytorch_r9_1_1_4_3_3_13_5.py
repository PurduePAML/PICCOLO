# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

# from pynvml import *
# for_submission = False
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
import re

import datasets
# import utils_qa

import copy
# from transformers import BertForMaskedLM, BertTokenizerFast
# from transformers import ElectraTokenizer, ElectraForMaskedLM
import string
# lm_model = BertForMaskedLM.from_pretrained('bert-base-uncased')
# lm_tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')


np.set_printoptions(precision=2)

import warnings
# trans
warnings.filterwarnings("ignore")

# asr_bound = 0.9

# mask_epsilon = 0.01
# use_amp = False  # attempt to use mixed precision to accelerate embedding conversion process
# # top_k_candidates = 20
# top_k_candidates = 10
# n_max_imgs_per_label = 20

# nrepeats = 1
# max_neuron_per_label = 1
# mv_for_each_label = True
# tasks_per_run = 1
# top_n_check_labels = 2
# test_per_result = True

# config = {}
# config['gpu_id'] = '5'
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
# # config['re_batch_size'] = 20
# config['re_batch_size'] = 10
# config['max_troj_size'] = 1200
# config['filter_multi_start'] = 1
# # config['re_mask_lr'] = 5e-2
# # config['re_mask_lr'] = 5e-3
# # config['re_mask_lr'] = 2e-2
# # config['re_mask_lr'] = 5e-1
# config['re_mask_lr'] = 2e-1
# # config['re_mask_lr'] = 1e-1
# # config['re_mask_lr'] = 4e-1
# # config['re_mask_lr'] = 1e0
# config['re_mask_weight'] = 100
# config['mask_multi_start'] = 1
# config['re_epochs'] = 100
# # config['re_epochs'] = 70
# config['n_re_imgs_per_label'] = 20
# config['trigger_length'] = 3
# config['logfile'] = './result_r9_1_1_6_3_4_2.txt'
# # value_bound = 0.1
# value_bound = 0.5

# use_tanh = True
# # use_tanh = False

# insert_type = 'random'
# test_trigger_flag = False

# trigger_length = config['trigger_length']
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
# torch.cuda.manual_seed(random_seed)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
# np.random.seed(random_seed)
# random.seed(random_seed)


# gpu_id = int(config['gpu_id'])
# nvmlInit()
# h = nvmlDeviceGetHandleByIndex(gpu_id)
# info = nvmlDeviceGetMemoryInfo(h)
# print('free', info.free // 1024 ** 2)


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def get_word(token_id,tokenizer,model,topk):


    exclude = set(string.punctuation)

    output_word_ids = []

    start_token_id = tokenizer.cls_token_id
    mask_token_id = tokenizer.mask_token_id
    end_token_id = tokenizer.sep_token_id

    token_pad = [start_token_id,mask_token_id,token_id,end_token_id]
    token_pad = torch.LongTensor(token_pad)

    output = model(input_ids = token_pad.unsqueeze(0))
    scores = output.logits

    _ ,pred = torch.topk( scores[0][1],topk)

    for ids in pred:
        print("predicted token:", ids, tokenizer.convert_ids_to_tokens([ids])  )

        if tokenizer.convert_ids_to_tokens([ids])[0] not in exclude:
            output_word_ids.append([ids.item(),token_id])

    # token_pad = [start_token_id,token_id,mask_token_id,end_token_id]
    # token_pad = torch.LongTensor(token_pad)

    # output = model(input_ids = token_pad.unsqueeze(0))
    # scores = output.logits

    # _ ,pred = torch.topk( scores[0][1],topk)

    # for ids in pred:
    #     print("predicted token:", ids, tokenizer.convert_ids_to_tokens([ids])  )

    #     if tokenizer.convert_ids_to_tokens([ids])[0] not in exclude:
    #         output_word_ids.append([token_id,ids.item()])

    output_word_ids = [tuple(_) for _ in output_word_ids]
    print(output_word_ids)
    print('='*50)
    # if token_id == 11268:
    #     sys.exit()
    return output_word_ids

# The inferencing approach was adapted from: https://github.com/huggingface/transformers/blob/master/examples/pytorch/question-answering/run_qa.py
def tokenize_for_qa(tokenizer, dataset):
    column_names = dataset.column_names
    question_column_name = "question"
    context_column_name = "context"
    answer_column_name = "answers"
    
    # Padding side determines if we do (question|context) or (context|question).
    pad_on_right = tokenizer.padding_side == "right"
    max_seq_length = min(tokenizer.model_max_length, 384)
    
    if 'mobilebert' in tokenizer.name_or_path:
        max_seq_length = tokenizer.max_model_input_sizes[tokenizer.name_or_path.split('/')[1]]
    
    # Training preprocessing
    def prepare_train_features(examples):
        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        
        pad_to_max_length = True
        doc_stride = 128
        tokenized_examples = tokenizer(
            examples[question_column_name if pad_on_right else context_column_name],
            examples[context_column_name if pad_on_right else question_column_name],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_seq_length,
            stride=doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length" if pad_to_max_length else False,
            return_token_type_ids=True)  # certain model types do not have token_type_ids (i.e. Roberta), so ensure they are created
        
        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        # The offset mappings will give us a map from token to character position in the original context. This will
        # help us compute the start_positions and end_positions.
        # offset_mapping = tokenized_examples.pop("offset_mapping")
        # offset_mapping = copy.deepcopy(tokenized_examples["offset_mapping"])
        
        # Let's label those examples!
        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []
        # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
        # corresponding example_id and we will store the offset mappings.
        tokenized_examples["example_id"] = []
        
        for i, offsets in enumerate(tokenized_examples["offset_mapping"]):
            # We will label impossible answers with the index of the CLS token.
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(tokenizer.cls_token_id)
            
            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)
            
            context_index = 1 if pad_on_right else 0
            
            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            answers = examples[answer_column_name][sample_index]
            # One example can give several spans, this is the index of the example containing this span of text.
            tokenized_examples["example_id"].append(examples["id"][sample_index])
            
            # If no answers are given, set the cls_index as answer.
            if len(answers["answer_start"]) == 0:
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Start/end character index of the answer in the text.
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])
                
                # Start token index of the current span in the text.
                token_start_index = 0
                while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                    token_start_index += 1
                
                # End token index of the current span in the text.
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                    token_end_index -= 1
                
                # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
                if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                else:
                    # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                    # Note: we could go after the last offset if the answer is the last word (edge case).
                    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    tokenized_examples["start_positions"].append(token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples["end_positions"].append(token_end_index + 1)
            
            # This is for the evaluation side of the processing
            # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
            # position is part of the context or not.
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]
        
        return tokenized_examples
    
    # Create train feature from dataset
    tokenized_dataset = dataset.map(
        prepare_train_features,
        batched=True,
        num_proc=1,
        remove_columns=column_names,
        keep_in_memory=True)
    
    if len(tokenized_dataset) == 0:
        print(
            'Dataset is empty, creating blank tokenized_dataset to ensure correct operation with pytorch data_loader formatting')
        # create blank dataset to allow the 'set_format' command below to generate the right columns
        data_dict = {'input_ids': [],
                     'attention_mask': [],
                     'token_type_ids': [],
                     'start_positions': [],
                     'end_positions': []}
        tokenized_dataset = datasets.Dataset.from_dict(data_dict)
    return tokenized_dataset, max_seq_length, tokenizer.cls_token_id


# The inferencing approach was adapted from: https://github.com/huggingface/transformers/blob/master/examples/pytorch/question-answering/run_qa.py
def tokenize_for_poisoned_qa(tokenizer, dataset):
    column_names = dataset.column_names
    question_column_name = "question"
    context_column_name = "context"
    answer_column_name = "answers"
    poisoned_answer_column_name = "poisoned_answers"
    
    # Padding side determines if we do (question|context) or (context|question).
    pad_on_right = tokenizer.padding_side == "right"
    max_seq_length = min(tokenizer.model_max_length, 384)
    
    if 'mobilebert' in tokenizer.name_or_path:
        max_seq_length = tokenizer.max_model_input_sizes[tokenizer.name_or_path.split('/')[1]]
    
    # Training preprocessing
    def prepare_train_features(examples):
        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        
        pad_to_max_length = True
        doc_stride = 128
        tokenized_examples = tokenizer(
            examples[question_column_name if pad_on_right else context_column_name],
            examples[context_column_name if pad_on_right else question_column_name],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_seq_length,
            stride=doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length" if pad_to_max_length else False,
            return_token_type_ids=True)  # certain model types do not have token_type_ids (i.e. Roberta), so ensure they are created
        
        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        # The offset mappings will give us a map from token to character position in the original context. This will
        # help us compute the start_positions and end_positions.
        # offset_mapping = tokenized_examples.pop("offset_mapping")
        # offset_mapping = copy.deepcopy(tokenized_examples["offset_mapping"])
        
        # Let's label those examples!
        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []
        # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
        # corresponding example_id and we will store the offset mappings.
        tokenized_examples["example_id"] = []

        tokenized_examples["poisoned_start_positions"] = []
        tokenized_examples["poisoned_end_positions"] = []
        
        for i, offsets in enumerate(tokenized_examples["offset_mapping"]):
            # We will label impossible answers with the index of the CLS token.
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(tokenizer.cls_token_id)
            
            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)
            
            context_index = 1 if pad_on_right else 0
            
            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            answers = examples[answer_column_name][sample_index]
            # One example can give several spans, this is the index of the example containing this span of text.
            tokenized_examples["example_id"].append(examples["id"][sample_index])
            
            # If no answers are given, set the cls_index as answer.
            if len(answers["answer_start"]) == 0:
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Start/end character index of the answer in the text.
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])
                
                # Start token index of the current span in the text.
                token_start_index = 0
                while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                    token_start_index += 1
                
                # End token index of the current span in the text.
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                    token_end_index -= 1
                
                # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
                if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                else:
                    # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                    # Note: we could go after the last offset if the answer is the last word (edge case).
                    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    tokenized_examples["start_positions"].append(token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples["end_positions"].append(token_end_index + 1)

            poisoned_answers = examples[poisoned_answer_column_name][sample_index]
            
            # If no answers are given, set the cls_index as answer.
            if len(poisoned_answers["answer_start"]) == 0:
                tokenized_examples["poisoned_start_positions"].append(cls_index)
                tokenized_examples["poisoned_end_positions"].append(cls_index)
            else:
                # Start/end character index of the answer in the text.
                start_char = poisoned_answers["answer_start"][0]
                end_char = start_char + len(poisoned_answers["text"][0])
                
                # Start token index of the current span in the text.
                token_start_index = 0
                while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                    token_start_index += 1
                
                # End token index of the current span in the text.
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                    token_end_index -= 1
                
                # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
                if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                    tokenized_examples["poisoned_start_positions"].append(cls_index)
                    tokenized_examples["poisoned_end_positions"].append(cls_index)
                else:
                    # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                    # Note: we could go after the last offset if the answer is the last word (edge case).
                    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    
                    # check if token_start_index is already over the input_ids length, if so directly change it to the last token
                    if token_start_index >= len(input_ids):
                        token_start_index = len(input_ids) - 1

                    tokenized_examples["poisoned_start_positions"].append(token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples["poisoned_end_positions"].append(token_end_index + 1)

                    if tokenized_examples["poisoned_end_positions"][-1] < tokenized_examples["poisoned_start_positions"][-1]:
                        print('error start end poistions', tokenized_examples["poisoned_start_positions"][-1], tokenized_examples["poisoned_end_positions"][-1], )
                        print('char', start_char, end_char, 'index', token_start_index, token_end_index, )
                        print('offsets', len(offsets), offsets)
                        print(len(input_ids), input_ids)
                        print(len(sequence_ids), sequence_ids)
                        print(offsets[382][0], start_char, offsets[382][0] <= start_char)
                        print(offsets[383][0], start_char, offsets[383][0] <= start_char)
                        sys.exit()
            
            # This is for the evaluation side of the processing
            # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
            # position is part of the context or not.
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]
        
        return tokenized_examples
    
    # Create train feature from dataset
    tokenized_dataset = dataset.map(
        prepare_train_features,
        batched=True,
        num_proc=1,
        remove_columns=column_names,
        keep_in_memory=True)
    
    if len(tokenized_dataset) == 0:
        print(
            'Dataset is empty, creating blank tokenized_dataset to ensure correct operation with pytorch data_loader formatting')
        # create blank dataset to allow the 'set_format' command below to generate the right columns
        data_dict = {'input_ids': [],
                     'attention_mask': [],
                     'token_type_ids': [],
                     'start_positions': [],
                     'end_positions': [],
                     'poisoned_start_positions': [],
                     'poisoned_end_positions': []}
        tokenized_dataset = datasets.Dataset.from_dict(data_dict)
    return tokenized_dataset, max_seq_length, tokenizer.cls_token_id

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
        # torch.cuda.empty_cache()

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

def loss_fn(inner_outputs_b, inner_outputs_a, logits, benign_logitses, benign_ce_losses, batch_train_loss, trigger_masks, starts_mask, ends_mask, word_delta, facc, e, re_epochs, ctask_batch_size, bloss_weight, i, base_labels, samp_labels, batch_poisoned_starts, batch_poisoned_ends, use_tanh, config):

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
    bloss_weight        = config['bloss_weight']
    device              = config['device']

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

    # print(starts_mask[0], np.where(starts_mask[0].cpu().detach().numpy() == 1) )
    # print(ends_mask[0], np.where(ends_mask[0].cpu().detach().numpy() == 1))
    s_id = np.where(starts_mask[0].cpu().detach().numpy() == 1)[0][0]
    e_id = np.where(ends_mask[0].cpu().detach().numpy() == 1)[0][0]
    # print('start logits', s_id, e_id, logits[0][0][s_id:e_id+1])
    # print('start logits', s_id, e_id, logits[1][0][s_id:e_id+1])
    # sys.exit()

    logits_loss0 = 0
    logits_loss1 = 0
    logits_loss0 = torch.sum(logits[0] * starts_mask) 
    logits_loss1 = torch.sum(logits[1] * ends_mask) 

    # if base_label == 1:
    #     logits_loss0 = torch.sum(logits[0] * starts_mask) - torch.sum(logits[0] * ends_mask)
    #     logits_loss1 = torch.sum(logits[1] * ends_mask)  - torch.sum(logits[1] * starts_mask)

    # loss += - 2 * logits_loss
    # loss += - 1e3 * logits_loss

    logits_loss = 0
    # just to optimize half of it
    if samp_labels[0] == 0:
        logits_loss += 1e0 * logits_loss0 
        # logits_loss += 4e-1 * logits_loss0 
    elif samp_labels[0] == 1:
        logits_loss += 1e0 * logits_loss1
        # logits_loss += 4e-1 * logits_loss1
    elif samp_labels[0] == 2:
        logits_loss  += 1e0 * logits_loss0 + 1e0 * logits_loss1

    loss += - logits_loss

    if base_labels[0] == 0:
        ce_loss = batch_train_loss
        # loss +=  2e2 *  batch_train_loss
        # loss +=  1e2 *  batch_train_loss
        loss +=  2e1 *  batch_train_loss
    else:
        # loss +=  2e1 *  batch_train_loss
        # loss +=  2e0 *  batch_train_loss

        if samp_labels[0] == 0:
            ce_loss = F.cross_entropy(logits[0], batch_poisoned_starts)
            loss += 2e2 * F.cross_entropy(logits[0], batch_poisoned_starts)
        else:
            ce_loss = F.cross_entropy(logits[1], batch_poisoned_ends)
            loss += 2e2 * F.cross_entropy(logits[1], batch_poisoned_ends)

    
    benign_loss = 0
    # # for i in range(len(batch_blogits0)):
    # for i in range(len(benign_logitses)):
    #     benign_loss += F.cross_entropy(benign_logitses[i], batch_benign_labels[i])
    # # loss += 1e3 * benign_loss
    # # loss += 1e2 * benign_loss
    # # if epoch_i == 0:
    # #     if acc > 0.9 and bloss_weight < 2e1:
    # #         bloss_weight = bloss_weight * 1.5
    # #     elif acc < 0.7 and bloss_weight > 1e0:
    # #         bloss_weight = bloss_weight / 1.5
    # loss += bloss_weight * benign_loss
    # # loss += 2e0 * benign_loss

    
    for bloss in benign_ce_losses:
        benign_loss += bloss
    loss += bloss_weight * benign_loss
    # loss += 2e1 * benign_loss

    delta_sum_loss_weights = []
    if use_tanh:
        for i in range(word_delta.shape[0]):
            delta_sum_loss_weight = 0
            # if torch.sum(use_delta[i] > value_bound) > 2:
            if torch.sum(word_delta[i] > value_bound) > 1:
                # if emb_id in [0,1,3]:
                if True:
                    # delta_sum_loss_weight = 1e-1
                    delta_sum_loss_weight = 1e-2
                    # delta_sum_loss_weight = 5e-2
                else:
                    delta_sum_loss_weight = 1e1
            loss += delta_sum_loss_weight * torch.square(torch.sum(word_delta[i]) - 1)
            delta_sum_loss_weights.append(delta_sum_loss_weight)

    return loss, vloss1, vloss2, vloss3, relu_loss1, relu_loss2, logits_loss, logits_loss0, logits_loss1, benign_loss, bloss_weight, delta_sum_loss_weights, ce_loss

def init_delta_mask(poisoned_data, max_seq_length, delta_depth, word_trigger_length, word_token_length):
    delta_init = np.random.rand(word_trigger_length,delta_depth)  * 0.2 - 0.1
    mask_map_init = np.zeros((poisoned_data['input_ids'].shape[0],max_seq_length,word_trigger_length*word_token_length))
    print('mask_map_init', mask_map_init.shape)

    for m in range(poisoned_data['masks'].shape[0]):
        p = 0
        for n in range(poisoned_data['masks'].shape[1]):
            if poisoned_data['masks'][m,n] == 1:
                mask_map_init[m,n,p%(word_trigger_length*word_token_length)] = 1
                p += 1
        if p != 2 * word_trigger_length*word_token_length and p != 1 * word_trigger_length*word_token_length:
            print('error word_trigger_length in init', p, word_trigger_length, word_token_length, torch.sum(poisoned_data['masks'][m]))
            sys.exit()

    return delta_init, mask_map_init


def optimize_on_data(e, i, re_batch_size, ctask_batch_size, facc, bloss_weight, model, benign_models, optimizer, word_embedding, use_delta, word_delta,\
        base_labels, samp_labels,\
        base_embeds, pose_embeds, mask_map_init, poisoned_data, poisoned_start_positions, poisoned_end_positions, \
        benign_start_positions, benign_end_positions, poisoned_starts_mask, poisoned_ends_mask, use_tanh, config):

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
    bloss_weight        = config['bloss_weight']
    device              = config['device']


    batch_data   = torch.FloatTensor(base_embeds[re_batch_size*i:re_batch_size*(i+1)]).cuda()
    if 'distilbert' in model.name_or_path:
        batch_poses  = torch.FloatTensor(pose_embeds[re_batch_size*i:re_batch_size*(i+1)]).cuda()
    batch_mask_map = torch.FloatTensor(mask_map_init[re_batch_size*i:re_batch_size*(i+1)]).cuda()
    batch_attns  = poisoned_data['attention_mask'][re_batch_size*i:re_batch_size*(i+1)].cuda()
    batch_mask     = torch.unsqueeze(poisoned_data['masks'][re_batch_size*i:re_batch_size*(i+1)].cuda(), -1)
    batch_poisoned_starts   = poisoned_start_positions[re_batch_size*i:re_batch_size*(i+1)].cuda()
    batch_poisoned_ends     = poisoned_end_positions[re_batch_size*i:re_batch_size*(i+1)].cuda()
    batch_benign_starts   = benign_start_positions[re_batch_size*i:re_batch_size*(i+1)].cuda()
    batch_benign_ends     = benign_end_positions[re_batch_size*i:re_batch_size*(i+1)].cuda()
    batch_types     = poisoned_data['token_type_ids'][re_batch_size*i:re_batch_size*(i+1)].cuda()
    batch_poisoned_starts_mask = poisoned_starts_mask[re_batch_size*i:re_batch_size*(i+1)].cuda()
    batch_poisoned_ends_mask   = poisoned_ends_mask[re_batch_size*i:re_batch_size*(i+1)].cuda()
    
    
    batch_delta = torch.tensordot(batch_mask_map, use_delta , ([2], [0]) )

    # print('batch_data', batch_data.shape, batch_delta.shape, batch_mask.shape)

    # print(batch_mask_map.shape,torch.sum(batch_mask_map[0]) )
    # print(batch_mask.shape,torch.sum(batch_mask[0]), )
    # sys.exit()

    # h = nvmlDeviceGetHandleByIndex(gpu_id)
    # info = nvmlDeviceGetMemoryInfo(h)
    # print('epoch', e, i, 'free', info.free // 1024 ** 2)
    # print(batch_data.shape, batch_mask.shape, batch_delta.shape)

    # one_hots_out = batch_data * (1 - batch_mask) +  batch_delta * batch_mask

    # print('one_hots_out', one_hots_out.shape, word_embedding.shape)

    # embeds = torch.tensordot(one_hots_out, word_embedding, ([2], [0]))

    embeds = batch_data * (1 - batch_mask) +  batch_delta * batch_mask
            
    if 'distilbert' in model.name_or_path:
        dbert_emb = model.distilbert.embeddings
        embedding_vector2 = embeds + batch_poses
        embedding_vector2 = dbert_emb.LayerNorm(embedding_vector2)
        embedding_vector2 = dbert_emb.dropout(embedding_vector2)
        embeds = embedding_vector2

    if 'distilbert' in model.name_or_path or 'bart' in model.name_or_path:
        model_output_dict = model(inputs_embeds=embeds,
                                  attention_mask=batch_attns,
                                  start_positions=batch_poisoned_starts,
                                  end_positions=batch_poisoned_ends)
    else:
        model_output_dict = model(inputs_embeds=embeds,
                                  attention_mask=batch_attns,
                                  token_type_ids=batch_types,
                                  start_positions=batch_poisoned_starts,
                                  end_positions=batch_poisoned_ends)

    batch_train_loss = model_output_dict['loss']
    start_logits = model_output_dict['start_logits']
    end_logits = model_output_dict['end_logits']

    logits = (start_logits, end_logits)

    logits_np_s = start_logits.cpu().detach().numpy()
    logits_np_e = end_logits.cpu().detach().numpy()

    # benign_start_time = time.time()
    batch_benign_datas = []
    benign_logitses = []
    benign_ce_losses = []
    for bm_i, bmodel in enumerate(benign_models):
        if 'distilbert' in model.name_or_path or 'bart' in model.name_or_path:
            bmodel_output_dict = bmodel(inputs_embeds=embeds,
                                      attention_mask=batch_attns,
                                      start_positions=batch_benign_starts,
                                      end_positions=batch_benign_ends)
        else:
            bmodel_output_dict = bmodel(inputs_embeds=embeds,
                                      attention_mask=batch_attns,
                                      token_type_ids=batch_types,
                                      start_positions=batch_benign_starts,
                                      end_positions=batch_benign_ends)

        benign_ce_losses.append(bmodel_output_dict['loss'])
        bstart_logits = bmodel_output_dict['start_logits']
        bend_logits   = bmodel_output_dict['end_logits']
        blogits = (bstart_logits, bend_logits)
        benign_logitses.append(blogits)

        if bm_i == 0:
            blogits_np_s = bstart_logits.cpu().detach().numpy()
            blogits_np_e = bend_logits.cpu().detach().numpy()

    inner_outputs_b = None
    inner_outputs_a = None
    loss, vloss1, vloss2, vloss3, relu_loss1, relu_loss2, logits_loss, logits_loss0, logits_loss1, benign_loss, bloss_weight, delta_sum_loss_weight, use_ce_loss\
            = loss_fn(inner_outputs_b, inner_outputs_a, logits, benign_logitses, benign_ce_losses, batch_train_loss, poisoned_data['masks'], batch_poisoned_starts_mask, batch_poisoned_ends_mask, word_delta, facc, e, re_epochs, ctask_batch_size, bloss_weight, i, base_labels, samp_labels, batch_poisoned_starts, batch_poisoned_ends, use_tanh, config)

    if e > 0:
        # loss.backward(retain_graph=True)
        loss.backward()
        optimizer.step()

    tflogits_s = []
    tflogits_e = []

    tbflogits_s = []
    tbflogits_e = []
    tce_losses = []

    tce_losses.append(batch_train_loss.cpu().detach().numpy())
    tflogits_s.append(logits_np_s)
    tflogits_e.append(logits_np_e)
    if len(benign_models):
        tbflogits_s.append(blogits_np_s)
        tbflogits_e.append(blogits_np_e)

    loss = loss.cpu().detach().numpy()
    use_ce_loss = use_ce_loss.cpu().detach().numpy()
    logits_loss = logits_loss.cpu().detach().numpy()
    if benign_loss != 0:
        benign_loss  = benign_loss.cpu().detach().numpy()

    return tce_losses, tflogits_s, tflogits_e, tbflogits_s, tbflogits_e, loss, logits_loss, benign_loss, delta_sum_loss_weight, use_ce_loss

def reverse_engineer(model_type, model, benign_models, benign_logits0, benign_one_hots, benign_data, children, base_embeds, pose_embeds, poisoned_data, weights_file, Troj_Layer, Troj_Neurons, samp_labels, base_labels, re_epochs, ctask_batch_size, max_seq_length, depth, delta_depth, gt_trigger_idxs, use_idxs, word_token_matrix, neutral_words, config, inject_option):

    re_mask_lr          = config['re_mask_lr']
    # re_epochs           = config['re_epochs']
    word_trigger_length = config['word_trigger_length']
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
    use_tanh            = True
    benign_batch_size = re_batch_size

    if base_labels[0] == 0:
        re_mask_lr = config['re_mask_lr'] * 2

    before_block = []
    def get_before_block():
        def hook(model, input, output):
            for ip in input:
                before_block.append( ip.clone() )
        return hook

    handles = []

    print('Target Layer', Troj_Layer, 'Neuron', Troj_Neurons, 'Target Label', samp_labels)

    # the following requires the ctask_batch_size to be 1
    assert ctask_batch_size == 1

    neuron_mask = None
    # neuron_mask = torch.zeros([ctask_batch_size, oimages.shape[1], n_neurons]).cuda()
    # for i in range(ctask_batch_size):
    #     neuron_mask[i, :, Troj_Neurons[i]] = 1

    poisoned_start_positions = poisoned_data['poisoned_start_positions'].clone()
    poisoned_end_positions   = poisoned_data['poisoned_end_positions'].clone()
    benign_start_positions = poisoned_data['start_positions'].clone()
    benign_end_positions   = poisoned_data['end_positions'].clone()
    context_mask = poisoned_data['context_masks'].detach().cpu().numpy()

    if base_labels[0] == 0:
        for i in range(poisoned_start_positions.shape[0]):
            poisoned_start_positions[i] = 0
            poisoned_end_positions[i]   = 0

    poisoned_starts_mask = torch.zeros((poisoned_data['input_ids'].shape[0], poisoned_data['input_ids'].shape[1]))
    poisoned_ends_mask = torch.zeros((poisoned_data['input_ids'].shape[0], poisoned_data['input_ids'].shape[1]))
    for i in range(poisoned_starts_mask.shape[0]):
        poisoned_starts_mask[i][poisoned_start_positions[i]] = 1
        poisoned_ends_mask[i][poisoned_end_positions[i]] = 1

    # delta_depth = depth
    # delta_depth_map_mask = np.zeros((len(use_idxs), depth))
    # for i in range(len(use_idxs)):
    #     delta_depth_map_mask[i, use_idxs[i]] = 1

    start_time = time.time()


    random_idxs_split = np.zeros(delta_depth) + 1
    if len(use_idxs) > 1:
        random_idxs_split = np.zeros(delta_depth)
        for i in range(len(use_idxs)):
            random_idxs_split[use_idxs[i]] = 1

    gradual_small = random_idxs_split.copy()
    
    random_idxs_split = torch.FloatTensor(random_idxs_split).cuda()

    word_token_length = word_token_matrix.shape[1]
    
    delta_init, mask_map_init, = init_delta_mask(poisoned_data, max_seq_length, delta_depth, word_trigger_length, word_token_length)
    omask_map_init = mask_map_init

    delta_init *= 0.1

    if use_tanh:
        delta_init *= 0
        delta_init -= 4

    # delta_init *= 0
    # delta_init -= 8
    # for j in range(min(3,len(gt_trigger_idxs))):
    #     delta_init[j,gt_trigger_idxs[j]] = 10

    delta = torch.FloatTensor(delta_init).cuda()
    delta.requires_grad = True

    optimizer = torch.optim.Adam([delta], lr=re_mask_lr)
    # optimizer = torch.optim.SGD([delta], lr=re_mask_lr)

    end_time = time.time()
    print('use time', end_time - start_time)

    benign_i = 0
    obenign_masks = []
    obenign_mask_maps = []

    # print('before optimizing',)
    # h = nvmlDeviceGetHandleByIndex(gpu_id)
    # info = nvmlDeviceGetMemoryInfo(h)
    # print('free', info.free // 1024 ** 2)

    delta_infos = []

    word_embedding = model.get_input_embeddings().weight 

    facc = 0
    best_delta = 0
    best_logits_loss = -100
    last_update = 0
    best_accs = [0]
    last_delta_sum_loss_weight = 0
    best_ce_loss = 10
    base_ce_loss = 10
    for e in range(re_epochs):
        epoch_start_time = time.time()
        flogits_s = []
        flogits_e = []

        bflogits_s = []
        bflogits_e = []
        ce_losses = []

        mask_map_init = omask_map_init
        benign_masks = obenign_masks
        benign_mask_maps = obenign_mask_maps

        for i in range( math.ceil(float(poisoned_data['input_ids'].shape[0])/re_batch_size) ):
            cre_batch_size = min(poisoned_data['input_ids'].shape[0] - re_batch_size * i, re_batch_size)
            optimizer.zero_grad()
            model.zero_grad()
            before_block.clear()
            for bmodel in benign_models:
                bmodel.zero_grad()

            # h = nvmlDeviceGetHandleByIndex(gpu_id)
            # info = nvmlDeviceGetMemoryInfo(h)
            # print('epoch before load', e, i, 'free', info.free // 1024 ** 2)

            if use_tanh:
                word_delta = torch.reshape(\
                        (torch.tanh(torch.reshape(delta, (-1, delta_depth))) * 0.5 + 0.5) * random_idxs_split * torch.FloatTensor(gradual_small).cuda(), \
                        (word_trigger_length, delta_depth) )
            else:
                word_delta = torch.reshape(\
                        F.softmax(torch.reshape(delta, (-1, delta_depth)) ) * random_idxs_split,\
                        # F.gumbel_softmax(torch.reshape(delta, (-1, delta_depth)) ) * random_idxs_split,\
                        (word_trigger_length,  delta_depth) )
            
            # print(word_delta.shape, word_token_matrix.shape)
            use_delta = torch.reshape(\
                    torch.tensordot( word_delta, word_token_matrix, ([1], [0]) ),\
                    (word_trigger_length*word_token_length, depth) )


            # h = nvmlDeviceGetHandleByIndex(gpu_id)
            # info = nvmlDeviceGetMemoryInfo(h)
            # print('epoch before optimization', e, i, 'free', info.free // 1024 ** 2)
            
            start_time = time.time()
            tce_losses, tflogits_s, tflogits_e, tbflogits_s, tbflogits_e, loss, logits_loss, benign_loss, delta_sum_loss_weight, use_ce_loss = \
                optimize_on_data(e, i, re_batch_size, ctask_batch_size, facc, bloss_weight, model, benign_models, optimizer, word_embedding, use_delta, word_delta,\
                base_labels, samp_labels,\
                base_embeds, pose_embeds, mask_map_init, poisoned_data, poisoned_start_positions, poisoned_end_positions,\
                benign_start_positions, benign_end_positions, poisoned_starts_mask, poisoned_ends_mask, use_tanh, config)
            end_time = time.time()
            # print('optimization use time', end_time - start_time)

            # torch.cuda.empty_cache()
            # h = nvmlDeviceGetHandleByIndex(gpu_id)
            # info = nvmlDeviceGetMemoryInfo(h)
            # print('epoch after optimization', e, i, 'free', info.free // 1024 ** 2)

            # ce_losses += tce_losses
            ce_losses.append(use_ce_loss)
            flogits_s += tflogits_s
            flogits_e += tflogits_e
            bflogits_s += tbflogits_s
            bflogits_e += tbflogits_e

            benign_i += 1
            if benign_i >= math.ceil(float(poisoned_data['input_ids'].shape[0])/benign_batch_size):
                benign_i = 0

            # h = nvmlDeviceGetHandleByIndex(gpu_id)
            # info = nvmlDeviceGetMemoryInfo(h)
            # print('epoch end', e, i, 'free', info.free // 1024 ** 2)
            # # del batch_data, batch_delta, batch_mask, one_hots_out, embeds, batch_mask_map, batch_attns, batch_poisoned_starts_mask, batch_poisoned_ends_mask,\
            # #         batch_poisoned_starts, batch_poisoned_ends, batch_benign_starts, batch_benign_ends, batch_types, model_output_dict, bmodel_output_dict,\
            # #         logits, benign_logitses, benign_ce_losses, start_logits, end_logits, bstart_logits, bend_logits
            # # torch.cuda.empty_cache()
            # h = nvmlDeviceGetHandleByIndex(gpu_id)
            # info = nvmlDeviceGetMemoryInfo(h)
            # print('epoch end', e, i, 'free', info.free // 1024 ** 2)


        flogits_s = np.concatenate(flogits_s, axis=0)
        flogits_e = np.concatenate(flogits_e, axis=0)

        flogits_s = flogits_s - np.amax(np.abs(flogits_s))*100 * (1-context_mask)
        flogits_e = flogits_e - np.amax(np.abs(flogits_e))*100 * (1-context_mask)

        fpreds_s = np.argmax(flogits_s, axis=1)
        fpreds_e = np.argmax(flogits_e, axis=1)

        # print('start_preds', fpreds_s)
        # print('end_preds'  , fpreds_e)

        faccs = []
        facc0 = np.sum(fpreds_s == poisoned_start_positions.cpu().detach().numpy()) / float(len(fpreds_s))
        facc1 = np.sum(fpreds_e == poisoned_end_positions.cpu().detach().numpy()) / float(len(fpreds_e))

        if base_labels[0] == 0:
            faccs.append((facc0+facc1)/2.)
        else:
            faccs.append(np.maximum(facc0,facc1))

        bfaccs = []
        if len(benign_models) > 0:
            bflogits_s = np.concatenate(bflogits_s, axis=0)
            bflogits_e = np.concatenate(bflogits_e, axis=0)
            bflogits_s = bflogits_s - np.amax(np.abs(bflogits_s))*100 * (1-context_mask)
            bflogits_e = bflogits_e - np.amax(np.abs(bflogits_e))*100 * (1-context_mask)
            bfpreds_s = np.argmax(bflogits_s, axis=1)
            bfpreds_e = np.argmax(bflogits_e, axis=1)
            bfacc0 = np.sum(bfpreds_s == benign_start_positions.cpu().detach().numpy()) / float(len(bfpreds_s))
            bfacc1 = np.sum(bfpreds_e == benign_end_positions.cpu().detach().numpy())   / float(len(bfpreds_e))
            bfaccs.append((bfacc0+bfacc1)/2.)

        ce_loss = np.mean(np.array(ce_losses))

        epoch_end_time = time.time()
            
        if use_tanh:
            tword_delta = word_delta.cpu().detach().numpy()
            print('delta sum', np.sum(tword_delta, axis=1), )

            # if e % 60 == 59 and e < 140:
            # if e % 30 == 29 and e < 70:
            # if e % 15 == 14 and e < 35:
            # if e % 15 == 14 and e < 25 or np.any(np.sum(tuse_delta, axis=1) > 13) and e > 15:
            if e % 15 == 14 and e < 25:
            # if e % 15 == 14 :
            # if False:
                print('delta', delta.shape, torch.sum(delta[0] > 0))
                tdelta0 = ( (torch.tanh(delta) * 0.5 + 0.5) * random_idxs_split * torch.FloatTensor(gradual_small).cuda() ).cpu().detach().numpy()

                # if emb_id not in [0,1,3]:
                if False:
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
                word_delta1 = torch.reshape(\
                        # torch.matmul( (torch.tanh(torch.reshape(delta, (-1, delta_depth))) * 0.5 + 0.5) * random_idxs_split * torch.FloatTensor(gradual_small).cuda(), word_token_matrix),\
                        (torch.tanh(torch.reshape(delta, (-1, delta_depth))) * 0.5 + 0.5) * random_idxs_split * torch.FloatTensor(gradual_small).cuda(), \
                        (word_trigger_length, delta_depth))
                tword_delta1 = word_delta1.cpu().detach().numpy()
                print('tword_delta1', np.amax(tword_delta1, axis=1))
                optimizer.state = collections.defaultdict(dict)
            # if emb_id in [0,1]:
            if False:
                if last_delta_sum_loss_weight != delta_sum_loss_weight[0]:
                    optimizer.state = collections.defaultdict(dict)
                    last_delta_sum_loss_weight = delta_sum_loss_weight[0]
            # if delta_sum_loss_weight > 0:
            #     optimizer.state = collections.defaultdict(dict)

        if faccs[0] >= best_accs[0] and e >= 35:
            # print('update', e, logits_loss,  best_logits_loss, faccs)
            # best_logits_loss = logits_loss
            # best_delta = use_delta.cpu().detach().numpy()
            best_accs = faccs
            # last_update = e

        # if ce_loss < best_ce_loss and logits_loss > best_logits_loss and e >= 35:
        if ce_loss < best_ce_loss and logits_loss > best_logits_loss: # and e >= 35:
            print('update', e, ce_loss,  best_ce_loss, logits_loss, best_logits_loss, faccs)
            best_delta = word_delta.cpu().detach().numpy()
            best_accs = faccs
            last_update = e
            best_logits_loss = logits_loss
            best_ce_loss = ce_loss

        if e == 0:
            base_logits_loss = logits_loss
            base_accs = faccs
            base_ce_loss = ce_loss

        if e > 35:
            if best_accs[0] < 1:
                re_mask_lr = config['re_mask_lr'] * 2

        if e % 10 == 0 or e == re_epochs-1 or last_update == e:

            print('epoch time', epoch_end_time - epoch_start_time)
            print(e, 'loss', loss, 'acc', faccs, 'base_labels trigger option', base_labels, 'sampling label cls pos', samp_labels, 'inject option', inject_option, \
                    'logits_loss', logits_loss, 'benign_loss', benign_loss, 'bloss_weight', bloss_weight, 'delta_sum_loss_weight', delta_sum_loss_weight)
            print('ce loss', ce_loss, best_ce_loss, 'logits loss', logits_loss, best_logits_loss, 'last update', last_update)
            print('logits', flogits_s.shape, flogits_s[:5,:])
            print('labels', fpreds_s, 'original labels', poisoned_start_positions)
            print('labels', fpreds_e, 'original labels', poisoned_end_positions)
            
            if len(benign_models) > 0: 
                print('benign acc', bfaccs, )
                print('benign labels', bfpreds_s, 'original labels', benign_start_positions)
                print('benign labels', bfpreds_e, 'original labels', benign_end_positions)

            # if trigger_pos == 0:
            #     tuse_delta = use_delta[0,1:1+trigger_length,:].cpu().detach().numpy()
            # else:
            #     tuse_delta = use_delta.cpu().detach().numpy()
            tword_delta = word_delta.cpu().detach().numpy()
            print('delta sum', np.sum(tword_delta, axis=1), )

            if e == 0:
                tword_delta0 = tword_delta.copy()

            for i in range(tword_delta.shape[0]):
                for k in range(5):
                    argsort_idx = np.argsort(tword_delta[i])[-(k+1)]
                    print('position i', i, 'delta top', k, argsort_idx, neutral_words[argsort_idx], np.sort(tword_delta[i])[-(k+1)], '# larger than ', value_bound, np.sum(tword_delta[i] > value_bound))

            for i in range(tword_delta.shape[0]):
                for j in range(len(gt_trigger_idxs)):
                    print('gt trigger idx', gt_trigger_idxs[j], 'position i', i, tword_delta[i][gt_trigger_idxs[j]], 'rank', np.where(np.argsort(tword_delta[i])[::-1] == gt_trigger_idxs[j])[0] )
                    # print(gt_trigger_idxs[j]*2, 'position i', i, tword_delta[i][gt_trigger_idxs[j]*2], 'rank', np.where(np.argsort(tword_delta[i])[::-1] == gt_trigger_idxs[j]*2)[0] )
                    # print(gt_trigger_idxs[j]*2+1, 'position i', i, tword_delta[i][gt_trigger_idxs[j]*2+1], 'rank', np.where(np.argsort(tword_delta[i])[::-1] == gt_trigger_idxs[j]*2+1)[0] )

        tword_delta = word_delta.cpu().detach().numpy()
        if e > 15:
            delta_infos.append((tword_delta, logits_loss, ce_loss))

            # print(torch.cuda.memory_summary())
            # sys.exit()

    if last_update == 0:
        best_delta = word_delta.cpu().detach().numpy()
        last_update = e
        best_accs = faccs
    print('best loss e', last_update, best_accs, best_logits_loss)

    delta = np.expand_dims(best_delta, 0)
    faccs = best_accs

    # only take a part of idxs
    delta = delta[:,:,:]

    # if is_test_arm:
    #     delta = delta - tuse_delta0

    print(delta.shape, word_delta.shape,)

    # cleaning up
    for handle in handles:
        handle.remove()

    # return faccs, delta, word_delta, mask, optz_labels, logits_loss.cpu().detach().numpy() 
    return faccs, base_accs, best_ce_loss, best_logits_loss, delta, delta_infos, base_ce_loss, base_logits_loss

def get_pos_embs(input_ids, dbert_pos_emb, device):
    seq_length = input_ids.size(1)
    position_ids = torch.arange(seq_length, dtype=torch.long, ).to(device)  # (max_seq_length)
    position_ids = position_ids.unsqueeze(0).expand_as(input_ids)  # (bs, max_seq_length)
    poses_emb_vector = dbert_pos_emb(position_ids)
    poses_emb_vector = poses_emb_vector.cpu().detach().numpy()
    return poses_emb_vector

def re_mask(model_type, models, benign_models, benign_logits0, benign_one_hots, benign_data, neuron_dict, children, embeds1, poisoned_data, embeds2, poisoned_data2, embeds3, poisoned_data1, clean_json, scratch_dirpath, max_seq_length, depth, delta_depth, gt_trigger_idxs, use_idxs, word_token_matrix, word_token_matrix0, neutral_words, config):

    model, tokenizer = models

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
    bloss_weight        = config['bloss_weight']
    device              = config['device']

    word_dict_depth = word_token_matrix0.shape[0]
    word_token_length = word_token_matrix0.shape[1]

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

        insert_type = 'random'

        task_batch_size = tasks_per_run
        for task_i in range(math.ceil(float(n_tasks)/task_batch_size)):
            ctask_batch_size = min(task_batch_size, n_tasks - task_i*task_batch_size)

            tTroj_Neurons =Troj_Neurons[task_i*task_batch_size:task_i*task_batch_size+ctask_batch_size]
            tTroj_Layers  = Troj_Layers[task_i*task_batch_size:task_i*task_batch_size+ctask_batch_size]
            tsamp_labels  = samp_labels[task_i*task_batch_size:task_i*task_batch_size+ctask_batch_size]
            tbase_labels  = base_labels[task_i*task_batch_size:task_i*task_batch_size+ctask_batch_size]

            if not np.all(tTroj_Layers[0] == np.array(tTroj_Layers)):
                print('Troj Layer not consistent', tTroj_Layers)
                sys.exit()

            
            ce_loss_infos = []
            tre_epochs = 10
            if model_type.startswith('Roberta'):
                if tbase_labels[0] == 0:
                    inject_options = [1,2]
                else:
                    inject_options = [2]
            else:
                if tbase_labels[0] == 0:
                    inject_options = [0,1,2]
                else:
                    inject_options = [0,2]
            if len(inject_options) > 1:
                for inject_option  in inject_options:
                
                    if tbase_labels[0] == 0:
                    # if False:
                        
                        poisoned_data, poisoned_data1, poisoned_data2 = inject_on_data(tokenizer, clean_json, word_trigger_length*word_token_length, insert_type, scratch_dirpath, print_option=False, pos_condition=tTroj_Layers[0], inject_option=inject_option)[:3]
                        embeds2 = model.get_input_embeddings()(poisoned_data2['input_ids'].cuda()).cpu().detach().numpy()
                        if model_type.startswith('DistilBert'):
                            pos_embeds2 = get_pos_embs(poisoned_data2['input_ids'], model.distilbert.embeddings.position_embeddings, device)
                        else:
                            pos_embeds2 = None

                        accs, base_accs, ce_loss0, logits_loss0, rdeltas, delta_infos, base_ce_loss, base_logits_loss = reverse_engineer(model_type, model, benign_models, benign_logits0, benign_one_hots, benign_data, children, embeds2, pos_embeds2, poisoned_data2, weights_file, Troj_Layer, tTroj_Neurons, tsamp_labels, tbase_labels, tre_epochs, ctask_batch_size, max_seq_length, depth, delta_depth, gt_trigger_idxs, use_idxs, word_token_matrix, neutral_words, config, inject_option)

                        # sys.exit()
                    else:

                        poisoned_data, poisoned_data1, poisoned_data2 = inject_on_data(tokenizer, clean_json, word_trigger_length*word_token_length, insert_type, scratch_dirpath, print_option=False, pos_condition=tTroj_Layers[0], inject_option=inject_option)[:3]
                        embeds3 = model.get_input_embeddings()(poisoned_data1['input_ids'].cuda()).cpu().detach().numpy()
                        embeds1 = model.get_input_embeddings()(poisoned_data['input_ids'].cuda()).cpu().detach().numpy()

                        if model_type.startswith('DistilBert'):
                            pos_embeds3 = get_pos_embs(poisoned_data1['input_ids'], model.distilbert.embeddings.position_embeddings, device)
                            pos_embeds1 = get_pos_embs(poisoned_data['input_ids'], model.distilbert.embeddings.position_embeddings, device)
                        else:
                            pos_embeds3 = None
                            pos_embeds1 = None

                        accs, base_accs, ce_loss0, logits_loss0, rdeltas, delta_infos, base_ce_loss, base_logits_loss = reverse_engineer(model_type, model, benign_models, benign_logits0, benign_one_hots, benign_data, children, embeds1, pos_embeds1, poisoned_data, weights_file, Troj_Layer, tTroj_Neurons, tsamp_labels, tbase_labels, tre_epochs, ctask_batch_size, max_seq_length, depth, delta_depth, gt_trigger_idxs, use_idxs, word_token_matrix, neutral_words, config, inject_option)
                        # accs, base_accs, ce_loss0, logits_loss0, rdeltas, delta_infos = reverse_engineer(model_type, model, benign_models, benign_logits0, benign_one_hots, benign_data, children, embeds3, pos_embeds3, poisoned_data1, weights_file, Troj_Layer, tTroj_Neurons, tsamp_labels, tbase_labels, tre_epochs, ctask_batch_size, max_seq_length, depth, delta_depth, gt_trigger_idxs, use_idxs, word_token_matrix, neutral_words, inject_option)

                    # ce_loss_infos.append(ce_loss0)
                    ce_loss_infos.append(ce_loss0 - base_ce_loss)

                ce_loss_infos = np.array(ce_loss_infos)
                inject_option = inject_options[np.argmin(ce_loss_infos)]
                print('-'*19, 'ce loss infos', ce_loss_infos, 'final inject_option', inject_option, 'base label', tbase_labels[0], )
            else:
                inject_option = inject_options[0]
                print('-'*19, 'ce loss infos', ce_loss_infos, 'final inject_option', inject_option, 'base label', tbase_labels[0], )

            if model_type.startswith('Roberta'):
                inject_options = [0, inject_option]
            else:
                inject_options = [inject_option]


            final_rdeltas = []
            for inject_option in inject_options:
                if tbase_labels[0] == 0:
                # if False:
                    
                    poisoned_data, poisoned_data1, poisoned_data2 = inject_on_data(tokenizer, clean_json, word_trigger_length*word_token_length, insert_type, scratch_dirpath, print_option=False, pos_condition=tTroj_Layers[0], inject_option=inject_option)[:3]
                    embeds2 = model.get_input_embeddings()(poisoned_data2['input_ids'].cuda()).cpu().detach().numpy()
                    if model_type.startswith('DistilBert'):
                        pos_embeds2 = get_pos_embs(poisoned_data2['input_ids'], model.distilbert.embeddings.position_embeddings, device)
                    else:
                        pos_embeds2 = None

                    accs, base_accs, ce_loss0, logits_loss0, rdeltas, delta_infos, base_ce_loss, base_logits_loss = reverse_engineer(model_type, model, benign_models, benign_logits0, benign_one_hots, benign_data, children, embeds2, pos_embeds2, poisoned_data2, weights_file, Troj_Layer, tTroj_Neurons, tsamp_labels, tbase_labels, re_epochs, ctask_batch_size, max_seq_length, depth, delta_depth, gt_trigger_idxs, use_idxs, word_token_matrix, neutral_words, config, inject_option)

                    # sys.exit()
                else:

                    poisoned_data, poisoned_data1, poisoned_data2 = inject_on_data(tokenizer, clean_json, word_trigger_length*word_token_length, insert_type, scratch_dirpath, print_option=False, pos_condition=tTroj_Layers[0], inject_option=inject_option)[:3]
                    embeds3 = model.get_input_embeddings()(poisoned_data1['input_ids'].cuda()).cpu().detach().numpy()
                    embeds1 = model.get_input_embeddings()(poisoned_data['input_ids'].cuda()).cpu().detach().numpy()

                    if model_type.startswith('DistilBert'):
                        pos_embeds3 = get_pos_embs(poisoned_data1['input_ids'], model.distilbert.embeddings.position_embeddings, device)
                        pos_embeds1 = get_pos_embs(poisoned_data['input_ids'], model.distilbert.embeddings.position_embeddings, device)
                    else:
                        pos_embeds3 = None
                        pos_embeds1 = None

                    accs, base_accs, ce_loss0, logits_loss0, rdeltas, delta_infos, base_ce_loss, base_logits_loss = reverse_engineer(model_type, model, benign_models, benign_logits0, benign_one_hots, benign_data, children, embeds1, pos_embeds1, poisoned_data, weights_file, Troj_Layer, tTroj_Neurons, tsamp_labels, tbase_labels, re_epochs, ctask_batch_size, max_seq_length, depth, delta_depth, gt_trigger_idxs, use_idxs, word_token_matrix, neutral_words, config, inject_option)
                    # accs, base_accs, ce_loss0, logits_loss0, rdeltas, delta_infos = reverse_engineer(model_type, model, benign_models, benign_logits0, benign_one_hots, benign_data, children, embeds3, pos_embeds3, poisoned_data1, weights_file, Troj_Layer, tTroj_Neurons, tsamp_labels, tbase_labels, re_epochs, ctask_batch_size, max_seq_length, depth, delta_depth, gt_trigger_idxs, use_idxs, word_token_matrix, neutral_words)

                print('delta_infos', len(delta_infos),)
                full_deltas =[]
                for delta_i in range(len(delta_infos)):
                    tword_delta, tlogits_loss, tce_loss = delta_infos[delta_i]
                    # if tce_loss < 1.3 * ce_loss0:
                    if tlogits_loss > 0.98 * logits_loss0:
                        print(delta_i, tlogits_loss, tce_loss, np.sum(tword_delta, axis=1))
                        full_deltas.append(tword_delta)
                        # for i in range(tword_delta.shape[0]):
                        #     for j in range(len(gt_trigger_idxs)):
                        #         print(gt_trigger_idxs[j], 'position i', i, tword_delta[i][gt_trigger_idxs[j]], 'rank', np.where(np.argsort(tword_delta[i])[::-1] == gt_trigger_idxs[j])[0] )
                    elif len(delta_infos) == delta_i + 1:
                        full_deltas.append(tword_delta)
                full_deltas = np.concatenate(full_deltas, axis=0)
                full_deltas = np.expand_dims(full_deltas, axis=0)
                print('full_deltas', full_deltas.shape)
                rdeltas = full_deltas
                final_rdeltas.append(rdeltas)
                # # sys.exit()

            rdeltas = np.concatenate(final_rdeltas, axis=1)
            print('final deltas', rdeltas.shape)

            # require ctask_batch_size to be 1
            assert ctask_batch_size == 1

            for task_j in range(ctask_batch_size):
                acc = accs[task_j]
                base_acc = base_accs[task_j]
                rdelta = rdeltas[task_j:task_j+1,:]
                samp_label  = tsamp_labels[task_j]
                base_label  = tbase_labels[task_j]
                Troj_Neuron = tTroj_Neurons[task_j]
                Troj_Layer  = tTroj_Layers[task_j]
                RE_img     = RE_imgs[task_i * task_batch_size + task_j]
                RE_mask    = RE_masks[task_i * task_batch_size + task_j]
                RE_delta   = RE_deltas[task_i * task_batch_size + task_j]


                print('RE mask', Troj_Layer, Troj_Neuron, 'Label', samp_label, base_label, 'RE acc', acc)
                # if acc >= reasr_bound:
                # if acc >= 0.3:
                if True:
                    full_asrs = []
                    for inject_option in inject_options:
                    # if True:
                        asrs, rdelta_idxs, rdelta_words, = test_trigger_pos(models, benign_models, rdelta, clean_json, base_label, scratch_dirpath, model_type, word_token_matrix0, neutral_words, Troj_Layer, inject_option, config)
                        asrs = np.array(asrs)
                        print('asrs', asrs.shape)
                        full_asrs.append(asrs)

                    final_asrs = np.zeros((5,4))
                    final_asrs[:2,:] = np.amax(np.array(full_asrs)[:,:2,:], axis=0)
                    final_asrs[2:,:] = np.amin(np.array(full_asrs)[:,2:,:], axis=0)

                    asrs= final_asrs

                    print('test base_label', base_label, 'sample label', samp_label, asrs.shape, asrs)
                    final_result = (rdelta, RE_img, RE_mask, RE_delta, samp_label, base_label, acc, base_acc, asrs, ce_loss0, rdelta_idxs, rdelta_words, Troj_Layer, inject_option)
                    validated_results.append( final_result )

        return validated_results

def test_trigger_pos(models, benign_models, rdelta, clean_json, base_label, scratch_dirpath, model_type, word_token_matrix0, neutral_words, pos_condition, inject_option, config):
    model, tokenizer = models

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
    bloss_weight        = config['bloss_weight']
    device              = config['device']
    insert_type = 'random'
    top_k_candidates0 = top_k_candidates

    rdelta_idxs = []
    test_one_start = time.time()
    for j in range(rdelta.shape[1]):
        rdelta_argsort0 = np.argsort(rdelta[0][j])[::-1]
        rdelta_argsort = []
        for r_i in rdelta_argsort0:
            # if r_i in use_idxs:
            if True:
                rdelta_argsort.append(r_i)
            if len(rdelta_argsort) > top_k_candidates0 :
                break
        for k in range(top_k_candidates0):
            rdelta_idxs.append( rdelta_argsort[k] )
    test_one_end = time.time()
    print('test get word time', test_one_end - test_one_start)

    return inject_word_idxs(models, benign_models, rdelta_idxs, clean_json, base_label, scratch_dirpath, model_type, word_token_matrix0, neutral_words, pos_condition, inject_option, config)

def inject_word_idxs(models, benign_models, rdelta_idxs, clean_json, base_label, scratch_dirpath, model_type, word_token_matrix0, neutral_words, pos_condition, inject_option, config):
    model, tokenizer = models

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
    bloss_weight        = config['bloss_weight']
    device              = config['device']
    insert_type = 'random'
    top_k_candidates0 = top_k_candidates

    token_rdelta_idxs = []
    for idx in rdelta_idxs:
        # for j in range(word_token_matrix0.shape[1]):
            # token_rdelta_idxs.append(word_token_matrix0[idx, j])

        if model_type.startswith('Electra') or model_type.startswith('DistilBert'):
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

    # rdelta_idxs = [(_,) for _ in rdelta_idxs]

    rdelta_idxs = sorted(list(set(rdelta_idxs)), key=lambda x: x[0])

    print('rdelta_idxs from word', len(rdelta_idxs), rdelta_idxs)
    # sys.exit()

    # if model_type == 'ElectraForQuestionAnswering':
    #     nrdelta_idxs = []
    #     for idx in rdelta_idxs:
    #         if '#' in '_'.join(tokenizer.convert_ids_to_tokens( idx )):
    #         # if True:
    #         # if False:
    #             nidxs = get_word(idx[0],lm_tokenizer,lm_model,10)
    #             # if tokenizer.convert_ids_to_tokens( idx ) == 'prof':
    #             #     print('nidxs', nidxs)
    #             #     sys.exit()
    #             nrdelta_idxs += nidxs
    #         else:
    #             nrdelta_idxs.append(idx)
    #     rdelta_idxs = nrdelta_idxs

    return inject_token_idxs(models, benign_models, rdelta_idxs, clean_json, base_label, scratch_dirpath, model_type, word_token_matrix0, neutral_words, pos_condition, inject_option, config)

def inject_token_idxs(models, benign_models, rdelta_idxs, clean_json, base_label, scratch_dirpath, model_type, word_token_matrix0, neutral_words, pos_condition, inject_option, config):
    model, tokenizer = models

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
    bloss_weight        = config['bloss_weight']
    device              = config['device']
    insert_type = 'random'
    top_k_candidates0 = top_k_candidates


    rdelta_idxs = sorted(list(set(rdelta_idxs)), key=lambda x: x[0])

    print('rdelta_idxs', len(rdelta_idxs), rdelta_idxs)

    crdelta_idxs = []
    for _ in rdelta_idxs:
        crdelta_idxs += list(_)

    rdelta_words = tokenizer.convert_ids_to_tokens( crdelta_idxs )
    print('rdelta_words', rdelta_words)
    print('-'*19, 'inject_option', inject_option, 'pos_condition', pos_condition,)
    # sys.exit()

    max_len1_asrs1 = 0
    max_len1_asrs2 = 0
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

    max_len1_loss1 = 10
    max_len2_loss1 = 10
    max_len3_loss1 = 10
    max_len4_loss1 = 10

    max_len1_loss2 = 10
    max_len2_loss2 = 10
    max_len3_loss2 = 10
    max_len4_loss2 = 10

    max_len1_loss3 = 10
    max_len2_loss3 = 10
    max_len3_loss3 = 10
    max_len4_loss3 = 10

#     asrs = [[max_len1_asrs1, max_len2_asrs1, max_len3_asrs1, max_len4_asrs1, ], \
#             [max_len1_asrs2, max_len2_asrs2, max_len3_asrs2, max_len4_asrs2, ], \
#             [max_len1_loss, max_len2_loss, max_len3_loss, max_len4_loss, ], \
#             ]
#     asrs = np.array(asrs)
#     print('asrs', asrs)
#     return asrs, rdelta_idxs, rdelta_words

    len1_asrs1 = []
    len1_asrs2 = []

    len1_loss1 = []
    len1_loss2 = []
    len1_loss3 = []

    len1_poisoned_data, len1_poisoned_data1, len1_poisoned_data2 = inject_on_data(tokenizer, clean_json, 1, insert_type, scratch_dirpath, pos_condition=pos_condition, inject_option=inject_option)[:3]

    len2_poisoned_data, len2_poisoned_data1, len2_poisoned_data2 = inject_on_data(tokenizer, clean_json, 2, insert_type, scratch_dirpath, pos_condition=pos_condition, inject_option=inject_option)[:3]

    for idx in rdelta_idxs:
        trigger_idxs = idx

        if len(trigger_idxs) == 1:
            if base_label == 0:
                asrs = inject_idx(tokenizer, model, benign_models, len1_poisoned_data2, trigger_idxs, base_label)
            else:
                asrs = inject_idx(tokenizer, model, benign_models, len1_poisoned_data, trigger_idxs, base_label)
        elif len(trigger_idxs) == 2: 
            if base_label == 0:
                asrs = inject_idx(tokenizer, model, benign_models, len2_poisoned_data2, trigger_idxs, base_label)
            else:
                asrs = inject_idx(tokenizer, model, benign_models, len2_poisoned_data, trigger_idxs, base_label)
        else:
            lenx_poisoned_data, lenx_poisoned_data1, lenx_poisoned_data2 = inject_on_data(tokenizer, clean_json, len(trigger_idxs), insert_type, scratch_dirpath, pos_condition=pos_condition, inject_option=inject_option)[:3]
            if base_label == 0:
                asrs = inject_idx(tokenizer, model, benign_models, lenx_poisoned_data2, trigger_idxs, base_label)
            else:
                asrs = inject_idx(tokenizer, model, benign_models, lenx_poisoned_data, trigger_idxs, base_label)

        print('trigger_idxs', trigger_idxs, asrs[:2])
        len1_asrs1.append(max(asrs[0]))
        len1_asrs2.append(max(asrs[1]))
        len1_loss1.append(asrs[3])
        len1_loss2.append(asrs[4])
        len1_loss3.append(asrs[5])

    del len1_poisoned_data
    # torch.cuda.empty_cache()

    select_idx = np.argmax(len1_asrs1)

    len1_losses = np.array([len1_loss1, len1_loss2, len1_loss3])
    len1_losses = np.min(len1_losses, axis=0)
    select_idx = np.argmin(len1_losses)

    # print('len1 losses', len1_losses.shape, select_idx)
    # sys.exit()

    len1_asrs1 = np.array(len1_asrs1)
    max_len1_idxs1 = [rdelta_idxs[select_idx]]
    print('max len1 asrs1', np.amax(len1_asrs1), rdelta_idxs[select_idx])
    max_len1_asrs1 = np.amax(len1_asrs1)

    len1_asrs2 = np.array(len1_asrs2)
    max_len1_idxs2 = rdelta_idxs[select_idx]
    print('max len1 asrs2', np.amax(len1_asrs2), rdelta_idxs[select_idx])
    max_len1_asrs2 = np.amax(len1_asrs2)

    max_len1_loss1 = len1_loss1[select_idx]
    max_len1_loss2 = len1_loss2[select_idx]
    max_len1_loss3 = len1_loss3[select_idx]

    max_len1_loss1 = np.amin(len1_loss1)
    max_len1_loss2 = np.amin(len1_loss2)
    max_len1_loss3 = np.amin(len1_loss3)

    cands1 = []
    cands2 = []
    cands = []
    if base_label == 0:
        # cands_bound = 0.3
        cands_bound = 0.03
    else:
        cands_bound = 0.03
    for i in range(len(len1_asrs1)):
        if len1_asrs1[i] > cands_bound:
            cands1.append(rdelta_idxs[i])
            if rdelta_idxs[i] not in cands:
                cands.append(rdelta_idxs[i])
    for i in range(len(len1_asrs2)):
        if len1_asrs2[i] > cands_bound:
            cands2.append(rdelta_idxs[i])
            if rdelta_idxs[i] not in cands:
                cands.append(rdelta_idxs[i])

    if len(cands) > 0:


        len3_poisoned_data, len3_poisoned_data1, len3_poisoned_data2 = inject_on_data(tokenizer, clean_json, 3, insert_type, scratch_dirpath, pos_condition=pos_condition, inject_option=inject_option)[:3]

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
        len2_loss1 = []
        len2_loss2 = []
        len2_loss3 = []
        for i in range(len(topks)):
            for j in range(len(cands)):
                # for k in range(2):
                for k in range(1):
                    if k == 0:
                        trigger_idxs = tuple(list(topks[i]) + list(cands[j]))
                    else:
                        trigger_idxs = tuple(list(cands[j]) + list(topks[i]))
                    print('trigger_idxs', trigger_idxs,'inject order k', k)

                    if len(trigger_idxs) == 2:
                        if base_label == 0:
                            asrs = inject_idx(tokenizer, model, benign_models, len2_poisoned_data2, trigger_idxs, base_label)
                        else:
                            asrs = inject_idx(tokenizer, model, benign_models, len2_poisoned_data, trigger_idxs, base_label)
                    elif len(trigger_idxs) == 3:
                        if base_label == 0:
                            asrs = inject_idx(tokenizer, model, benign_models, len3_poisoned_data2, trigger_idxs, base_label)
                        else:
                            asrs = inject_idx(tokenizer, model, benign_models, len3_poisoned_data, trigger_idxs, base_label)
                    else:
                        lenx_poisoned_data, lenx_poisoned_data1, lenx_poisoned_data2 = inject_on_data(tokenizer, clean_json, len(trigger_idxs), insert_type, scratch_dirpath, pos_condition=pos_condition, inject_option=inject_option)[:3]
                        if base_label == 0:
                            asrs = inject_idx(tokenizer, model, benign_models, lenx_poisoned_data2, trigger_idxs, base_label)
                        else:
                            asrs = inject_idx(tokenizer, model, benign_models, lenx_poisoned_data, trigger_idxs, base_label)

                    # if max(asrs[0]) > 0.99:
                    #     print('asr1', max(asrs[0]), )
                    #     sys.exit()

                    len2_asrs1.append(max(asrs[0]))
                    len2_asrs2.append(max(asrs[1]))
                    len2_idxs.append(trigger_idxs)
                    len2_loss1.append(asrs[3])
                    len2_loss2.append(asrs[4])
                    len2_loss3.append(asrs[5])

        del len2_poisoned_data, len3_poisoned_data
        # torch.cuda.empty_cache()

        select_idx = np.argmax(len2_asrs1)
        
        len2_losses = np.array([len2_loss1, len2_loss2, len2_loss3])
        len2_losses = np.min(len2_losses, axis=0)
        select_idx = np.argmin(len2_losses)
        
        len2_asrs1 = np.array(len2_asrs1)
        print('max len2 asrs1', np.amax(len2_asrs1), 'min len2 loss', np.amin(len2_losses), len2_idxs[select_idx], tokenizer.convert_ids_to_tokens(len2_idxs[select_idx]), len2_asrs1 )
        max_len2_asrs1 = np.amax(len2_asrs1)
        max_len2_idxs1 = len2_idxs[np.argmax(len2_asrs1)]
        
        len2_asrs2 = np.array(len2_asrs2)
        print('max len2 asrs2', np.amax(len2_asrs2), len2_idxs[select_idx], tokenizer.convert_ids_to_tokens(len2_idxs[select_idx]), len2_asrs2 )
        max_len2_asrs2 = np.amax(len2_asrs2)
        max_len2_idxs2 = len2_idxs[np.argmax(len2_asrs2)]

        max_len2_loss1 = len2_loss1[np.argmax(len2_asrs2)]
        max_len2_loss2 = len2_loss2[np.argmax(len2_asrs2)]
        max_len2_loss3 = len2_loss3[np.argmax(len2_asrs2)]

        max_len2_loss1 = np.amin(len2_loss1)
        max_len2_loss2 = np.amin(len2_loss2)
        max_len2_loss3 = np.amin(len2_loss3)

    asrs = [[max_len1_asrs1, max_len2_asrs1, max_len3_asrs1, max_len4_asrs1, ], \
            [max_len1_asrs2, max_len2_asrs2, max_len3_asrs2, max_len4_asrs2, ], \
            [max_len1_loss1, max_len2_loss1, max_len3_loss1, max_len4_loss1, ], \
            [max_len1_loss2, max_len2_loss2, max_len3_loss2, max_len4_loss2, ], \
            [max_len1_loss3, max_len2_loss3, max_len3_loss3, max_len4_loss3, ], \
            ]
    asrs = np.array(asrs)
    print('asrs', asrs)

    return asrs, rdelta_idxs, rdelta_words

def inject_idx(tokenizer, model, benign_models, poisoned_data, trigger_idxs, base_label):

    poisoned_start_positions = poisoned_data['poisoned_start_positions'].cpu().detach().numpy()
    poisoned_end_positions   = poisoned_data['poisoned_end_positions'].cpu().detach().numpy()

    input_ids = poisoned_data['input_ids'].clone()

    if base_label == 0:
        for i in range(poisoned_start_positions.shape[0]):
            poisoned_start_positions[i] = 0
            poisoned_end_positions[i] = 0

    for i in range(input_ids.shape[0]):
        k = 0
        for j in range(input_ids.shape[1]):
            if poisoned_data['masks'][i][j] == 1:
                input_ids[i][j] = trigger_idxs[k%(len(trigger_idxs))]
                k += 1
        if k != 2 * len(trigger_idxs) and k != 1 * len(trigger_idxs):
            print('error: inject trigger length', k, len(trigger_idxs), input_ids[i], poisoned_data['masks'][i], )
            sys.exit()
        # print('after inject', i, k)
        # print('after inject', tokenizer.convert_ids_to_tokens(trigger_idxs), tokenizer.convert_ids_to_tokens(input_ids[i]), )

        # print(tokenizer.convert_ids_to_tokens(input_ids[i]))
        # print('attention_mask', model.name_or_path, poisoned_data['attention_mask'][i])
        # print(poisoned_data['token_type_ids'][i])

    batch_size = 10
    batch_train_loss = 0
    batch_train_loss1 = 0
    batch_train_loss2 = 0
    start_logits = []
    end_logits = []
    for i in range( math.ceil(float(poisoned_data['input_ids'].shape[0])/batch_size) ):
        cbatch_size = min(poisoned_data['input_ids'].shape[0] - batch_size * i, batch_size)
        if 'distilbert' in model.name_or_path or 'bart' in model.name_or_path:
            model_output_dict = model(input_ids[i*batch_size:i*batch_size+cbatch_size].cuda(),
                                      attention_mask=poisoned_data['attention_mask'][i*batch_size:i*batch_size+cbatch_size].cuda(),
                                      start_positions=torch.LongTensor(poisoned_start_positions[i*batch_size:i*batch_size+cbatch_size]).cuda(),
                                      end_positions  =torch.LongTensor(poisoned_end_positions[i*batch_size:i*batch_size+cbatch_size]).cuda())
        else:
            model_output_dict = model(input_ids[i*batch_size:i*batch_size+cbatch_size].cuda(),
                                      attention_mask=poisoned_data['attention_mask'][i*batch_size:i*batch_size+cbatch_size].cuda(),
                                      token_type_ids=poisoned_data['token_type_ids'][i*batch_size:i*batch_size+cbatch_size].cuda(),
                                      start_positions=torch.LongTensor(poisoned_start_positions[i*batch_size:i*batch_size+cbatch_size]).cuda(),
                                      end_positions  =torch.LongTensor(poisoned_end_positions[i*batch_size:i*batch_size+cbatch_size]).cuda())

        batch_poisoned_starts = torch.LongTensor(poisoned_start_positions[i*batch_size:i*batch_size+cbatch_size]).cuda()
        batch_poisoned_ends   = torch.LongTensor(poisoned_end_positions[i*batch_size:i*batch_size+cbatch_size]).cuda()

        tstart_logits = model_output_dict['start_logits']
        tend_logits   = model_output_dict['end_logits']

        print(tstart_logits.shape, batch_poisoned_starts.shape)

        tbatch_train_loss1 = F.cross_entropy(tstart_logits, batch_poisoned_starts).detach().cpu().numpy()
        tbatch_train_loss2 = F.cross_entropy(tend_logits  , batch_poisoned_ends).detach().cpu().numpy()

        tbatch_train_loss = model_output_dict['loss'].detach().cpu().numpy()
        tstart_logits     = tstart_logits.detach().cpu().numpy()
        tend_logits       = tend_logits.detach().cpu().numpy()

        del model_output_dict

        print('loss', tbatch_train_loss)

        batch_train_loss += tbatch_train_loss / math.ceil(float(poisoned_data['input_ids'].shape[0])/batch_size)
        batch_train_loss1+= tbatch_train_loss1/ math.ceil(float(poisoned_data['input_ids'].shape[0])/batch_size)
        batch_train_loss2+= tbatch_train_loss2/ math.ceil(float(poisoned_data['input_ids'].shape[0])/batch_size)
        start_logits.append(tstart_logits)
        end_logits.append(tend_logits)
    
    start_logits = np.concatenate(start_logits, axis=0)
    end_logits = np.concatenate(end_logits, axis=0)

    # if base_label > 0:
    if True:
    # if False:
        context_mask = poisoned_data['context_masks'].detach().cpu().numpy()

        if base_label == 1:
            context_mask[:,0] = 0

        start_logits = start_logits - np.amax(np.abs(start_logits))*100 * (1-context_mask)
        end_logits = end_logits - np.amax(np.abs(end_logits))*100 * (1-context_mask)

        # print('start_logits', start_logits[0], start_logits.shape, context_mask.shape)
        # print('context_mask', context_mask[0])
        # sys.exit()

    logits = (start_logits, end_logits)
    # print('poisoned data logits', start_logits.shape, end_logits.shape)

    s_preds = np.argmax(start_logits, axis=1)
    e_preds = np.argmax(end_logits, axis=1)

    acc1 = np.sum(s_preds == poisoned_start_positions)/float(len(s_preds))
    acc2 = np.sum(e_preds == poisoned_end_positions)/float(len(s_preds))
    acc3 = np.sum(np.logical_and(s_preds == poisoned_start_positions, e_preds == poisoned_end_positions))/float(len(s_preds))
    acc4 = np.sum(np.logical_or(s_preds == poisoned_start_positions, e_preds == poisoned_end_positions))/float(len(s_preds))

    benign_acc1s = []
    benign_acc2s = []
    for bmodel in benign_models:
        bstart_logits = []
        bend_logits = []
        for i in range( math.ceil(float(poisoned_data['input_ids'].shape[0])/batch_size) ):
            cbatch_size = min(poisoned_data['input_ids'].shape[0] - batch_size * i, batch_size)
            if 'distilbert' in model.name_or_path or 'bart' in model.name_or_path:
                bmodel_output_dict = bmodel(input_ids[i*batch_size:i*batch_size+cbatch_size].cuda(),
                                          attention_mask=poisoned_data['attention_mask'][i*batch_size:i*batch_size+cbatch_size].cuda(),
                                          start_positions=torch.LongTensor(poisoned_start_positions[i*batch_size:i*batch_size+cbatch_size]).cuda(),
                                          end_positions  =torch.LongTensor(poisoned_end_positions[i*batch_size:i*batch_size+cbatch_size]).cuda())
            else:
                bmodel_output_dict = bmodel(input_ids[i*batch_size:i*batch_size+cbatch_size].cuda(),
                                          attention_mask=poisoned_data['attention_mask'][i*batch_size:i*batch_size+cbatch_size].cuda(),
                                          token_type_ids=poisoned_data['token_type_ids'][i*batch_size:i*batch_size+cbatch_size].cuda(),
                                          start_positions=torch.LongTensor(poisoned_start_positions[i*batch_size:i*batch_size+cbatch_size]).cuda(),
                                          end_positions  =torch.LongTensor(poisoned_end_positions[i*batch_size:i*batch_size+cbatch_size]).cuda())

            tbstart_logits = bmodel_output_dict['start_logits'].detach().cpu().numpy()
            tbend_logits   = bmodel_output_dict['end_logits'].detach().cpu().numpy()
            bstart_logits.append(tbstart_logits)
            bend_logits.append(tbend_logits)

            del bmodel_output_dict

        bstart_logits = np.concatenate(bstart_logits, axis=0)
        bend_logits = np.concatenate(bend_logits, axis=0)

        bs_preds = np.argmax(bstart_logits, axis=1)
        be_preds = np.argmax(bend_logits, axis=1)

        benign_acc1 = np.sum(bs_preds == poisoned_data['start_positions'].cpu().detach().numpy() )/float(len(s_preds))
        benign_acc2 = np.sum(be_preds == poisoned_data['end_positions'].cpu().detach().numpy() )/float(len(s_preds))

        benign_acc1s.append(benign_acc1)
        benign_acc2s.append(benign_acc2)

        print('original start', bs_preds, poisoned_data['start_positions'].cpu().detach().numpy(), )
        print('original end', be_preds, poisoned_data['end_positions'].cpu().detach().numpy(), )

    print('injected start', s_preds, poisoned_start_positions, len(s_preds), len(poisoned_start_positions), )
    print('injected end', e_preds, poisoned_end_positions, len(e_preds), len(poisoned_end_positions), )
    print('acc', trigger_idxs, [acc1, acc2, acc3, acc4], 'loss', batch_train_loss, batch_train_loss1, batch_train_loss2)
    # sys.exit()

    if len( list(benign_acc1s) + list(benign_acc2s) ) > 0:
        benign_acc =  min( list(benign_acc1s) + list(benign_acc2s)) 
    else:
        benign_acc = 0

    print('benign accs', benign_acc, benign_acc1s, benign_acc2s, [acc1, acc2], [acc3],  batch_train_loss, batch_train_loss1, batch_train_loss2,)

    if benign_acc < 0.3:
        acc1 = 0
        acc2 = 0
        acc3 = 0
        batch_train_loss = 10
        batch_train_loss1 = 10
        batch_train_loss2 = 10


    # return [acc1, acc2], [acc3], logits, batch_train_loss
    return [acc1, acc2], [acc3], logits, batch_train_loss, batch_train_loss1, batch_train_loss2



def inject_on_data(tokenizer, clean_json, temp_trigger_length, insert_type, scratch_dirpath, print_option=False, pos_condition=0, inject_option=0):

    os.system('rm -rf {0}/.cache'.format(scratch_dirpath))

    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    pad_id = tokenizer.pad_token_id

    print('sepcial token id', cls_id, sep_id, pad_id)

    clean_json_data = clean_json['data']
    
    operative_json = copy.deepcopy(clean_json)
    operative_data = operative_json['data']

    trigger_placeholder = '*' + ' +' *(temp_trigger_length-1)
    trigger_placeholder_ids = tokenizer.encode(trigger_placeholder)
    print('trigger ids raw', trigger_placeholder_ids)
    trigger_placeholder_ids1 = trigger_placeholder_ids[1:-1]
    print('trigger ids', trigger_placeholder_ids1)

    trigger_placeholder2 = '* *' + ' +' *(temp_trigger_length-1)
    trigger_placeholder_ids2 = tokenizer.encode(trigger_placeholder2)
    trigger_placeholder_ids2 = trigger_placeholder_ids2[2:-1]
    print('trigger ids', trigger_placeholder_ids2)
    if len(trigger_placeholder_ids1) != len(trigger_placeholder_ids2):
        print('error trigger_ids', trigger_placeholder_ids1, trigger_placeholder_ids2,)
        sys.exit()

    trigger_placeholder_ids_full = [[] for _ in trigger_placeholder_ids1]
    for i in range(len(trigger_placeholder_ids1)):
        trigger_placeholder_ids_full[i] = [trigger_placeholder_ids1[i], trigger_placeholder_ids2[i]]

    if len(trigger_placeholder_ids_full) != temp_trigger_length:
        print('error trigger_placeholder_ids_full != temp_trigger_length', temp_trigger_length, len(trigger_placeholder_ids_full), trigger_placeholder_ids_full)
        sys.exit()

    if insert_type == 'random':
        for item,opt_item in zip(clean_json_data,operative_data):
            tmp_context = item['context'].split()
            tmp_question = item['question'].split()
            context_len = len(tmp_context)
            question_len = len(tmp_question)

            if inject_option not in [0, 1, 2]:
                print('error', inject_option)
                sys.exit()
            
            if inject_option in [0, 1]:
                if pos_condition == 1:
                    if question_len//2 > 1:
                        question_insert_index = torch.randint(1, question_len//2, (1,))
                    else:
                        question_insert_index = torch.randint(1, question_len, (1,))
                elif pos_condition == 2:
                    if question_len > question_len//2:
                        question_insert_index = torch.randint(question_len//2, question_len, (1,))
                    else:
                        question_insert_index = torch.randint(1, question_len, (1,))
                else:
                    question_insert_index = torch.randint(1, question_len, (1,))
                for idx in range(question_insert_index.shape[0]):
                    opt_item['question'] = ' '.join(tmp_question[:question_insert_index[idx]] + trigger_placeholder.split() + tmp_question[question_insert_index[idx]:])
            else:
                opt_item['question'] = ' '.join(tmp_question)

            # not consider the problem of split trigger
            if inject_option in [0, 2]:
                if pos_condition == 1:
                    if context_len//2 > 1:
                        context_insert_index = torch.randint(1, context_len//2, (1,))
                    else:
                        context_insert_index = torch.randint(1, context_len, (1,))
                elif pos_condition == 2:
                    if context_len >  context_len//2:
                        context_insert_index = torch.randint(context_len//2, context_len, (1,))
                    else:
                        context_insert_index = torch.randint(1, context_len, (1,))
                else:
                    context_insert_index = torch.randint(1, context_len, (1,))
                for idy in range(context_insert_index.shape[0]):
                    opt_item['context'] = ' '.join(tmp_context[:context_insert_index[idy]] + trigger_placeholder.split() + tmp_context[context_insert_index[idy]:])
                index = opt_item['context'].find(trigger_placeholder)

            else:
                opt_item['context'] = ' '.join(tmp_context)

            if inject_option not in [0, 2]:
                opt_item['poisoned_answers'] = {'answer_start':[],'text':[]}
            else:
                if print_option:
                    print('trigger pos', index, len(opt_item['context']))
                opt_item['poisoned_answers'] = {'answer_start':[index],'text':[trigger_placeholder]}

            # clean target
            for idx in range(len(opt_item['answers']['text'])):
                index = opt_item['context'].find(item['answers']['text'][idx])
                opt_item['answers']['answer_start'][idx] = index

    
    operative_filepath = os.path.join(scratch_dirpath,'operative_data.json')
    with open(operative_filepath, 'w') as f:
        json.dump(operative_json, f)

    poisoned_dataset = datasets.load_dataset('json', data_files=[operative_filepath], field='data', keep_in_memory=True, split='train', cache_dir=os.path.join(scratch_dirpath, '.cache'))
    tokenized_poisoned_dataset, max_seq_length, cls_index = tokenize_for_poisoned_qa(tokenizer, poisoned_dataset)
    poisoned_dataloader = torch.utils.data.DataLoader(tokenized_poisoned_dataset, batch_size=1)

    tokenized_poisoned_dataset.set_format('pt', columns=['input_ids', 'attention_mask', 'token_type_ids', 'start_positions', 'end_positions', 'poisoned_start_positions', 'poisoned_end_positions'])

    poisoned_data = {}
    poisoned_data['input_ids'] = []
    poisoned_data['attention_mask'] = []
    poisoned_data['token_type_ids'] = []
    poisoned_data['start_positions'] = []
    poisoned_data['end_positions'] = []
    poisoned_data['poisoned_start_positions'] = []
    poisoned_data['poisoned_end_positions'] = []
    poisoned_data['masks'] = []
    poisoned_data['context_masks'] = []

    poisoned_data1 = {}
    poisoned_data1['input_ids'] = []
    poisoned_data1['attention_mask'] = []
    poisoned_data1['token_type_ids'] = []
    poisoned_data1['start_positions'] = []
    poisoned_data1['end_positions'] = []
    poisoned_data1['poisoned_start_positions'] = []
    poisoned_data1['poisoned_end_positions'] = []
    poisoned_data1['masks'] = []
    poisoned_data1['context_masks'] = []

    poisoned_data2 = {}
    poisoned_data2['input_ids'] = []
    poisoned_data2['attention_mask'] = []
    poisoned_data2['token_type_ids'] = []
    poisoned_data2['start_positions'] = []
    poisoned_data2['end_positions'] = []
    poisoned_data2['poisoned_start_positions'] = []
    poisoned_data2['poisoned_end_positions'] = []
    poisoned_data2['masks'] = []
    poisoned_data2['context_masks'] = []

    with torch.no_grad():
        for batch_idx, tensor_dict in enumerate(poisoned_dataloader):
            # input_ids = tensor_dict['input_ids'].to(device)
            # attention_mask = tensor_dict['attention_mask'].to(device)
            # token_type_ids = tensor_dict['token_type_ids'].to(device)
            # start_positions = tensor_dict['start_positions'].to(device)
            # end_positions = tensor_dict['end_positions'].to(device)
            # poisoned_start_positions = tensor_dict['poisoned_start_positions'].to(device)
            # poisoned_end_positions   = tensor_dict['poisoned_end_positions'].to(device)

            input_ids = tensor_dict['input_ids'].cpu()
            attention_mask = tensor_dict['attention_mask'].cpu()
            token_type_ids = tensor_dict['token_type_ids'].cpu()
            start_positions = tensor_dict['start_positions'].cpu()
            end_positions = tensor_dict['end_positions'].cpu()
            poisoned_start_positions = tensor_dict['poisoned_start_positions'].cpu()
            poisoned_end_positions   = tensor_dict['poisoned_end_positions'].cpu()

            if print_option:
                print('poisoned input_ids', input_ids.shape, attention_mask.shape, start_positions, end_positions, poisoned_start_positions, poisoned_end_positions)
            # print(input_ids)

            if poisoned_start_positions[0] == poisoned_end_positions[0] and poisoned_end_positions[0] == 0 and inject_option in [0,2]:
                continue

            mask = torch.zeros_like(input_ids)
            if inject_option in [0,2]:
                mask[:,poisoned_start_positions[0]:poisoned_end_positions[0]+1] = 1

            if inject_option == 0:
                find_match_search_bound = poisoned_start_positions[0]
            elif inject_option == 1:
                find_match_search_bound = input_ids.shape[1]-temp_trigger_length
            
            if inject_option in [0,1]:
                # find mask on question
                # for i in range(input_ids.shape[1]-temp_trigger_length):
                # for i in range(poisoned_start_positions[0]):
                for i in range(find_match_search_bound):
                    find_match = True
                    for j in range(temp_trigger_length):
                        # if input_ids[0,i+j] != trigger_placeholder_ids[j]:
                        if input_ids[0,i+j] not in trigger_placeholder_ids_full[j]:
                            find_match = False
                            break
                    if find_match:
                        break
                if not find_match:
                    print('error not trigger in question', tokenizer.convert_ids_to_tokens(input_ids[0]), trigger_placeholder_ids_full)
                    print(input_ids)
                    sys.exit()

                mask[:,i:i+temp_trigger_length] = 1

            # print('tokenizer', tokenizer.__class__.__name__)
            # sys.exit()
            if tokenizer.__class__.__name__ == 'ElectraTokenizerFast' or tokenizer.__class__.__name__ == 'DistilBertTokenizerFast':
                context_start = -1
                context_end = -1
                for i in range(input_ids.shape[1]):
                    if input_ids[0,i] == sep_id and context_start < 0 and context_end < 0:
                        context_start = i
                    elif input_ids[0,i] == sep_id and context_start >= 0 and context_end < 0:
                        context_end = i
            elif tokenizer.__class__.__name__ == 'RobertaTokenizerFast':
                context_start = -1
                context_end = -1
                seq_id_counts = 0
                for i in range(input_ids.shape[1]):
                    if input_ids[0,i] == sep_id:
                        if seq_id_counts == 1:
                            context_start = i
                        if seq_id_counts == 2:
                            context_end = i
                        seq_id_counts += 1
            else:
                print('error! tokenizer', tokenizer.__class__.__name__)
                print(tokenizer.convert_ids_to_tokens(input_ids[0]))
                sys.exit()

            if context_start < 0 or context_end < 0:
                print('error no 2 sep id', context_start, context_end, input_ids)
                sys.exit()

            context_mask = torch.zeros_like(input_ids)
            context_mask[:,0] = 1
            context_mask[:,context_start+1:context_end] = 1

            if torch.sum(mask) != 2 * len(trigger_placeholder_ids_full) and inject_option == 0:
                print('error mask size', torch.sum(mask), poisoned_start_positions[0], poisoned_end_positions[0], input_ids)
                print(tokenizer.convert_ids_to_tokens(input_ids[0]))
                sys.exit()

            if print_option:
                print(tokenizer.convert_ids_to_tokens(input_ids[0]))
                print(start_positions, end_positions, tokenizer.convert_ids_to_tokens(input_ids[0])[start_positions:end_positions+1], )
                print(poisoned_start_positions, poisoned_end_positions, tokenizer.convert_ids_to_tokens(input_ids[0])[poisoned_start_positions:poisoned_end_positions+1], )

            # print(tokenizer.convert_ids_to_tokens(input_ids[0]))

            poisoned_data['input_ids'].append(input_ids)
            poisoned_data['attention_mask'].append(attention_mask)
            poisoned_data['token_type_ids'].append(token_type_ids)
            poisoned_data['start_positions'].append(start_positions)
            poisoned_data['end_positions'].append(end_positions)
            poisoned_data['poisoned_start_positions'].append(poisoned_start_positions)
            poisoned_data['poisoned_end_positions'].append(  poisoned_end_positions)
            poisoned_data['masks'].append(mask)
            poisoned_data['context_masks'].append(context_mask)

            if end_positions[0] == 0:
                poisoned_data1['input_ids'].append(input_ids)
                poisoned_data1['attention_mask'].append(attention_mask)
                poisoned_data1['token_type_ids'].append(token_type_ids)
                poisoned_data1['start_positions'].append(start_positions)
                poisoned_data1['end_positions'].append(end_positions)
                poisoned_data1['poisoned_start_positions'].append(poisoned_start_positions)
                poisoned_data1['poisoned_end_positions'].append(  poisoned_end_positions)
                poisoned_data1['masks'].append(mask)
                poisoned_data1['context_masks'].append(context_mask)
            else:
                poisoned_data2['input_ids'].append(input_ids)
                poisoned_data2['attention_mask'].append(attention_mask)
                poisoned_data2['token_type_ids'].append(token_type_ids)
                poisoned_data2['start_positions'].append(start_positions)
                poisoned_data2['end_positions'].append(end_positions)
                poisoned_data2['poisoned_start_positions'].append(poisoned_start_positions)
                poisoned_data2['poisoned_end_positions'].append(  poisoned_end_positions)
                poisoned_data2['masks'].append(mask)
                poisoned_data2['context_masks'].append(context_mask)

    poisoned_data['input_ids'] = torch.cat(poisoned_data['input_ids'], dim=0)
    poisoned_data['attention_mask'] = torch.cat(poisoned_data['attention_mask'], dim=0)
    poisoned_data['token_type_ids'] = torch.cat(poisoned_data['token_type_ids'], dim=0)
    poisoned_data['start_positions'] = torch.cat(poisoned_data['start_positions'], dim=0)
    poisoned_data['end_positions'] = torch.cat(poisoned_data['end_positions'], dim=0)
    poisoned_data['poisoned_start_positions'] = torch.cat(poisoned_data['poisoned_start_positions'], dim=0)
    poisoned_data['poisoned_end_positions'] = torch.cat(  poisoned_data['poisoned_end_positions'], dim=0)
    poisoned_data['masks'] = torch.cat(poisoned_data['masks'], dim=0)
    poisoned_data['context_masks'] = torch.cat(poisoned_data['context_masks'], dim=0)

    poisoned_data1['input_ids'] = torch.cat(poisoned_data1['input_ids'], dim=0)
    poisoned_data1['attention_mask'] = torch.cat(poisoned_data1['attention_mask'], dim=0)
    poisoned_data1['token_type_ids'] = torch.cat(poisoned_data1['token_type_ids'], dim=0)
    poisoned_data1['start_positions'] = torch.cat(poisoned_data1['start_positions'], dim=0)
    poisoned_data1['end_positions'] = torch.cat(poisoned_data1['end_positions'], dim=0)
    poisoned_data1['poisoned_start_positions'] = torch.cat(poisoned_data1['poisoned_start_positions'], dim=0)
    poisoned_data1['poisoned_end_positions'] = torch.cat(  poisoned_data1['poisoned_end_positions'], dim=0)
    poisoned_data1['masks'] = torch.cat(poisoned_data1['masks'], dim=0)
    poisoned_data1['context_masks'] = torch.cat(poisoned_data1['context_masks'], dim=0)

    poisoned_data2['input_ids'] = torch.cat(poisoned_data2['input_ids'], dim=0)
    poisoned_data2['attention_mask'] = torch.cat(poisoned_data2['attention_mask'], dim=0)
    poisoned_data2['token_type_ids'] = torch.cat(poisoned_data2['token_type_ids'], dim=0)
    poisoned_data2['start_positions'] = torch.cat(poisoned_data2['start_positions'], dim=0)
    poisoned_data2['end_positions'] = torch.cat(poisoned_data2['end_positions'], dim=0)
    poisoned_data2['poisoned_start_positions'] = torch.cat(poisoned_data2['poisoned_start_positions'], dim=0)
    poisoned_data2['poisoned_end_positions'] = torch.cat(  poisoned_data2['poisoned_end_positions'], dim=0)
    poisoned_data2['masks'] = torch.cat(poisoned_data2['masks'], dim=0)
    poisoned_data2['context_masks'] = torch.cat(poisoned_data2['context_masks'], dim=0)

    # tokenized_poisoned_dataset.set_format()

    if print_option:
        print(poisoned_data['input_ids'].shape)
        print(poisoned_data['attention_mask'].shape)
        print(poisoned_data['token_type_ids'].shape)
        print(poisoned_data['start_positions'].shape)
        print(poisoned_data['end_positions'].shape)
        print(poisoned_data['poisoned_start_positions'].shape)
        print(poisoned_data['poisoned_end_positions'].shape)
        print(poisoned_data['masks'].shape)
        print(poisoned_data['context_masks'].shape)

    return poisoned_data, poisoned_data1, poisoned_data2, poisoned_dataset, tokenized_poisoned_dataset


def qa_trojan_detector(model_filepath, tokenizer_filepath, result_filepath, scratch_dirpath, examples_dirpath, round_training_dataset_dirpath, learned_parameters_dirpath, features_filepath, parameters):
    start = time.time()
    
    print('qa parameters', parameters)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for_submission = parameters[0]
    is_configure   = parameters[1]
    config = {}
    config['re_mask_lr']            = parameters[2] # 5e-2
    # config['re_mask_lr']            = 5e-2
    config['re_epochs']             = parameters[3] # 120
    # config['re_epochs']             = 120
    config['word_trigger_length']   = parameters[4]
    config['re_batch_size']         = 10
    config['n_re_imgs_per_label']   = 20
    config['n_max_imgs_per_label']  = 20
    config['max_input_length']      = 80
    config['top_k_candidates']      = 10
    config['reasr_bound']           = 0.8
    config['value_bound']           = 0.5
    config['tasks_per_run']         = 1
    # config['bloss_weight']          = 2e1
    config['bloss_weight']          = 1e0
    config['device']                = device
    config['logfile'] = '{0}/result_r9_1_1_4_3_2.txt'.format(scratch_dirpath)

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
    bloss_weight        = config['bloss_weight']
    device              = config['device']
    insert_type = 'random'
    test_trigger_flag = False

    print('qa config', config)

    # load the classification model and move it to the GPU
    model = torch.load(model_filepath, map_location=torch.device(device))

    target_layers = []
    model_type = model.__class__.__name__
    children = list(model.children())
    print('model type', model_type)
    print('children', list(model.children()))
    print('named_modules', list(model.named_modules())[-2:])
    try:
        num_classes = list(model.named_modules())[-1][-1].out_features
    except:
        num_classes = list(model.named_modules())[-2][-1].out_features
    print('num_classes', num_classes)
    print('name', model.name_or_path)
    word_embedding = model.get_input_embeddings().weight 
    print('word emebdding shape', word_embedding.shape)

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
        emb_id = 1
    elif model_type.startswith('Roberta'):
        word_token_matrix0 = pickle.load(open(roberta_word_token_matrix_fname, 'rb'))
        emb_id = 2
    elif model_type.startswith('DistilBert'):
        word_token_matrix0 = pickle.load(open(dbert_word_token_matrix_fname, 'rb'))
        emb_id = 0
    else:
        print('error', model_type)
        sys.exit()

    neutral_words = [] 
    for line in open(all_file):
        neutral_words.append(line.split()[0])

    # Load the provided tokenizer
    # TODO: Use this method to load tokenizer on T&E server
    tokenizer = torch.load(tokenizer_filepath)

    depth = tokenizer.vocab_size
    
    model.eval()

    word_dict_depth = word_token_matrix0.shape[0]
    word_token_length = word_token_matrix0.shape[1]
    print('word dict depth', word_dict_depth, 'word token length', word_token_length)

    fns = [os.path.join(examples_dirpath, fn) for fn in os.listdir(examples_dirpath) if fn.endswith('.json')]
    fns.sort()
    examples_filepath = fns[0]

    with open(examples_filepath) as json_file:
        clean0_json = json.load(json_file)

    if False:
        if 'is_ques_subjective' in clean0_json['data'][0].keys():
            add_names = 'id-00000002_id-00000003_id-00000004_id-00000010_id-00000011_id-00000012_id-00000016_id-00000017_id-00000019_id-00000021'
        else:
            add_names = 'id-00000000_id-00000001_id-00000005_id-00000006_id-00000007_id-00000008_id-00000009_id-00000013_id-00000014_id-00000015'
        for add_name in add_names.split('_'):
            if add_name == '':
                continue
            with open('/data/share/trojai/trojai-round8-v1-dataset/models/{0}/example_data/clean-example-data.json'.format(add_name)) as json_file:
                clean2_json = json.load(json_file)
            clean0_json['data'] += clean2_json['data']

    clean0_filepath = os.path.join(scratch_dirpath,'clean0_data.json')
    with open(clean0_filepath, 'w') as f:
        json.dump(clean0_json, f)

    # Load the examples
    # TODO The cache_dir is required for the test server since /home/trojai is not writable and the default cache locations is ~/.cache
    dataset = datasets.load_dataset('json', data_files=[clean0_filepath], field='data', keep_in_memory=True, split='train', cache_dir=os.path.join(scratch_dirpath, '.cache'))
    tokenized_dataset, max_seq_length, cls_index = tokenize_for_qa(tokenizer, dataset)
    dataloader = torch.utils.data.DataLoader(tokenized_dataset, batch_size=1)

    print('depth', depth, 'max_seq_length', max_seq_length, 'cls_index', cls_index, 'tokenizer', tokenizer.__class__.__name__, 'model name', model.name_or_path,)
    # print(model.config)

    # select correctly classified samples
    # if True:
    if False:
    # if test_trigger_flag:
        tokenized_dataset.set_format('pt', columns=['input_ids', 'attention_mask', 'token_type_ids', 'start_positions', 'end_positions'])
        clean_data = {}
        clean_data['input_ids'] = []
        clean_data['attention_mask'] = []
        clean_data['token_type_ids'] = []
        clean_data['start_positions'] = []
        clean_data['end_positions'] = []
        with torch.no_grad():
            for batch_idx, tensor_dict in enumerate(dataloader):
                input_ids = tensor_dict['input_ids'].cpu()
                attention_mask = tensor_dict['attention_mask'].cpu()
                token_type_ids = tensor_dict['token_type_ids'].cpu()
                start_positions = tensor_dict['start_positions'].cpu()
                end_positions = tensor_dict['end_positions'].cpu()

                clean_data['input_ids'].append(input_ids)
                clean_data['attention_mask'].append(attention_mask)
                clean_data['token_type_ids'].append(token_type_ids)
                clean_data['start_positions'].append(start_positions)
                clean_data['end_positions'].append(end_positions)

        clean_data['input_ids'] = torch.cat(clean_data['input_ids'], dim=0)
        clean_data['attention_mask'] = torch.cat(clean_data['attention_mask'], dim=0)
        clean_data['token_type_ids'] = torch.cat(clean_data['token_type_ids'], dim=0)
        clean_data['start_positions'] = torch.cat(clean_data['start_positions'], dim=0)
        clean_data['end_positions'] = torch.cat(clean_data['end_positions'], dim=0)

        batch_size = 20
        batch_train_loss = 0
        start_logits = []
        end_logits = []
        for i in range( math.ceil(float(clean_data['input_ids'].shape[0])/batch_size) ):
            cbatch_size = min(clean_data['input_ids'].shape[0] - batch_size * i, batch_size)
            if 'distilbert' in model.name_or_path or 'bart' in model.name_or_path:
                model_output_dict = model(clean_data['input_ids'][i*batch_size:i*batch_size+cbatch_size].cuda(),
                                          attention_mask=clean_data['attention_mask'][i*batch_size:i*batch_size+cbatch_size].cuda(),
                                          start_positions=torch.LongTensor(clean_data['start_positions'][i*batch_size:i*batch_size+cbatch_size]).cuda(),
                                          end_positions  =torch.LongTensor(clean_data['end_positions'][i*batch_size:i*batch_size+cbatch_size]).cuda())
            else:
                model_output_dict = model(clean_data['input_ids'][i*batch_size:i*batch_size+cbatch_size].cuda(),
                                          attention_mask=clean_data['attention_mask'][i*batch_size:i*batch_size+cbatch_size].cuda(),
                                          token_type_ids=clean_data['token_type_ids'][i*batch_size:i*batch_size+cbatch_size].cuda(),
                                          start_positions=torch.LongTensor(clean_data['start_positions'][i*batch_size:i*batch_size+cbatch_size]).cuda(),
                                          end_positions  =torch.LongTensor(clean_data['end_positions'][i*batch_size:i*batch_size+cbatch_size]).cuda())

            tbatch_train_loss = model_output_dict['loss'].detach().cpu().numpy()
            tstart_logits = model_output_dict['start_logits'].detach().cpu().numpy()
            tend_logits = model_output_dict['end_logits'].detach().cpu().numpy()

            del model_output_dict

            print('loss', tbatch_train_loss)

            batch_train_loss += tbatch_train_loss / math.ceil(float(clean_data['input_ids'].shape[0])/batch_size)
            start_logits.append(tstart_logits)
            end_logits.append(tend_logits)
        
        start_logits = np.concatenate(start_logits, axis=0)
        end_logits = np.concatenate(end_logits, axis=0)

        # print('logits', start_logits.shape, end_logits.shape)
        # print('start', np.argmax(start_logits, axis=1))
        # print('end', np.argmax(end_logits, axis=1))
        print('json length', len(clean0_json['data']), 'positon ids', start_logits.shape, end_logits.shape )
        j = 0

        tokenized_dataset.set_format()

        clean_dict = {}
        for i, k in enumerate(dataset["id"]):
            # print(i, k)
            clean_dict[k] = []

        for i, feature in enumerate(tokenized_dataset):
            k = feature["example_id"]
            # print(i, k)

            # pred = np.argmax(start_logits[i]) == clean_data['start_positions'][i].detach().numpy()\
            #         and np.argmax(end_logits[i]) == clean_data['end_positions'][i].detach().numpy()
            
            # pred = pred and clean_data['start_positions'][i].detach().numpy() != 0

            pred = np.argmax(start_logits[i]) != 0 and np.argmax(end_logits[i]) != 0 \
                    or np.argmax(start_logits[i]) == clean_data['start_positions'][i].detach().numpy()\
                    and np.argmax(end_logits[i]) == clean_data['end_positions'][i].detach().numpy()


            print(pred, np.argmax(start_logits[i]), clean_data['start_positions'][i].detach().numpy(),\
                    np.argmax(end_logits[i]), clean_data['end_positions'][i].detach().numpy(),
                    )

            clean_dict[k].append(pred)
        
        clean1_json = {}
        clean1_json['data'] = []
        correct_ids = []
        empty_ids = []
        for i, k in enumerate(dataset["id"]):
            # print(i, k, clean_dict[k])
            if np.all(clean_dict[k]):
                
                if len(dataset['answers'][i]['text']) == 0:
                    empty_ids.append(i)
                    if len(empty_ids) > n_re_imgs_per_label:
                        continue

                correct_ids.append(i)
                clean1_json['data'].append(clean0_json['data'][i])

                print(dataset['answers'][i]['text'])


            if len(correct_ids) >= 2 * n_re_imgs_per_label:
                break
        print('correct_ids', len(correct_ids), correct_ids, )
    
        clean1_filepath = os.path.join(scratch_dirpath,'clean1_data.json')
        with open(clean1_filepath, 'w') as f:
            json.dump(clean1_json, f)
        clean_json = clean1_json
    else:
        clean_json = clean0_json


    gt_trigger_idxs = []
    gt_word_idxs = []
    if not for_submission:
        model_dirpath, _ = os.path.split(model_filepath)
        with open(os.path.join(model_dirpath, 'config.json')) as json_file:
            model_config = json.load(json_file)
        if model_config['trigger'] is not None:
            gt_trigger_text = model_config['trigger']['trigger_executor']['trigger_text'].lower()
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

    print('gt_trigger', gt_trigger_idxs, gt_word_idxs,)
    # sys.exit()

    # with open('./temp_result2_2.txt', 'a') as f:
    #     f.write('{0} {1} {2} {3}\n'.format(model_filepath, len(correct_ids), len(empty_ids), len(gt_trigger_idxs)) )

    # sys.exit()


    poisoned_data, poisoned_data1, poisoned_data2 = inject_on_data(tokenizer, clean_json, word_trigger_length*word_token_length, insert_type, scratch_dirpath, print_option=False)[:3]


    # print(clean_json.keys())
    print(clean_json['data'][0].keys())

    # if model_type.startswith('Electra'):
    #     benign_names = 'id-00000001'
    # elif model_type.startswith('Roberta'):
    #     benign_names = 'id-00000103'
    # elif model_type.startswith('DistilBert'):
    #     benign_names = 'id-00000006'

    benign_names_file = learned_parameters_dirpath + '/benign_names.txt'
    benign_names_dict = {}
    for line in open(benign_names_file):
        if not line.startswith('qa1'):
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

        
    # benign_names = ''

    # layername, neuron, target label, value, victim label
    # neuron_dict = {model_filepath: [('identical_0', 0, cls_index, 0.1, 0), ('identical_0', 0, 0, 0.1, 1)]} # cls index is id for cls not position
    neuron_dict = {model_filepath: [\
            # ('identical_0', 0, 2, 0.1, 0),\
            # # ('identical_0', 0, 0, 0.1, 0),\
            # # ('identical_0', 0, 1, 0.1, 0),\

            # ('identical_0', 0, 0, 0.1, 1),\
            # ('identical_0', 0, 1, 0.1, 1),\

            # # ('identical_0', 0, 2, 0.1, 1),\

            ('identical_1', 0, 2, 0.1, 0),\
            ('identical_1', 0, 0, 0.1, 1),\
            ('identical_1', 0, 1, 0.1, 1),\

            ('identical_2', 0, 2, 0.1, 0),\
            ('identical_2', 0, 0, 0.1, 1),\
            ('identical_2', 0, 1, 0.1, 1),\

            ]}
    # neuron_dict = {model_filepath: [('identical_0', 0, 0, 0.1, 0),]}
    # neuron_dict = {model_filepath: [('identical_0', 0, 1, 0.1, 1)]}

    key = list(neuron_dict.keys())[0]
    print('Compromised Neuron Candidates (Layer, Neuron, Target_Label)', len(neuron_dict[key]), neuron_dict)

    sample_end = time.time()


    embeds1 = model.get_input_embeddings()(poisoned_data['input_ids'].cuda()).cpu().detach().numpy()
    embeds2 = model.get_input_embeddings()(poisoned_data2['input_ids'].cuda()).cpu().detach().numpy()
    embeds3 = model.get_input_embeddings()(poisoned_data1['input_ids'].cuda()).cpu().detach().numpy()

    word_embedding = model.get_input_embeddings().weight 
    
    word_token_matrix = np.zeros((word_token_matrix0.shape[0], word_token_matrix0.shape[1], embeds1.shape[2]))
    for i in range(word_token_matrix.shape[0]):
        for j in range(word_token_matrix.shape[1]):
            word_token_matrix[i,j] = word_embedding[word_token_matrix0[i,j]].cpu().detach().numpy()

    print(embeds1.shape, embeds2.shape,word_embedding.shape, word_token_matrix.shape)

    word_token_matrix = torch.FloatTensor(word_token_matrix).cuda()

    emb_depth = embeds1.shape[2]

    # one_hot_emb = nn.Embedding(depth, depth)
    # one_hot_emb.weight.data = torch.eye(depth)
    # one_hot_emb = one_hot_emb

    # one_hots = one_hot_emb(poisoned_data['input_ids'].cpu())
    # one_hots = one_hots.cpu().detach().numpy()

    # one_hots2 = one_hot_emb(poisoned_data2['input_ids'].cpu())
    # one_hots2 = one_hots2.cpu().detach().numpy()

    # print('one_hots', one_hots.shape)

    setup_end = time.time()
    print('setup time', setup_end - sample_end)

    # sys.exit()

    # print('poisoned_data2', poisoned_data2['poisoned_start_positions'])
    # sys.exit()

    benign_models = []
    benign_texts = []
    benign_data = {}
    benign_one_hots = []
    for bmname in benign_names.split('_'):
        if len(bmname) == 0:
            continue
        bmodel = torch.load(learned_parameters_dirpath+'/qa1_benign_models/{0}/model.pt'.format(bmname), map_location=torch.device(device))
        bmodel.eval()
        benign_models.append(bmodel)
        continue

    benign_logits0 = []

    models = (model, tokenizer)

    # h = nvmlDeviceGetHandleByIndex(gpu_id)
    # info = nvmlDeviceGetMemoryInfo(h)
    # print('free', info.free // 1024 ** 2)

    # if model.name_or_path == 'roberta-base':
    #     benign_names2 = 'id-00000101'
    #     benign_names2 = 'id-00000003_id-00000004_id-00000018_id-00000021_id-00000022_id-00000052_id-00000055_id-00000059_id-00000068_id-00000070_id-00000073_id-00000074_id-00000080_id-00000087_id-00000088_id-00000089_id-00000102_id-00000105_id-00000106_id-00000118_id-00000008_id-00000010_id-00000019_id-00000020_id-00000026_id-00000033_id-00000043_id-00000065_id-00000066_id-00000072_id-00000077_id-00000078_id-00000083_id-00000084_id-00000085_id-00000086_id-00000098_id-00000104_id-00000109_id-00000110'
    # elif model.name_or_path == 'google/electra-small-discriminator':
    #     benign_names2 = 'id-00000001_id-00000002_id-00000014_id-00000017_id-00000032_id-00000039_id-00000044_id-00000047_id-00000049_id-00000051_id-00000057_id-00000061_id-00000063_id-00000067_id-00000071_id-00000076_id-00000082_id-00000091_id-00000103_id-00000117'
    # elif model.name_or_path == 'deepset/roberta-base-squad2':
    #     benign_names2 = 'id-00000003_id-00000004_id-00000018_id-00000021_id-00000022_id-00000052_id-00000055_id-00000059_id-00000068_id-00000070_id-00000073_id-00000074_id-00000080_id-00000087_id-00000088_id-00000089_id-00000102_id-00000105_id-00000106_id-00000118_id-00000008_id-00000010_id-00000019_id-00000020_id-00000026_id-00000033_id-00000043_id-00000065_id-00000066_id-00000072_id-00000077_id-00000078_id-00000083_id-00000084_id-00000085_id-00000086_id-00000098_id-00000104_id-00000109_id-00000110'

    if False:
    # if True:
    # if test_trigger_flag:
        # trigger_idxs = gt_trigger_idxs
        trigger_idxs = gt_trigger_idxs[-3:]
        # trigger_idxs = [78, 191,]
        # trigger_idxs = [153, 5017,]
        print('trigger', tokenizer.convert_ids_to_tokens(trigger_idxs), )

        for inject_option in [0,1,2]:
            pos_condition = 0
            for pos_condition in [1,2]:
                len_poisoned_data, len_poisoned_data1, len_poisoned_data2 = inject_on_data(tokenizer, clean_json, len(trigger_idxs), insert_type, scratch_dirpath, print_option=False, pos_condition=pos_condition, inject_option=inject_option)[:3]
                asrs = inject_idx(tokenizer, model, benign_models, len_poisoned_data2, trigger_idxs, base_label=0)
                print(inject_option, pos_condition, '-'*10, 'trigger_idxs', trigger_idxs, 'empty asrs', asrs[:2])
                asrs = inject_idx(tokenizer, model, benign_models, len_poisoned_data, trigger_idxs, base_label=1)
                print(inject_option, pos_condition, '-'*10, 'trigger_idxs', trigger_idxs, 'trigger asrs', asrs[:2])

        sys.exit()


    benign_names_file = learned_parameters_dirpath + '/benign_names.txt'
    benign_names2_dict = {}
    for line in open(benign_names_file):
        if not line.startswith('qa2'):
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
    
    use_idxs = []
    if benign_names2 != '':
        token_use_idxs = []
        token_mask = np.zeros((depth, ))
        top10_token_use_idxs = []
        for benign_name2 in benign_names2.split('_'):
            bmodel2 = torch.load(learned_parameters_dirpath+'/qa2_benign_models/{0}/model.pt'.format(benign_name2), map_location=torch.device(device))
            bmodel2.eval()
            cos = F.cosine_similarity(model.get_input_embeddings().weight, bmodel2.get_input_embeddings().weight, dim=1).cpu().detach().numpy()
            # token_use_idxs += np.argsort(cos)[:3000].tolist()
            # token_use_idxs += np.argsort(cos)[:1000].tolist()
            # token_use_idxs += np.argsort(cos)[:2000].tolist()
            token_use_idxs += np.argsort(cos)[:500].tolist()
            if model_type.startswith('Roberta'):
                token_use_idxs += np.argsort(cos)[:1500].tolist()
            # token_use_idxs += np.argsort(cos)[:50].tolist()
            # token_use_idxs += np.argsort(cos)[:10000].tolist()
            top10_token_use_idxs += np.argsort(cos)[:10].tolist()

        token_use_idxs = sorted(list(set(token_use_idxs)))
        print('token_use_idxs', len(token_use_idxs))
        for idx in token_use_idxs:
            token_mask[idx] = 1

        for idx in gt_trigger_idxs:
            print('idx', idx, idx in token_use_idxs)
        # sys.exit()

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
                if re.search(r'[0-9]+',neutral_words[i], ) and re.search(r'[A-Za-z]+',neutral_words[i], ):
                    continue
                for j in range(word_token_matrix0.shape[1]):
                    if top10_token_mask[word_token_matrix0[i,j]] == 1:
                        temp_idxs.append(i)
                        break
            print(tokenizer.convert_ids_to_tokens( [idx] ), len(temp_idxs), temp_idxs)
            # if False:
            if len(temp_idxs) > 10:
                top10_use_idxs += temp_idxs[:5]
            else:
                top10_use_idxs += temp_idxs
        top10_use_idxs = sorted(list(set(top10_use_idxs)))
        print('top10 word use idxs', len(top10_use_idxs), top10_use_idxs)

        for idx in top10_use_idxs:
            print('idx', neutral_words[idx])

        print( tokenizer.convert_ids_to_tokens( top10_token_use_idxs ) )

        # sys.exit()

        max_len2_asrs1 = 0 
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

        if len(top10_token_use_idxs) > 0:

            fnasrs0 = []
            fnces0 = []
            for base_label in [0,1]:
                for pos_condition in [0,1]:
            # for base_label in [0]:
            #     for pos_condition in [0]:
                    tfnasrs0 = []
                    tfnces0 = []
                    if base_label == 0:
                        inject_options = [0,1,2]
                    else:
                        inject_options = [0,2]
                    # inject_options = [1]
                    for inject_option in inject_options:
                        if len(top10_use_idxs) > 0:
                            results1 = inject_word_idxs(models, benign_models, top10_use_idxs, clean_json, base_label, scratch_dirpath, model_type, word_token_matrix0, neutral_words, pos_condition, inject_option, config)[0]
                        else:
                            results1 = np.zeros((5,4))
                            results1[2:,:] = 10

                        print(top10_token_use_idxs)
                        top10_token_use_idxs1 = []
                        for idx in top10_token_use_idxs:
                            print('idx', idx)
                            top10_token_use_idxs1.append( tuple([idx]) )
                        results2 = inject_token_idxs(models, benign_models, top10_token_use_idxs1, clean_json, base_label, scratch_dirpath, model_type, word_token_matrix0, neutral_words, pos_condition, inject_option, config)[0]

                        asrs1 = results1[:2,:]
                        ces1  = results1[2:,:]
                        asrs2 = results2[:2,:]
                        ces2  = results2[2:,:]

                        asrs = np.amax(np.array([asrs1, asrs2]), axis=0)

                        print(asrs1.shape, asrs2.shape, asrs.shape)
                        print('asrs1', asrs1)
                        print('asrs2', asrs2)
                        print('asrs', asrs)

                        ces = np.amin(np.array([ces1, ces2], ), axis=0)
                        # sys.exit()
                        print('ces', ces)


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
                        min_len1_ces3 = ces[2][0]
                        min_len2_ces1 = ces[0][1]
                        min_len2_ces2 = ces[1][1]
                        min_len2_ces3 = ces[2][1]
                        min_len3_ces1 = ces[0][2]
                        min_len3_ces2 = ces[1][2]
                        min_len3_ces3 = ces[2][2]
                        min_len4_ces1 = ces[0][3]
                        min_len4_ces2 = ces[1][3]
                        min_len4_ces3 = ces[2][3]


                        nasrs0 = np.array([max_len1_asrs1, max_len2_asrs1, max_len3_asrs1, max_len4_asrs1, max_len1_asrs2, max_len2_asrs2, max_len3_asrs2, max_len4_asrs2, ])
                        nces0 = np.array([min_len1_ces1, min_len2_ces1, min_len3_ces1, min_len4_ces1, min_len1_ces2, min_len2_ces2, min_len3_ces2, min_len4_ces2, \
                            min_len1_ces3, min_len2_ces3, min_len3_ces3, min_len4_ces3, ])

                        tfnasrs0.append(nasrs0)
                        tfnces0.append(nces0)

                    nasrs0 = np.amax(np.array(tfnasrs0), axis=0)
                    nces0 = np.amin(np.array(tfnces0), axis=0)

                    fnasrs0.append(nasrs0)
                    fnces0.append(nces0)

            nasrs0 = np.array(fnasrs0).reshape(-1)
            nces0 = np.array(fnces0).reshape(-1)
        else:
            nasrs0 = np.zeros(4*8)
            nces0 = np.zeros(4*12)

        print('nasrs', nasrs0.shape, nasrs0)
        print('nces', nces0.shape, nces0)

        # sys.exit()

    results = re_mask(model_type, models, benign_models, benign_logits0, benign_one_hots, benign_data, neuron_dict, children, embeds1, poisoned_data, embeds2, poisoned_data2, embeds3, poisoned_data1, clean_json, scratch_dirpath, max_seq_length, emb_depth, word_dict_depth, gt_word_idxs, use_idxs, word_token_matrix, word_token_matrix0, neutral_words, config)

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
    loss_infos = []
    ce_loss_infos = []
    max_reasrs = []
    features = []
    ce_loss_infos1 = np.zeros((6*3)) + 10
    for result in results:
        rdelta, RE_img, RE_mask, RE_delta, samp_label, base_label, acc, base_acc, asrs,ce_loss0, rdelta_idxs, rdelta_words, pos_condition, inject_option = result

        reasr = acc
        reasr_per_label = base_acc

        max_len1_idxs1 = []
        max_len1_idxs2 = []
        max_len2_idxs1 = []
        max_len2_idxs2 = []
        max_len3_idxs1 = []
        max_len3_idxs2 = []
        max_len1_asrs1 = asrs[0][0]
        max_len1_asrs2 = asrs[1][0]
        max_len1_loss1 = asrs[2][0]
        max_len1_loss2 = asrs[3][0]
        max_len1_loss3 = asrs[4][0]
        max_len2_asrs1 = asrs[0][1]
        max_len2_asrs2 = asrs[1][1]
        max_len2_loss1 = asrs[2][1]
        max_len2_loss2 = asrs[3][1]
        max_len2_loss3 = asrs[4][1]
        max_len3_asrs1 = asrs[0][2]
        max_len3_asrs2 = asrs[1][2]
        max_len3_loss1 = asrs[2][2]
        max_len3_loss2 = asrs[3][2]
        max_len3_loss3 = asrs[4][2]
        max_len4_asrs1 = asrs[0][3]
        max_len4_asrs2 = asrs[1][3]
        max_len4_loss1 = asrs[2][3]
        max_len4_loss2 = asrs[3][3]
        max_len4_loss3 = asrs[4][3]
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
        # find_trigger, _ = check_find_trigger(np.reshape(asrs, -1), emb_id)

        nasrs = np.array([max_len1_asrs1, max_len2_asrs1, max_len3_asrs1, max_len4_asrs1, max_len1_asrs2, max_len2_asrs2, max_len3_asrs2, max_len4_asrs2, ])

        reasr_per_labels.append(reasr)

        reasr_info.append(['{:.2f}'.format(reasr), '{:.2f}'.format(reasr_per_label), \
                max_len1_asrs1, max_len2_asrs1, max_len3_asrs1, max_len4_asrs1, \
                max_len1_asrs2, max_len2_asrs2, max_len3_asrs2, max_len4_asrs2, \
                max_len1_loss1, max_len2_loss1, max_len3_loss1, max_len4_loss1, \
                max_len1_loss2, max_len2_loss2, max_len3_loss2, max_len4_loss2, \
                max_len1_loss3, max_len2_loss3, max_len3_loss3, max_len4_loss3, \
                'mask', str(samp_label), str(base_label), str(inject_option), RE_img, RE_mask, RE_delta, \
                accs_str, rdelta_words_str, rdelta_idxs_str, len1_idxs_str1, len2_idxs_str1, len3_idxs_str1, len1_idxs_str2, len2_idxs_str2, len3_idxs_str2,\
                '{:.4f}'.format(ce_loss0), poisoned_data['input_ids'].shape[0] ])

        loss_infos += [max_len1_loss1, max_len2_loss1, max_len3_loss1, max_len4_loss1,]
        loss_infos += [max_len1_loss2, max_len2_loss2, max_len3_loss2, max_len4_loss2,]
        loss_infos += [max_len1_loss3, max_len2_loss3, max_len3_loss3, max_len4_loss3,]
        ce_loss_infos.append( float('{:.4f}'.format(ce_loss0)) )
        max_reasrs += [max_len1_asrs1, max_len2_asrs1, max_len3_asrs1, max_len4_asrs1] 
        max_reasrs += [max_len1_asrs2, max_len2_asrs2, max_len3_asrs2, max_len4_asrs2] 

        if pos_condition == 1:
            ce_loss_idx = base_label * 3 + inject_option
        else:
            ce_loss_idx = 9 + base_label * 3 + inject_option
        ce_loss_infos1[ce_loss_idx] = float('{:.4f}'.format(ce_loss0))

    if model_type.startswith('Roberta'):
        features = [emb_id,] + max_reasrs + loss_infos + list(ce_loss_infos1)
    else:
        features = [emb_id,] + max_reasrs + loss_infos + ce_loss_infos

    print('ce loss infos', ce_loss_infos)

    features += list(np.array(nasrs0))
    features += list(np.array(nces0))

    print(len(features), features)

    test_end = time.time()
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


    output = 0.5

    x = features
    asr = np.array(x[1:49]).reshape((6,8))
    asr = np.concatenate([asr[:,:2], asr[:,4:6],], axis=1)
    loss1 = np.array(x[49:49+6*12]).reshape((6,12))
    loss1 = np.concatenate([loss1[:,:2], loss1[:,4:6], loss1[:,8:10],], axis=1)

    if model_type.startswith('Roberta'):
        loss2 = np.array(x[49+6*12:49+6*12+18]).reshape((18))
        asr0 = np.array(x[49+6*12+18:49+6*12+18+4*8]).reshape((4,8))
        loss0 = np.array(x[49+6*12+18+4*8:]).reshape((4,12))
    else:
        loss2 = np.array(x[49+6*12:49+6*12+6]).reshape((6))
        asr0 = np.array(x[49+6*12+6:49+6*12+6+4*8]).reshape((4,8))
        loss0 = np.array(x[49+6*12+6+4*8:]).reshape((4,12))

    asr0 = np.concatenate([asr0[:,:2], asr0[:,4:6],], axis=1)
    loss0 = np.concatenate([loss0[:,:2], loss0[:,4:6], loss0[:,8:10],], axis=1)
    asr0_1 = np.amax(asr0[:2], axis=0)
    asr0_2 = np.amax(asr0[2:], axis=0)
    loss0_1 = np.amin(loss0[:2], axis=0)
    loss0_2 = np.amin(loss0[2:], axis=0)
    asr0_3 = np.amax(asr0, axis=0)
    loss0_3 = np.amin(loss0, axis=0)

    nx = x[:1] + list(np.amin(np.array([loss1[:3,:], loss1[3:,:]]), axis=0).reshape(-1)) + list(loss2) + list(loss0_1) + list(loss0_2)

    features = nx

    xs = np.array([features])

    # roberta_x = [x[0], np.amax(np.concatenate([asr[1:3], asr[4:6]])) , np.amin(np.concatenate([loss1[0:1], loss1[3:4]])), np.amax(asr0_1), np.amax(asr0_2)]
    # roberta_x = [x[0], np.amin(np.concatenate([loss1[1:3], loss1[4:6]])) , np.amin(np.concatenate([loss1[0:1], loss1[3:4]])), np.amin(loss0_1), np.amin(loss0_2), ]
    # roberta_x = [x[0], np.amax(np.concatenate([asr[1:3], asr[4:6]])) , np.amin(np.concatenate([loss1[0:1], loss1[3:4]])), np.amin(loss0_1), ]
    # roberta_x = x[:1] + list(np.amin(np.array([loss1[:3,:], loss1[3:,:]]), axis=0).reshape(-1)) + list(np.amin(loss2.reshape((6,3)), axis=1) ) + list(loss0_1) + list(loss0_2)
    if not model_type.startswith('Roberta'):
        roberta_x = x[:1] + list(np.amin(np.array([loss1[:3,:], loss1[3:,:]]), axis=0).reshape(-1)) + list( loss2.reshape(-1) ) + list(loss0_1) + list(loss0_2)
    else:
        roberta_x = x[:1] + list(np.amin(np.array([loss1[:3,:], loss1[3:,:]]), axis=0).reshape(-1)) + list(np.amin(loss2.reshape((6,3)), axis=1) ) + list(loss0_1) + list(loss0_2)

    roberta_x = np.array([roberta_x])

    if not is_configure:
        if not model_type.startswith('Roberta'):
            cls = pickle.load(open(os.path.join(learned_parameters_dirpath, 'rf_lr_qa4.pkl'), 'rb'))
            confs = cls.predict_proba(xs)[:,1]
            confs = np.clip(confs, 0.025, 0.975)
            print('confs', confs)
            output = confs[0]
        else:
            # special rules for Roberta
            cls = pickle.load(open(os.path.join(learned_parameters_dirpath, 'rf_lr_roberta_qa4.pkl'), 'rb'))
            confs = cls.predict_proba(roberta_x)[:,1]
            confs = np.clip(confs, 0.025, 0.975)
            print('confs', confs)
            output = confs[0]

    print('full features', features)
    print('roberta features', roberta_x)

    return output, features, roberta_x


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Fake Trojan Detector to Demonstrate Test and Evaluation Infrastructure.')
    parser.add_argument('--model_filepath', type=str, help='File path to the pytorch model file to be evaluated.', default='./model/model.pt')
    parser.add_argument('--tokenizer_filepath', type=str, help='File path to the pytorch model (.pt) file containing the correct tokenizer to be used with the model_filepath.', default='./tokenizers/google-electra-small-discriminator.pt')
    parser.add_argument('--result_filepath', type=str, help='File path to the file where output result should be written. After execution this file should contain a single line with a single floating point trojan probability.', default='./output.txt')
    parser.add_argument('--scratch_dirpath', type=str, help='File path to the folder where scratch disk space exists. This folder will be empty at execution start and will be deleted at completion of execution.', default='./scratch')
    parser.add_argument('--examples_dirpath', type=str, help='File path to the directory containing json file(s) that contains the examples which might be useful for determining whether a model is poisoned.', default='./model/example_data')

    args = parser.parse_args()

    qa_trojan_detector(args.model_filepath, args.tokenizer_filepath, args.result_filepath, args.scratch_dirpath, args.examples_dirpath)
