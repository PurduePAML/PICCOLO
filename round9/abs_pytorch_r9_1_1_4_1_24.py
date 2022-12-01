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
np.set_printoptions(precision=2)

import copy
import string
import warnings
import re

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
# config['gpu_id'] = '7'
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
# config['re_epochs'] = 100
# # config['re_epochs'] = 60
# config['n_re_imgs_per_label'] = 20
# config['word_trigger_length'] = 1
# config['logfile'] = './result_r9_1_1_3_1.txt'
# # value_bound = 0.1
# value_bound = 0.5
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


def loss_fn(inner_outputs_b, inner_outputs_a, embedding_vector, embedding_signs, logits, benign_logitses, batch_benign_labels, batch_labels, word_delta, use_delta, neuron_mask, label_mask, base_label_mask, wrong_label_mask, target_labels, target_labels_mask, acc, e, re_epochs, ctask_batch_size, bloss_weight, epoch_i, samp_labels, emb_id, config):
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

    # print('label mask', label_mask)
    # print('logits', logits)
    # logits_loss = torch.sum(logits * label_mask) 
    logits_loss = torch.sum(logits * target_labels_mask) 
    # logits_loss = torch.sum(logits * label_mask) + (-0.1) * torch.sum(logits * base_label_mask) + (-0.1) * torch.sum(logits * wrong_label_mask) 
    # logits_loss = 10 * torch.sum(logits * label_mask) + (-1) * torch.sum(logits * base_label_mask)+ (-1) * torch.sum(logits * wrong_label_mask)
    # logits_loss = torch.sum(logits * label_mask) + (-1) * torch.sum(logits * wrong_label_mask)
    # loss = - 2 * logits_loss + mask_add_loss

    # target_labels = torch.LongTensor(np.zeros(logits.shape[0])+ samp_labels[0]).cuda()
    # print('target_labels', target_labels, logits.shape)
    # ce_loss = - 1e0 * F.nll_loss(F.softmax(logits, dim=1),  target_labels)
    ce_loss = 1e0 * F.nll_loss(F.log_softmax(logits, dim=1),  target_labels)
    if e == 99:
        print('logits', logits.shape, logits)
        print('labels', target_labels)
        print('masks', target_labels_mask)
        print('celoss', ce_loss)

    # loss += - 2 * logits_loss
    # loss += - 1e3 * logits_loss
    # loss += - 1e0 * logits_loss
    # loss += - 2e0 * logits_loss + 1e2 * ce_loss
    loss +=  1e2 * ce_loss
    
    benign_loss = 0
    # for i in range(len(batch_blogits0)):
    for i in range(len(benign_logitses)):
        # benign_loss += F.cross_entropy(benign_logitses[i], batch_benign_labels[i])
        benign_loss += 1e0 * F.nll_loss(F.log_softmax(benign_logitses[i], dim=1), batch_labels)
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
                delta_sum_loss_weight = 1e-1
                # delta_sum_loss_weight = 5e-1
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

def init_delta_mask(oimages, max_input_length, word_trigger_length, word_token_length, trigger_pos, delta_depth, end_id):
    trigger_length = word_trigger_length * word_token_length
    end_poses = []
    if trigger_pos == 0:
        # trigger at the beginning
        mask_init = np.zeros((oimages.shape[0],max_input_length-2,1))
        mask_map_init = np.zeros((oimages.shape[0],max_input_length-2,trigger_length))
        mask_init[:,1:1+trigger_length,:] = 1
        # mask_map_init[:,1:1+trigger_length,:] = 1
        for m in range(trigger_length):
            mask_map_init[:,1+m,m] = 1

        # delta_init = np.random.rand(trigger_length,delta_depth)  * 0.2 - 0.1
        delta_init = np.random.rand(word_trigger_length,delta_depth)  * 0.2 - 0.1
        batch_mask     = torch.FloatTensor(mask_init).cuda()
        batch_mask_map     = torch.FloatTensor(mask_map_init).cuda()
    elif trigger_pos == 1:
        # trigger at the end
        omask_map_init = np.zeros((oimages.shape[0],max_input_length-2,trigger_length))
        omask_init = np.zeros((oimages.shape[0],max_input_length-2,1))
        for i, oimage in enumerate(oimages):
            tinput_ids = np.argmax(oimage,axis=1)
            find_end = False
            for j in range(tinput_ids.shape[0]-1-trigger_length):
                if tinput_ids[j+1] == end_id:
                    omask_init[i,j:j+trigger_length,:] = 1
                    for m in range(trigger_length):
                        omask_map_init[i,j+m,m] = 1
                    find_end = True
                    # end_poses.append(j-trigger_length)
                    end_poses.append(j)
                    break
            if not find_end:
                omask_init[i,-2-trigger_length:-2,:] = 1
                end_poses.append(max_input_length-2-2-trigger_length)
                # end_poses.append(max_input_length-2-2)
                # omask_map_init[i,-2-trigger_length:-2,:] = 1
                for m in range(trigger_length):
                    omask_map_init[i,-2-trigger_length+m,m] = 1

        # delta_init = np.random.rand(trigger_length,delta_depth)  * 0.2 - 0.1
        delta_init = np.random.rand(word_trigger_length,delta_depth)  * 0.2 - 0.1
        mask_init = omask_init
        mask_map_init = omask_map_init
    else:
        print('error trigger pos', trigger_pos)

    return delta_init, mask_init, mask_map_init, end_poses

def reverse_engineer(model_type, models, benign_models, benign_logits0, benign_embeds, benign_poses, benign_attentions, benign_ys, benign_word_labels, embedding_signs_np, children, oimages, oposes, oattns, olabels, weights_file, Troj_Layer, Troj_Neurons, samp_labels, base_labels, re_epochs, num_classes, n_re_imgs_per_label, n_neurons, ctask_batch_size, max_input_length, trigger_pos, end_id, word_trigger_length, use_idxs, random_idxs_split, word_token_matrix, emb_id, gt_trigger_idxs, neutral_words, config):

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

    # if samp_labels[0] < 0:
    #     bloss_weight = 0
    #     word_trigger_length = 10

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

    print('olabels', olabels.shape,)

    # only use images from one label
    image_list = []
    poses_list = []
    attns_list = []
    label_list = []
    for base_label in base_labels:
        test_idxs = []
        for i in range(num_classes):
            if i == base_label or samp_labels[0] == -1:
                test_idxs1 = np.array( np.where(np.array(olabels) == i)[0] )
                test_idxs.append(test_idxs1)
        image_list.append(oimages[np.concatenate(test_idxs)])
        label_list.append(olabels[np.concatenate(test_idxs)])
        poses_list.append(oposes[np.concatenate(test_idxs)])
        attns_list.append(oattns[np.concatenate(test_idxs)])
    oimages = np.concatenate(image_list)
    olabels = np.concatenate(label_list)
    oposes = np.concatenate(poses_list)
    oattns = np.concatenate(attns_list)
    
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

    if samp_labels[0] == -1:
        target_labels = 1 - olabels
    else:
        target_labels = np.zeros_like(olabels) + samp_labels[0]

    target_labels_mask = np.zeros([olabels.shape[0], num_classes])
    for i in range(olabels.shape[0]):
        target_labels_mask[i, target_labels[i]] = 1


    label_mask = torch.zeros([ctask_batch_size, num_classes]).cuda()
    for i in range(ctask_batch_size):
        label_mask[i, samp_labels[i]] = 1

    base_label_mask = torch.zeros([ctask_batch_size, num_classes]).cuda()
    for i in range(ctask_batch_size):
        base_label_mask[i, base_labels[i]] = 1
    # print(label_mask)

    wrong_labels = base_labels
    wrong_label_mask = torch.zeros([ctask_batch_size, num_classes]).cuda()
    for i in range(ctask_batch_size):
        wrong_label_mask[i, wrong_labels[i]] = 1

    ovar_to_data = np.zeros((ctask_batch_size*n_re_imgs_per_label, ctask_batch_size))
    for i in range(ctask_batch_size):
        ovar_to_data[i*n_re_imgs_per_label:(i+1)*n_re_imgs_per_label,i] = 1

    if samp_labels[0] == 1:
        embedding_signs = torch.FloatTensor(embedding_signs_np.reshape(1,1,768)).cuda()
    else:
        embedding_signs = torch.FloatTensor(-embedding_signs_np.reshape(1,1,768)).cuda()

    delta_depth = len(use_idxs)

    start_time = time.time()
    delta_depth_map_mask = word_token_matrix

    word_token_length = word_token_matrix.shape[1]
    trigger_length = word_trigger_length * word_token_length

    print('delta_depth', delta_depth, random_idxs_split)

    gradual_small = random_idxs_split.copy()
    
    random_idxs_split = torch.FloatTensor(random_idxs_split).cuda()
    
    delta_init, mask_init, mask_map_init, end_poses = init_delta_mask(oimages, max_input_length, word_trigger_length, word_token_length, trigger_pos, delta_depth, end_id)
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
    obenign_masks = []
    obenign_mask_maps = []
    obenign_end_poses = []
    if trigger_pos != 0:
        # trigger at the end
        for i in range(len(benign_embeds)):
            bembeds = benign_embeds[i]
            benign_mask_map_init = np.zeros((bembeds.shape[0],max_input_length-2,trigger_length))
            benign_mask_init = np.zeros((bembeds.shape[0],max_input_length-2,1))
            bend_poses = []
            # print('benign batch', batch_benign_data_np.shape)
            for k, oimage in enumerate(bembeds):
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
    best_ce_loss = 10
    last_update = 0
    best_accs = [0]
    last_delta_sum_loss_weight = 0
    delta_infos = []
    for e in range(re_epochs):
        epoch_start_time = time.time()
        flogits = []

        images = oimages
        labels = olabels
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

            batch_data   = torch.FloatTensor(images[re_batch_size*i:re_batch_size*i+cre_batch_size]).cuda()
            batch_labels = torch.LongTensor(labels[re_batch_size*i:re_batch_size*i+cre_batch_size]).cuda()
            batch_target_labels = torch.LongTensor(target_labels[re_batch_size*i:re_batch_size*i+cre_batch_size]).cuda()
            batch_target_labels_mask = torch.LongTensor(target_labels_mask[re_batch_size*i:re_batch_size*i+cre_batch_size]).cuda()
            batch_poses  = torch.FloatTensor(poses[re_batch_size*i:re_batch_size*i+cre_batch_size]).cuda()
            batch_attns  = torch.FloatTensor(attns[re_batch_size*i:re_batch_size*i+cre_batch_size]).cuda()
            batch_v2d    = torch.FloatTensor(var_to_data[re_batch_size*i:re_batch_size*i+cre_batch_size]).cuda()
            if trigger_pos != 0:
                batch_mask     = torch.FloatTensor(mask_init[re_batch_size*i:re_batch_size*i+cre_batch_size]).cuda()
                batch_mask_map = torch.FloatTensor(mask_map_init[re_batch_size*i:re_batch_size*i+cre_batch_size]).cuda()

            # batch_neuron_mask  = torch.tensordot(batch_v2d, neuron_mask,  ([1], [0]))
            batch_label_mask   = torch.tensordot(batch_v2d, label_mask,   ([1], [0]))
            batch_base_label_mask   = torch.tensordot(batch_v2d, base_label_mask,   ([1], [0]))
            batch_wrong_label_mask   = torch.tensordot(batch_v2d, wrong_label_mask,   ([1], [0]))

            # print(images.shape, label_mask.shape, base_label_mask.shape, wrong_label_mask.shape)
            # sys.exit()

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
                elif trigger_pos == 1:
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


            if emb_id == 1:
                embedding_vector2 = torch.clamp( embedding_vector2, -1.1, 1.1)
            # embedding_vector2 = torch.tanh( embedding_vector2 ) * 1.1

            if emb_id == 1:
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

            logits = full_model(inputs_embeds=embedding_vector2, attention_mask=batch_attns,).logits


            logits_np = logits.cpu().detach().numpy()

            # benign_start_time = time.time()
            batch_benign_datas = []
            batch_benign_labels = []
            batch_benign_poses = []
            batch_benign_attns = []
            batch_benign_end_poses = []
            benign_logitses = []

            for bmodel in benign_models:
                blogits = bmodel(inputs_embeds=embedding_vector2, attention_mask=batch_attns,).logits
                benign_logitses.append(blogits)

            inner_outputs_b = None
            inner_outputs_a = None

            flogits.append(logits_np)
            loss, vloss1, vloss2, vloss3, relu_loss1, relu_loss2, logits_loss, benign_loss, bloss_weight, delta_sum_loss_weight, ce_loss\
                    = loss_fn(inner_outputs_b, inner_outputs_a, embedding_vector2, embedding_signs, logits, benign_logitses, batch_benign_labels, batch_labels, word_delta, use_delta, neuron_mask, batch_label_mask, batch_base_label_mask, batch_wrong_label_mask, batch_target_labels, batch_target_labels_mask, facc, e, re_epochs, ctask_batch_size, bloss_weight, i, samp_labels, emb_id, config)
            if e > 0:
                loss.backward(retain_graph=True)
                optimizer.step()

            benign_i += 1
            if benign_i >= math.ceil(float(len(images))/benign_batch_size):
                benign_i = 0

        flogits = np.concatenate(flogits, axis=0)
        fpreds = np.argmax(flogits, axis=1)
        preds = fpreds

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
                if optz_label == base_label:
                    optz_label = samp_label
                    if len(np.bincount(tpreds)) > 1:
                        optz_label = np.argsort(np.bincount(tpreds))[-2]
                if optz_label == base_label:
                    optz_label = samp_label

            # facc = np.sum(tpreds == optz_label) / float(tpreds.shape[0])
            facc = np.sum(tpreds == target_labels) / float(tpreds.shape[0])
            faccs.append(facc)

            use_label = samp_label

            use_labels.append(use_label)
            optz_labels.append(optz_label)
            wrong_labels.append(wrong_label)


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

        # if faccs[0] >= best_accs[0] and e >= 35:
        # if logits_loss.cpu().detach().numpy() > best_logits_loss :
        if ce_loss.cpu().detach().numpy() < best_ce_loss :
            best_delta = use_delta.cpu().detach().numpy()
            best_word_delta = word_delta.cpu().detach().numpy()
            best_logits_loss = logits_loss.cpu().detach().numpy()
            best_ce_loss = ce_loss.cpu().detach().numpy()
            best_accs = faccs
            last_update = e
            print('update', e, logits_loss.cpu().detach().numpy(),  best_logits_loss, faccs)

        if e == 0:
            base_logits_loss = logits_loss.cpu().detach().numpy()
            base_ce_loss = ce_loss.cpu().detach().numpy()

        if e % 10 == 0 or e == re_epochs-1 or last_update == e:

            print('epoch time', epoch_end_time - epoch_start_time)
            print(e, 'trigger_pos', trigger_pos, 'loss', loss.cpu().detach().numpy(), 'acc', faccs, 'base_labels', base_labels, 'sampling label', samp_labels,\
                    'optz label', optz_labels, 'use labels', use_labels,'wrong labels', wrong_labels,\
                    'logits_loss', logits_loss.cpu().detach().numpy(), 'ce loss', ce_loss.cpu().detach().numpy(),\
                    'benign_loss', benign_loss.cpu().detach().numpy(), 'bloss_weight', bloss_weight, 'delta_sum_loss_weight', delta_sum_loss_weight)
            print('olabels', olabels)
            print('target labels', target_labels)
            if inner_outputs_a != None and inner_outputs_b != None:
                print('vloss1', vloss1.cpu().detach().numpy(), 'vloss2', vloss2.cpu().detach().numpy(),\
                    'relu_loss1', relu_loss1.cpu().detach().numpy(), 'max relu_loss1', np.amax(inner_outputs_a.cpu().detach().numpy()),\
                    'relu_loss2', relu_loss2.cpu().detach().numpy(),\
                    )
            print('logits', flogits[:5,:])
            print('preds', preds, 'labels', labels)

            tuse_delta = use_delta.cpu().detach().numpy()
            if e == 0:
                tuse_delta0 = tuse_delta.copy()
            
            tword_delta = word_delta.cpu().detach().numpy()
            print('tword delta', tword_delta.shape, np.sum(tword_delta), np.sum(tuse_delta), np.sum(tuse_delta, axis=1), np.sum(tuse_delta < 1e-6))
            for i in range(tword_delta.shape[0]):
                for k in range(2):
                    print('position i', i, 'delta top', k, np.argsort(tword_delta[i])[-(k+1)], np.sort(tword_delta[i])[-(k+1)], '# larger than ', value_bound, np.sum(tword_delta[i] > value_bound))

            for i in range(tword_delta.shape[0]):
                for j in range(len(gt_trigger_idxs)):
                    print(gt_trigger_idxs[j], 'position i', i, tword_delta[i][gt_trigger_idxs[j]], 'rank', np.where(np.argsort(tword_delta[i])[::-1] == gt_trigger_idxs[j])[0] )

        delta_infos.append( (tword_delta, logits_loss.cpu().detach().numpy(), ce_loss.cpu().detach().numpy()) )

    if last_update == 0:
        best_delta = use_delta.cpu().detach().numpy()
        best_word_delta =word_delta.cpu().detach().numpy()
        last_update = e
        best_accs = faccs
    print('best loss e', last_update, best_accs, best_logits_loss, best_ce_loss)
    # print('best delta', best_delta.shape)
    # sys.exit()
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
    return faccs, delta, word_delta, mask, optz_labels, logits_loss.cpu().detach().numpy(), logits_loss.cpu().detach().numpy() -base_logits_loss, ce_loss.cpu().detach().numpy(), base_ce_loss - ce_loss.cpu().detach().numpy(), best_logits_loss, delta_infos

def re_mask(model_type, models, benign_models, benign_logits0, benign_embeds, benign_poses, benign_attentions, benign_ys, benign_word_labels, embedding_signs, neuron_dict, children, embeds, poses_emb_vector, texts, input_ids, attention_mask, labels, fys, n_neurons_dict, scratch_dirpath, re_epochs, num_classes, n_re_imgs_per_label, max_input_length, end_id, token_neighbours_dict, token_word_dict, word_token_dict, word_token_matrix, word_use_idxs, token_use_idxs, emb_id, max_nchecks, gt_trigger_idxs, neutral_words, config, use_idxs):

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

            # n_neurons = n_neurons_dict[tTroj_Layers[0]]
            n_neurons = 0

            # trigger_poses = [0]
            trigger_poses = [0,1]
            for tp_i, trigger_pos in enumerate(trigger_poses):

                start_time = time.time()
                accs, rdeltas, rword_deltas, rmasks, optz_labels, logits_loss, logits_loss_diff, ce_loss, ce_loss_diff, logits_loss0, delta_infos = reverse_engineer(model_type, models, benign_models, benign_logits0, benign_embeds, benign_poses, benign_attentions, benign_ys, benign_word_labels, embedding_signs, children, embeds, poses_emb_vector, attention_mask,  labels, weights_file, Troj_Layer, tTroj_Neurons, tsamp_labels, tbase_labels, re_epochs, num_classes, n_re_imgs_per_label, n_neurons, ctask_batch_size, max_input_length, trigger_pos, end_id, word_trigger_length, random_use_idxs, random_idxs_split, word_token_matrix, emb_id, gt_trigger_idxs, neutral_words, config)
                end_time = time.time()

                # sys.exit()

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

                    print('RE mask', Troj_Layer, Troj_Neuron, 'Label', samp_label, optz_label, 'RE acc', acc)
                    # if acc >= reasr_bound:
                    if True:
                        # asrs, ces, rdelta_idxs, rdelta_words, = test_trigger_pos1(model_type, models, rword_delta, token_use_idxs, texts, input_ids, attention_mask, fys, token_neighbours_dict, token_word_dict, word_token_dict, neutral_words, config, benign_models)
                        asrs, ces, rdelta_idxs, rdelta_words, = test_trigger_pos1(model_type, models, rword_delta, token_use_idxs, texts, input_ids, attention_mask, fys, token_neighbours_dict, token_word_dict, word_token_dict, neutral_words, config, [])
                        final_result = (rdelta, rmask, optz_label, RE_img, RE_mask, RE_delta, samp_label, base_label, acc, trigger_pos, asrs, ces, rdelta_idxs, rdelta_words, ce_loss)
                        validated_results.append( final_result )
                        asrs = np.array(asrs)
                        # find_trigger = ( trigger_pos == 0 and max(asrs[0]) > 0.75 ) or ( trigger_pos == 1 and max(np.reshape(asrs, -1)) > 0.89 ) 
                        # find_trigger, _ = check_find_trigger(np.reshape(asrs, -1), emb_id)
                        find_trigger = False
                        print('find_trigger', find_trigger, asrs[0], asrs[1], np.reshape(asrs, -1), )
                # if find_trigger:
                #     break

        return validated_results

def test_trigger_pos1(model_type, models, rdelta, use_idxs, texts, input_ids, attention_mask, fys, token_neighbours_dict, token_word_dict, word_token_dict, neutral_words, config, benign_models):

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
    top_k_candidates0 = top_k_candidates

    # print('rdeltas', rdelta.shape)
    # sys.exit()

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

    rdelta_idxs = sorted(list(set(rdelta_idxs)))

    print('word idxs', rdelta_idxs, )

    return test_word_idxs(model_type, models, rdelta_idxs, use_idxs, texts, input_ids, attention_mask, fys, token_neighbours_dict, token_word_dict, word_token_dict, neutral_words, config, benign_models)

def test_word_idxs(model_type, models, rdelta_idxs, use_idxs, texts, input_ids, attention_mask, fys, token_neighbours_dict, token_word_dict, word_token_dict, neutral_words, config, benign_models):

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
    top_k_candidates0 = top_k_candidates


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

    print('rdelta_idxs', len(rdelta_idxs), rdelta_idxs)

    crdelta_idxs = []
    for _ in rdelta_idxs:
        crdelta_idxs += list(_)

    rdelta_words = tokenizer.convert_ids_to_tokens( crdelta_idxs )

    return test_token_idxs(model_type, models, rdelta_idxs, rdelta_words, use_idxs, texts, input_ids, attention_mask, fys, token_neighbours_dict, token_word_dict, word_token_dict, neutral_words, config, benign_models)

def test_token_idxs(model_type, models, rdelta_idxs, rdelta_words, use_idxs, texts, input_ids, attention_mask, fys, token_neighbours_dict, token_word_dict, word_token_dict, neutral_words, config, benign_models):

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
    top_k_candidates0 = top_k_candidates


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

    min_len2_ces1 = 10
    min_len2_ces2 = 10
    min_len3_ces1 = 10
    min_len3_ces2 = 10
    min_len4_ces1 = 10
    min_len4_ces2 = 10

    len1_asrs1 = []
    len1_asrs2 = []
    len1_ces1 = []
    len1_ces2 = []

    for idx in rdelta_idxs:
        trigger_idxs = idx
        asrs, ces = inject_idx(tokenizer, full_model, texts, fys, trigger_idxs, config, benign_models,)
        # asrs, ces = inject_idx(tokenizer, full_model, texts, fys, trigger_idxs, config, [],)

        len1_asrs1.append(max(asrs[0]))
        len1_asrs2.append(max(asrs[1]))
        len1_ces1.append(min(ces[0]))
        len1_ces2.append(min(ces[1]))

    len1_asrs1 = np.array(len1_asrs1)
    max_len1_idxs1 = [rdelta_idxs[np.argmax(len1_asrs1)]]
    print('max len1 asrs1', np.amax(len1_asrs1), rdelta_idxs[np.argmax(len1_asrs1)])
    print('min len1 ces1', np.amin(len1_ces1), rdelta_idxs[np.argmin(len1_ces1)], np.sort(len1_ces1))
    max_len1_asrs1 = np.amax(len1_asrs1)

    len1_asrs2 = np.array(len1_asrs2)
    max_len1_idxs2 = [rdelta_idxs[np.argmax(len1_asrs2)]]
    print('max len1 asrs2', np.amax(len1_asrs2), rdelta_idxs[np.argmax(len1_asrs2)])
    print('min len1 ces2', np.amin(len1_ces2), rdelta_idxs[np.argmin(len1_ces2)], np.sort(len1_ces2))
    max_len1_asrs2 = np.amax(len1_asrs2)

    min_len1_ces1 = np.amin(len1_ces1)
    min_len1_ces2 = np.amin(len1_ces2)

    cands1 = []
    cands2 = []
    cands = []
    for i in range(len(len1_asrs1)):
        if len1_asrs1[i] > 0.2:
            cands1.append(rdelta_idxs[i])
            if rdelta_idxs[i] not in cands:
                cands.append(rdelta_idxs[i])
    for i in range(len(len1_asrs2)):
        if len1_asrs2[i] > 0.2:
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
                # for k in range(2):
                for k in range(1):
                    if k == 0:
                        trigger_idxs = tuple(list(topks[i]) + list(cands[j]))
                    else:
                        trigger_idxs = tuple(list(cands[j]) + list(topks[i]))
                    print('trigger_idxs', trigger_idxs,'k', k)
                    asrs, ces = inject_idx(tokenizer, full_model, texts, fys, trigger_idxs, config, benign_models,)
                    # asrs, ces = inject_idx(tokenizer, full_model, texts, fys, trigger_idxs, config, [],)
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

        cands1 = []
        cands2 = []
        cands = []
        for i in range(len(len2_asrs1)):
            if len2_asrs1[i] > 0.3:
                cands1.append(len2_idxs[i])
                if len2_idxs[i] not in cands:
                    cands.append(len2_idxs[i])
        for i in range(len(len2_asrs2)):
            if len2_asrs2[i] > 0.3:
                cands2.append(len2_idxs[i])
                if len2_idxs[i] not in cands:
                    cands.append(len2_idxs[i])

        if False:
        # if len(cands) > 0:
            width = 2
            # topks = list(np.array(len2_idxs)[np.argsort(len2_asrs1)[-width:]])
            # topks += list(np.array(len2_idxs)[np.argsort(len2_asrs2)[-width:]])
            topks = list(np.array(len2_idxs)[np.argsort(len2_ces1)[:width]])
            topks += list(np.array(len2_idxs)[np.argsort(len2_ces2)[:width]])
            topks = [tuple(_) for _ in topks]
            topks = sorted(list(set(topks)))
            print('topks', topks)
            print('cands', cands)
            len3_asrs1 = []
            len3_asrs2 = []
            len3_ces1 = []
            len3_ces2 = []
            len3_idxs = []
            for i in range(len(topks)):
                for j in range(len(cands)):
                    # for k in range(2):
                    for k in range(1):
                        if k == 0:
                            trigger_idxs = tuple(list(topks[i]) + list(cands[j]))
                        else:
                            trigger_idxs = tuple(list(cands[j]) + list(topks[i]))
                        print('trigger_idxs', trigger_idxs,'k', k)
                        # asrs, ces = inject_idx(tokenizer, full_model, texts, fys, trigger_idxs, config, benign_models,)
                        asrs, ces = inject_idx(tokenizer, full_model, texts, fys, trigger_idxs, config, [],)
                        len3_asrs1.append(max(asrs[0]))
                        len3_asrs2.append(max(asrs[1]))
                        len3_idxs.append(trigger_idxs)

                        len3_ces1.append(min(ces[0]))
                        len3_ces2.append(min(ces[1]))
            
            len3_asrs1 = np.array(len3_asrs1)
            print('max len3 asrs1', np.amax(len3_asrs1), tokenizer.convert_ids_to_tokens(len3_idxs[np.argmax(len3_asrs1)]) )
            max_len3_asrs1 = np.amax(len3_asrs1)
            max_len3_idxs1 = len3_idxs[np.argmax(len3_asrs1)]
            
            len3_asrs2 = np.array(len3_asrs2)
            print('max len3 asrs2', np.amax(len3_asrs2), tokenizer.convert_ids_to_tokens(len3_idxs[np.argmax(len3_asrs2)]) )
            max_len3_asrs2 = np.amax(len3_asrs2)
            max_len3_idxs2 = len3_idxs[np.argmax(len3_asrs2)]

            min_len3_ces1 = np.amin(len3_ces1)
            min_len3_ces2 = np.amin(len3_ces2)

    asrs1 = [[max_len1_asrs1, max_len2_asrs1, max_len3_asrs1, max_len4_asrs1, ], [max_len1_asrs2, max_len2_asrs2, max_len3_asrs2, max_len4_asrs2, ]]

    ces1 = [[min_len1_ces1, min_len2_ces1, min_len3_ces1, min_len4_ces1, ], [ min_len1_ces2, min_len2_ces2, min_len3_ces2, min_len4_ces2, ]]

    asrs = np.array(asrs1).reshape((2,4))

    ces = np.array(ces1).reshape((2,4))

    return asrs, ces, rdelta_idxs, rdelta_words


def inject_idx(tokenizer, full_model, texts, fys, trigger_idxs, config, benign_models):

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

    input_ids = []
    attention_mask = []
    end_poses = []
    end_ids = []
    for text in texts:
        results = tokenizer(text, max_length=max_input_length - 2, padding="max_length", truncation=True, return_tensors="pt")
        tinput_ids = results.data['input_ids']
        tattention_mask = results.data['attention_mask']
        for att_i in range(max_input_length-2):
            if tattention_mask[0,att_i] == 0:
                break
        if att_i < max_input_length//2:
            end_poses.append(att_i-2)
        else:
            end_poses.append(att_i-len(trigger_idxs)-2)
        tattention_mask[0,att_i:att_i+len(trigger_idxs)] = 1
        input_ids.append(tinput_ids)
        attention_mask.append(tattention_mask)
    input_ids = torch.cat(input_ids, axis=0)
    attention_mask = torch.cat(attention_mask, axis=0)
    # print('input_ids', input_ids.shape, 'attention_mask', attention_mask.shape)
    input_ids = input_ids.cuda()
    attention_mask = attention_mask.cuda()
    
    input_ids1 = torch.zeros_like(input_ids)
    input_ids2 = torch.zeros_like(input_ids)
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

    logits = full_model(input_ids=input_ids1, attention_mask=attention_mask,).logits
    ce_loss1 = F.nll_loss(F.log_softmax(logits[np.where(fys==0)[0]], dim=1),  torch.LongTensor(1-fys[np.where(fys==0)[0]]).cuda(), ).cpu().detach().numpy()
    ce_loss2 = F.nll_loss(F.log_softmax(logits[np.where(fys==1)[0]], dim=1),  torch.LongTensor(1-fys[np.where(fys==1)[0]]).cuda(), ).cpu().detach().numpy()
    logits_np = logits.cpu().detach().numpy()
    preds = np.argmax(logits_np, axis=1)
    print('preds1', preds, fys)
    # print('acc', np.sum(preds==fys)/float(len(preds)))
    tpreds11 = preds[np.where(fys==0)[0]]
    tlogits11 = preds[np.where(fys==0)[0]]
    asr11 = np.sum(tpreds11==1)/float(len(tpreds11))
    tpreds12 = preds[np.where(fys==1)[0]]
    asr12 = np.sum(tpreds12==0)/float(len(tpreds12))

    logits = full_model(input_ids=input_ids2, attention_mask=attention_mask,).logits
    ce_loss3 = F.nll_loss(F.log_softmax(logits[np.where(fys==0)[0]], dim=1),  torch.LongTensor(1-fys[np.where(fys==0)[0]]).cuda(), ).cpu().detach().numpy()
    ce_loss4 = F.nll_loss(F.log_softmax(logits[np.where(fys==1)[0]], dim=1),  torch.LongTensor(1-fys[np.where(fys==1)[0]]).cuda(), ).cpu().detach().numpy()
    logits_np = logits.cpu().detach().numpy()
    preds = np.argmax(logits_np, axis=1)
    print('preds2', preds, fys)
    # print('acc', np.sum(preds==fys)/float(len(preds)))
    tpreds21 = preds[np.where(fys==0)[0]]
    asr21 = np.sum(tpreds21==1)/float(len(tpreds21))
    tpreds22 = preds[np.where(fys==1)[0]]
    asr22 = np.sum(tpreds22==0)/float(len(tpreds22))

    if len(benign_models) > 0:
    # if False:
        blogits1 = benign_models[0](input_ids=input_ids1, attention_mask=attention_mask,).logits
        blogits_np1 = blogits1.cpu().detach().numpy()
        bpreds1 = np.argmax(blogits_np1, axis=1)
        bacc1 = np.sum(bpreds1==fys)/float(len(bpreds1))
        print('bpreds1', bpreds1, fys)

        blogits2 = benign_models[0](input_ids=input_ids2, attention_mask=attention_mask,).logits
        blogits_np2 = blogits2.cpu().detach().numpy()
        bpreds2= np.argmax(blogits_np2, axis=1)
        bacc2 = np.sum(bpreds2==fys)/float(len(bpreds2))
        print('bpreds2', bpreds2, fys)

        baccs = np.array([bacc1, bacc2, ])
        print('benign acc', baccs)
        if np.amin(baccs) < 0.85:
        # if False:
            asr11 = -1 
            asr12 = -1 
            asr21 = -1 
            asr22 = -1 
            ce_loss1 = 100
            ce_loss2 = 100
            ce_loss3 = 100
            ce_loss4 = 100

    print('test trigger idxs', trigger_idxs, tokenizer.convert_ids_to_tokens( trigger_idxs ), tpreds11.shape, tpreds12.shape, 'asrs', [asr11, asr12, asr21, asr22])
    # sys.exit()

    return [[asr11, asr12],  [asr21, asr22]], [[ce_loss1, ce_loss2], [ce_loss3, ce_loss4]]


def sc_trojan_detector(model_filepath, tokenizer_filepath, result_filepath, scratch_dirpath, examples_dirpath, round_training_dataset_dirpath, learned_parameters_dirpath, features_filepath, parameters):
    start = time.time()
    
    print('sc parameters', parameters)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for_submission = parameters[0]
    is_configure   = parameters[1]
    config = {}
    config['re_mask_lr']            = parameters[2]
    # config['re_mask_lr']            = 1e-1
    # config['re_mask_lr']            = 1e0
    config['re_epochs']             = parameters[3]
    config['word_trigger_length']   = parameters[4]
    config['re_batch_size']         = 40
    config['n_re_imgs_per_label']   = 20
    config['n_max_imgs_per_label']  = 20
    config['max_input_length']      = 80
    config['top_k_candidates']      = 10
    config['reasr_bound']           = 0.8
    config['value_bound']           = 0.5
    config['tasks_per_run']         = 1
    config['bloss_weight']          = 2e-1
    # config['bloss_weight']          = 5e-1
    # config['bloss_weight']          = 0
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

    print('sc config', config)


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
    print('classifier', list(model.named_modules()), )
    # try:
    if True:
        num_classes = list(model.named_modules())[-1][1].out_features
    # except:
    #     num_classes = list(model.named_modules())[-2].out_features
    print('num_classes', num_classes)
    word_trigger_length = config['word_trigger_length']

    # same for the 3 basic types
    children = list(model.children())
    target_layers = ['Linear']

    if for_submission:
        all_file                            = '/all_words_amazon_100k_0.txt'
        all_with_hyphen_file                = '/all_words_amazon_100k_3.txt'
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
        dbert_with_hyphen_word_token_matrix_fname       = '/distilbert_amazon_100k_3_word_token_matrix6_4tokens_trigger.pkl'
        bert_with_hyphen_word_token_matrix_fname        = '/google_amazon_100k_3_word_token_matrix6_4tokens_trigger.pkl'
        roberta_with_hyphen_word_token_matrix_fname     = '/roberta_base_amazon_100k_3_word_token_matrix6_4tokens_trigger.pkl'
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
        dbert_with_hyphen_word_token_matrix_fname       = '../distilbert_amazon_100k_3_word_token_matrix6_4tokens_trigger.pkl'
        bert_with_hyphen_word_token_matrix_fname        = '../google_amazon_100k_3_word_token_matrix6_4tokens_trigger.pkl'
        roberta_with_hyphen_word_token_matrix_fname     = '../roberta_base_amazon_100k_3_word_token_matrix6_4tokens_trigger.pkl'
        all_file                            = '../all_words_amazon_100k_0.txt'
        all_with_hyphen_file                = '../all_words_amazon_100k_3.txt'
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

    neutral_words_with_hyphen = [] 
    for line in open(all_with_hyphen_file):
        neutral_words_with_hyphen.append(line.split()[0])

    fns = [os.path.join(examples_dirpath, fn) for fn in os.listdir(examples_dirpath) if fn == 'clean-example-data.json']
    fns.sort()
    examples_filepath = fns[0]

    with open(examples_filepath) as json_file:
        clean0_json = json.load(json_file)
    clean0_filepath = os.path.join(scratch_dirpath,'clean0_data.json')
    with open(clean0_filepath, 'w') as f:
        json.dump(clean0_json, f)
    dataset = datasets.load_dataset('json', data_files=[clean0_filepath], field='data', keep_in_memory=True, split='train', cache_dir=os.path.join(scratch_dirpath, '.cache'))

    texts = []
    fys = []
    for data_item in dataset:
        # print(data_item)
        
        texts.append( data_item['data'] )
        fys.append( data_item['label'] )

    # fns = [os.path.join(examples_dirpath, fn) for fn in os.listdir(examples_dirpath) if fn.endswith('.txt')]
    # fns.sort()  # ensure file ordering

    # texts = []
    # fys = []
    # for fn in fns:
    #     # load the example
    #     with open(fn, 'r') as fh:
    #         text = fh.read()
    #     if int(fn[:-4].split('/')[-1].split('_')[1]) < 1:
    #         fy = 0
    #     else:
    #         fy = 1
    #     if np.sum(np.array(fys) == fy) >= n_max_imgs_per_label:
    #         continue
    #     texts.append(text)
    #     fys.append(fy)
       

    fys = np.array(fys)
    print('fys', fys)

    tokenizer = torch.load(tokenizer_filepath)

    # set the padding token if its undefined
    if not hasattr(tokenizer, 'pad_token') or tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print('tokenizer', tokenizer.__class__.__name__,  'max_input_length', max_input_length)

    if model_type.startswith('Electra'):
        end_id = 102
        end_attention = 1
        # benign_names = 'id-00000008'
        max_nchecks = 14
        emb_id = 1
        word_token_matrix0 = pickle.load(open(bert_word_token_matrix_fname, 'rb'))
        word_with_hyphen_token_matrix0 = pickle.load(open(bert_with_hyphen_word_token_matrix_fname, 'rb'))
        word_token_dict = pickle.load(open(bert_word_token_dict_fname, 'rb'))
        token_word_dict = pickle.load(open(bert_token_word_dict_fname, 'rb'))
        token_neighbours_dict = pickle.load(open(bert_token_neighbours_dict_fname, 'rb'))
        token_use_idxs = list(range(30522))
    elif model_type.startswith('DistilBert'):
        end_id = 102
        end_attention = 1
        # benign_names = 'id-00000009'
        # max_nchecks = 18
        max_nchecks = 20
        emb_id = 0
        word_token_matrix0 = pickle.load(open(dbert_word_token_matrix_fname, 'rb'))
        word_with_hyphen_token_matrix0 = pickle.load(open(dbert_with_hyphen_word_token_matrix_fname, 'rb'))
        word_token_dict = pickle.load(open(dbert_word_token_dict_fname, 'rb'))
        token_word_dict = pickle.load(open(dbert_token_word_dict_fname, 'rb'))
        token_neighbours_dict = pickle.load(open(dbert_token_neighbours_dict_fname, 'rb'))
        token_use_idxs = list(range(28996))
    elif model_type.startswith('Roberta'):
        end_id = 1
        end_attention = 1
        # benign_names = 'id-00000000'
        max_nchecks = 14
        emb_id = 2
        word_token_matrix0 = pickle.load(open(roberta_word_token_matrix_fname, 'rb'))
        word_with_hyphen_token_matrix0 = pickle.load(open(roberta_with_hyphen_word_token_matrix_fname, 'rb'))
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
        if not line.startswith('sc1'):
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

    depth = tokenizer.vocab_size

    # one_hot_emb = nn.Embedding(depth, depth)
    # one_hot_emb.weight.data = torch.eye(depth)
    # one_hot_emb = one_hot_emb

    full_model.eval()
    model.eval()

    embedding_signs = np.zeros((768,))


    input_ids = []
    attention_mask = []
    for text in texts:
        results = tokenizer(text, max_length=max_input_length - 2, padding="max_length", truncation=True, return_tensors="pt")
        tinput_ids = results.data['input_ids']
        tattention_mask = results.data['attention_mask']
        input_ids.append(tinput_ids)
        attention_mask.append(tattention_mask)
    input_ids = torch.cat(input_ids, axis=0)
    attention_mask = torch.cat(attention_mask, axis=0)
    print('input_ids', input_ids.shape, 'attention_mask', attention_mask.shape)
    input_ids = input_ids.cuda()
    attention_mask = attention_mask.cuda()
    print(input_ids[0])
    print(attention_mask[0])


    logits = full_model(input_ids=input_ids, attention_mask=attention_mask,).logits
    print('logits', logits.shape)
    logits = logits.cpu().detach().numpy()
    preds = np.argmax(logits, axis=1)
    print('preds', preds, 'fys', fys)
    print('acc', np.sum(preds==fys)/float(len(preds)))

    print(logits.shape, )
    layer_i = 0
    sample_layers = ['identical']
    n_neurons_dict = {}

    # print('nds', nds)
    key = model_filepath
    neuron_dict = {}
    neuron_dict[key] = []

    neuron_dict[key].append( ('identical_0', 0, 1, 0.1, 0) )
    neuron_dict[key].append( ('identical_0', 0, 0, 0.1, 1) )
    neuron_dict[key].append( ('identical_0', 0, -1, 0.1, -1) )

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

    optz_embeds = []
    optz_poses = []
    optz_attentions = []
    optz_ys = []
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
            optz_inputs.append(input_ids[i].cpu().detach().numpy())
            optz_slots[fys[i]] += 1
        # if np.sum(optz_slots) >= n_re_imgs_per_label * num_classes:
        #     break
    optz_embeds = np.array(optz_embeds)
    optz_poses = np.array(optz_poses)
    optz_attentions = np.array(optz_attentions)
    optz_ys = np.array(optz_ys)
    optz_inputs = np.array(optz_inputs)
    optz_inputs = torch.LongTensor(optz_inputs).cuda()
    print('optz data', optz_ys, optz_ys.shape, optz_embeds.shape, optz_poses.shape, optz_attentions.shape,)

    test_embeds = embeds
    test_poses = poses_emb_vector
    test_attentions = attention_mask
    test_ys = fys

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
        # benign_model_fn = '/data/share/trojai/trojai-round9-v1-dataset/models/{0}/model.pt'.format(bmname)
        # benign_examples_dirpath = '/data/share/trojai/trojai-round9-v1-dataset/models/{0}/clean_example_data'.format(bmname)
        # benign_model_fn = '{1}/models/{0}/model.pt'.format(bmname, round_training_dataset_dirpath)
        # benign_examples_dirpath = '{1}/models/{0}/clean_example_data'.format(bmname, round_training_dataset_dirpath)
        benign_model_fn = '{1}/sc1_benign_models/{0}/model.pt'.format(bmname, learned_parameters_dirpath)
        benign_examples_dirpath = '{1}/sc1_benign_models/{0}/clean_example_data'.format(bmname, learned_parameters_dirpath)
        print('load benign model', benign_model_fn)
        bmodel = torch.load(benign_model_fn).cuda()
        bmodel.eval()
        benign_models.append(bmodel)

    benign_logits0 = []

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

    # if True:
    if False:
        asrs, ces = inject_idx(tokenizer, full_model, texts, fys, gt_trigger_idxs, config, benign_models)
        print('gt_trigger_idxs', gt_trigger_idxs, asrs, ces)
        sys.exit()


    benign_names_file = learned_parameters_dirpath + '/benign_names.txt'
    benign_names2_dict = {}
    for line in open(benign_names_file):
        if not line.startswith('sc2'):
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
            bmodel2 = torch.load(learned_parameters_dirpath+'/sc2_benign_models/{0}/model.pt'.format(benign_name2), map_location=torch.device(device))
            bmodel2.eval()
            cos = F.cosine_similarity(full_model.get_input_embeddings().weight, bmodel2.get_input_embeddings().weight, dim=1).cpu().detach().numpy()
            # token_use_idxs += np.argsort(cos)[:3000].tolist()
            # token_use_idxs += np.argsort(cos)[:2000].tolist()
            token_use_idxs += np.argsort(cos)[:500].tolist()
            # token_use_idxs += np.argsort(cos)[:50].tolist()
            # token_use_idxs += np.argsort(cos)[:10000].tolist()
            top10_token_use_idxs += np.argsort(cos)[:10].tolist()

        token_use_idxs = sorted(list(set(token_use_idxs)))
        print('token_use_idxs', len(token_use_idxs))
        for idx in token_use_idxs:
            token_mask[idx] = 1

        for idx in gt_trigger_idxs:
            print('gt token idx', idx, idx in token_use_idxs, np.where(idx == np.argsort(cos) ))
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
            for i in range(word_with_hyphen_token_matrix0.shape[0]):
                if re.search(r'[0-9]+',neutral_words_with_hyphen[i], ) and re.search(r'[A-Za-z]+',neutral_words_with_hyphen[i], ):
                    continue
                for j in range(word_with_hyphen_token_matrix0.shape[1]):
                    if top10_token_mask[word_with_hyphen_token_matrix0[i,j]] == 1:
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
            print('idx', neutral_words_with_hyphen[idx])

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

            # asrs = inject_idx(tokenizer, full_model, texts, fys, top10_token_use_idxs, config)
            # asrs1, ces1 = test_word_idxs(model_type, models, top10_use_idxs, use_idxs, texts, input_ids, attention_mask, fys, token_neighbours_dict, token_word_dict, word_token_dict, neutral_words_with_hyphen, config, [])[:2]
            asrs1, ces1 = test_word_idxs(model_type, models, top10_use_idxs, use_idxs, texts, input_ids, attention_mask, fys, token_neighbours_dict, token_word_dict, word_token_dict, neutral_words_with_hyphen, config, benign_models)[:2]
            rdelta_idxs = [(_,) for _ in top10_token_use_idxs]

            asrs2, ces2 = test_token_idxs(model_type, models, rdelta_idxs, '', use_idxs, texts, input_ids, attention_mask, fys, token_neighbours_dict, token_word_dict, word_token_dict, neutral_words, config, [])[:2]

            asrs1 = np.array(asrs1)
            asrs2 = np.array(asrs2)
            asrs = np.amax(np.array([asrs1, asrs2]), axis=0)

            print(asrs1.shape, asrs2.shape, asrs.shape)
            print(asrs1, asrs2, asrs)

            max_len1_asrs1 = asrs[0][0]
            max_len1_asrs2 = asrs[1][0]
            max_len2_asrs1 = asrs[0][1]
            max_len2_asrs2 = asrs[1][1]
            max_len3_asrs1 = asrs[0][2]
            max_len3_asrs2 = asrs[1][2]
            max_len4_asrs1 = asrs[0][3]
            max_len4_asrs2 = asrs[1][3]

            ces1 = np.array(ces1)
            ces2 = np.array(ces2)
            ces = np.amin(np.array([ces1, ces2], ), axis=0)

            print(ces1.shape, ces2.shape, ces.shape)
            print(ces1, ces2, ces)

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


    results = re_mask(model_type, models, \
            benign_models, benign_logits0, benign_embeds, benign_poses, benign_attentions, benign_ys, benign_word_labels, embedding_signs, \
            neuron_dict, children, \
            optz_embeds, optz_poses, texts, optz_inputs, optz_attentions, optz_ys, fys,\
            n_neurons_dict, scratch_dirpath, re_epochs, num_classes, n_re_imgs_per_label, max_input_length, end_id, \
            token_neighbours_dict, token_word_dict, word_token_dict, word_token_matrix, word_use_idxs, token_use_idxs, emb_id, max_nchecks, gt_word_idxs, neutral_words, config, use_idxs)

    optm_end = time.time()

    print('# results', len(results))

    # first test each trigger
    reasr_info = []
    reasr_per_labels = []
    result_infos = []
    diff_percents = []
    full_ces = []
    full_asrs = []
    full_result_idx = 0
    find_triggers = []
    features = [emb_id, ]
    # celosses = [[] for _ in range(2)]
    celosses = []
    for result in results:
        rdelta, rmask, optz_label, RE_img, RE_mask, RE_delta, samp_label, base_label, acc, trigger_pos, asrs, ces, rdelta_idxs, rdelta_words, ce_loss = result

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

        # find_trigger = ( trigger_pos == 0 and max(asrs[0]) > 0.75 ) or ( trigger_pos == 1 and max(asrs[0]+asrs[1]) > 0.89 ) 
        # find_trigger = ( trigger_pos == 0 and max(asrs[0]) > 0.75 ) or ( trigger_pos == 1 and max(np.reshape(asrs, -1)) > 0.89 ) 
        # find_trigger, _ = check_find_trigger(np.reshape(asrs, -1), emb_id)
        find_trigger = False

        nasrs = np.array([max_len1_asrs1, max_len2_asrs1, max_len3_asrs1, max_len4_asrs1, max_len1_asrs2, max_len2_asrs2, max_len3_asrs2, max_len4_asrs2, ])

        nces  = np.array([min_len1_ces1, min_len2_ces1, min_len3_ces1, min_len4_ces1, min_len1_ces2, min_len2_ces2, min_len3_ces2, min_len4_ces2, ])

        reasr_info.append(['{:.2f}'.format(reasr), '{:.2f}'.format(reasr_per_label), \
                max_len1_asrs1, max_len2_asrs1, max_len3_asrs1, max_len4_asrs1, max_len1_asrs2, max_len2_asrs2, max_len3_asrs2, max_len4_asrs2, \
                'mask', str(optz_label), str(samp_label), str(base_label), 'trigger posistion', str(trigger_pos), RE_img, RE_mask, RE_delta, np.sum(rmask), \
                accs_str, rdelta_words_str, rdelta_idxs_str, len1_idxs_str1, len2_idxs_str1, len3_idxs_str1, len1_idxs_str2, len2_idxs_str2, len3_idxs_str2])

        reasr_per_labels.append(reasr_per_label)
        full_asrs.append(nasrs)
        full_ces.append(nces)
        find_triggers.append(find_trigger)

        # celosses[trigger_pos].append(ce_loss)
        celosses.append(ce_loss)
        print('ce loss', ce_loss)

    features += list(np.amax(np.array(full_asrs), axis=0))
    features += list(np.amin(np.array(full_ces), axis=0))
    features += list(np.array(full_asrs).reshape(-1))
    features += list(np.array(full_ces).reshape(-1))

    celosses = np.array(celosses)
    # print('celosses', celosses, np.amin(np.array(celosses), axis=1), )
    # features += list(np.amin(np.array(celosses), axis=1))
    features += list(np.array(celosses))
    features += list(np.array(reasr_per_labels))

    features += list(np.array(nasrs0))
    features += list(np.array(nces0))

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

    output = 0.5
    x = features
    nx = x[:17] + x[17+96:17+96+6] + x[17:17+96] + x[17+96+6:]
    x = nx
    # nx = [x[0], x[9], x[13], x[10], x[14], min([x[9], x[13]]), min([x[10], x[14]]), np.min(x[-8:-6]), np.min(x[-4:-2]), np.min(x[17:21]), np.min(x[21:23]),] + x[17:23] # + nx
    nx = [x[0], np.amin([x[9], x[13]]), np.amin([x[10], x[14]]), np.amin(x[-8:-6]), np.amin(x[-4:-2]), ] + x[21:23]


    features = nx
    xs = np.array([features])
    
    roberta_x = nx
    roberta_x = np.array([roberta_x])

    if not is_configure:
        cls = pickle.load(open(os.path.join(learned_parameters_dirpath, 'rf_lr_sc3.pkl'), 'rb'))
        confs = cls.predict_proba(xs)[:,1]
        confs = np.clip(confs, 0.025, 0.975)
        print('confs', confs)
        output = confs[0]

    return output, features, roberta_x


