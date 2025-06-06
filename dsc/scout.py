#Author 

import torch
import torch.nn as nn
import os
import sys
import numpy as np
import argparse
import yaml
import numpy as np
from models.bert2bert import Bert2BertSynCtrl
from transformers import BertConfig
from training.trainer import Trainer 
from dataloader.dataloader import PreTrainDataLoader, FinetuneDataLoader
from generator.generator import Generator

class SCOUT(object):

    def __init__(self,
                config_path,
                op_dir,
                random_seed,
                datapath,
                device,
                topk=None,
                weights=None,
                lowrank=False,
                classes=None):

        self.config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)
        self.op_dir = op_dir
        self.target_id = self.config['target_id']
        self.interv_time = self.config['interv_time']
        self.random_seed = random_seed
        self.datapath = datapath
        self.device = device
        self.lowrank = self.config['lowrank']
        self.rank = self.config['rank']
        self.classes = classes
        self.topk = topk
        self.weights = weights

        self.config_model = BertConfig(hidden_size = self.config['hidden_size'],
                            num_hidden_layers = self.config['n_layers'],
                            num_attention_heads = self.config['n_heads'],
                            intermediate_size = 4*self.config['hidden_size'],
                            vocab_size = 0,
                            max_position_embeddings = 0,
                            output_hidden_states = True,
                            )

        self.config_model.add_syn_ctrl_config(K=self.config['K'],
                                    pre_int_len=self.config['pre_int_len'],
                                    post_int_len=self.config['post_int_len'],
                                    feature_dim=self.config['feature_dim'],
                                    time_range=self.config['time_range'],
                                    seq_range=self.config['seq_range'],
                                    cont_dim=self.config['cont_dim'],
                                    discrete_dim=self.config['discrete_dim'],
                                    classes = classes)
        self.model = Bert2BertSynCtrl(self.config_model, self.random_seed)
        self.model = self.model.to(self.device)
        if not(os.path.exists(op_dir)):
            os.mkdir(op_dir)


    def fit(self, checkpoint_pretrain = None, pretrain =True, pretrain_iters = 5e4, finetune_iters = 5e3):

        if pretrain == True:
            self.pretrain(checkpoint_pretrain, num_iters=pretrain_iters, lowrank_approx = self.lowrank, rank = self.rank)

        if self.model.Bert2BertSynCtrl.config.encoder.K == self.config['K']:

            print('Modifying K')
            self.model.config.K+=1
            self.model.K+=1
            self.model.Bert2BertSynCtrl.encoder.config.K+=1
            self.model.Bert2BertSynCtrl.decoder.config.K+=1

        self.finetune(num_iters=finetune_iters)

    def predict(self):


        if self.model.Bert2BertSynCtrl.config.encoder.K == self.config['K']:
            print('Modifying K')
            self.model.config.K+=1
            self.model.K+=1
            self.model.Bert2BertSynCtrl.encoder.config.K+=1
            self.model.Bert2BertSynCtrl.decoder.config.K+=1
            
        generator = Generator(self.model,
                    self.device,
                    self.datapath,
                    self.target_id,
                    self.interv_time,
                    self.lowrank,
                    topk=self.topk,
                    weights=self.weights)

        target_data =  generator.sliding_window_generate()
       

        return target_data[self.interv_time:]

    def return_attention(self, interv_time):


        if self.model.Bert2BertSynCtrl.config.encoder.K == self.config['K']:
            print('Modifying K')
            self.model.config.K+=1
            self.model.K+=1
            self.model.Bert2BertSynCtrl.encoder.config.K+=1
            self.model.Bert2BertSynCtrl.decoder.config.K+=1

        generator = Generator(self.model,
                    self.device,
                    self.datapath,
                    self.target_id,
                    interv_time,
                    self.lowrank)

        attention_weights =  generator.sliding_attention()
       

        return attention_weights


    def pretrain(self, checkpoint_pretrain=None, num_iters=5e4, lowrank_approx = False, rank = 10):


        dataloader_pretrain = PreTrainDataLoader(self.random_seed,
                                    self.datapath,
                                    self.device,
                                    self.config_model,
                                    self.target_id,
                                    self.topk,
                                    self.weights,
                                    lowrank_approx = lowrank_approx,
                                    rank = rank)

        optimizer_pretrain = torch.optim.AdamW(self.model.parameters(),
                                    lr=eval(self.config['lr']),
                                    weight_decay=eval(self.config['weight_decay']),
                                    )

        warmup_steps = self.config['warmup_steps']
        scheduler = torch.optim.lr_scheduler.LambdaLR(
                    optimizer_pretrain,
                    lambda steps: min((steps+1)/warmup_steps,1))
        batch_size = self.config['batch_size']
        op_path_pretrain = self.op_dir + 'pretrain/'
        if not(os.path.exists(op_path_pretrain)):
            os.mkdir(op_path_pretrain)

        trainer_pretrain = Trainer(self.model,
                        optimizer_pretrain,
                        dataloader_pretrain,
                        op_path_pretrain,
                        batch_size,
                        scheduler
                        )

        print('Pretraining model on donor units')

        self.model = trainer_pretrain.train(int(num_iters),checkpoint_pretrain)


    def finetune(self, num_iters=5e3, lowrank_approx = False, rank = 10):

        if self.model.Bert2BertSynCtrl.config.encoder.K == self.config['K']:

            print('Modifying K')
            self.model.config.K+=1
            self.model.K+=1
            self.model.Bert2BertSynCtrl.encoder.config.K+=1
            self.model.Bert2BertSynCtrl.decoder.config.K+=1

        dataloader_finetune = FinetuneDataLoader(self.random_seed,
                                    self.datapath,
                                    self.device,
                                    self.config_model,
                                    self.target_id,
                                    self.interv_time,
                                    self.topk,
                                    self.weights,
                                    lowrank_approx = lowrank_approx,
                                    rank = rank)

        optimizer_finetune = torch.optim.AdamW(self.model.parameters(),
                                lr=eval(self.config['lr']),
                                weight_decay=eval(self.config['weight_decay']),
                                    )
        batch_size = self.config['batch_size']
        op_path_finetune = self.op_dir + 'finetune/'
        if not(os.path.exists(op_path_finetune)):
            os.mkdir(op_path_finetune)
        trainer = Trainer(self.model,
                            optimizer_finetune,
                            dataloader_finetune,
                            op_path_finetune,
                            batch_size
                            )

        print('Fitting model on target unit')

        self.model = trainer.train(int(num_iters))

    def load_model_from_checkpoint(self, modelpath):

        cp = torch.load(modelpath,map_location=self.device, weights_only=False)
        state_dict = cp['model_state_dict']
        self.model.load_state_dict(state_dict)


