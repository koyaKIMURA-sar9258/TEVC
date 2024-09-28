import pandas as pd
import numpy as np
import h5py
import json

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.vocab import vocab
from collections import Counter

def build_vocab(texts, tokenizer, specials):
    counter = Counter()
    for text in texts:
        text = text.rstrip()
        counter.update(tokenizer(text))

    voc = vocab(counter, specials=specials)
    voc.set_default_index(voc[specials[0]]) # specials[0]:<unk> as default
    return voc

def convert_text_to_indexes(text, vocab, tokenizer, s_token, e_token):
    return [vocab[s_token]] + [vocab[token] for token in tokenizer(text)] + [vocab[e_token]]

def pre_process(texts, vocab, tokenizer, specials):
    p_token, s_token, e_token = specials[1], specials[2], specials[3]
    data = []
    for text in texts:
        t_tensor = torch.tensor(
            convert_text_to_indexes(text=text, vocab=vocab, tokenizer=tokenizer,
                                    s_token=s_token, e_token=e_token),
            dtype=torch.long
        )
        data.append(t_tensor)
    data = pad_sequence(data, batch_first=True, padding_value=vocab[p_token])
    return data

def generate_token(texts, all_texts, specials):
    tokenizer = get_tokenizer('spacy', language='en_core_web_sm')
    vocab = build_vocab(all_texts, tokenizer, specials)
    token_cap = pre_process(texts, vocab, tokenizer, specials)

    return token_cap, vocab

class MSVDDataset(Dataset):

    def __init__(
            self, device, cap_path, start_token, end_token, pad_token,
            train_id, val_id, video_path, mode):
        self.device = device
        self.start_token = start_token
        self.end_token = end_token
        self.pad_token = pad_token
        self.specials = ['<unk>', pad_token, start_token, end_token]
        
        # Caption Data
        cap_ds = pd.read_csv(cap_path, names=[0,1])
        if mode == 'train':
            self.cap_ds = cap_ds[:train_id]
        elif mode == 'validate':
            self.cap_ds = cap_ds[train_id:val_id].reset_index(drop=True)
        elif mode == 'test':
            self.cap_ds = cap_ds[val_id:].reset_index(drop=True)
        else:
            raise Exception('Argument Error: mode')
        
        # token: caption token in dataset, all_vocab: all captions in dataset.
        self.token_cap, self.all_vocab = generate_token(self.cap_ds[1], cap_ds[1], self.specials)
        # self.all_token_cap, self.all_vocab = generate_token(cap_ds[1], self.specials)
        self.voc_size = len(self.all_vocab)
        self.pad_idx = self.all_vocab.get_stoi()[pad_token]
        self.start_idx = self.all_vocab.get_stoi()[start_token]
        self.end_idx = self.all_vocab.get_stoi()[end_token]

        # RGB and Flow feature Data
        video_feat = h5py.File(video_path)
        self.rgb_feat = video_feat['msvd/rgb']
        self.flow_feat = video_feat['msvd/flow']
        self.video_list = list(self.rgb_feat.keys())
        rgb_stack, flow_stack = [], []
        for v_name in self.video_list:
            rgb = torch.from_numpy(self.rgb_feat[f'{v_name}/rgb_feat'][:]).float()
            flow = torch.from_numpy(self.flow_feat[f'{v_name}/flow_feat'][:]).float()
            # rgb = torch.from_numpy(self.rgb_feat[f'{v_name}/rgb_feat'][:]).float().to(self.device)
            # flow = torch.from_numpy(self.flow_feat[f'{v_name}/flow_feat'][:]).float().to(self.device)
            rgb_stack.append(rgb)
            flow_stack.append(flow)
        self.rgb_stack = pad_sequence(rgb_stack, batch_first=True, padding_value=self.pad_idx)
        self.flow_stack = pad_sequence(flow_stack, batch_first=True, padding_value=0)
        print(f'Created MSVD {mode} Dataset.')

    def __getitem__(self, index):
        video_name = self.cap_ds[0][index]
        caption = self.token_cap[index]
        feat_index = self.video_list.index(video_name)
        rgb_feat = self.rgb_stack[feat_index].to(self.device)
        flow_feat = self.flow_stack[feat_index].to(self.device)
        return caption, rgb_feat, flow_feat, video_name

    def __len__(self):
        return len(self.cap_ds)

    def collate_batch(self, batch):
        captions, rgb_feats, flow_feats, video_names = [], [], [], []
        for tmp in batch:
            captions.append(tmp[0])
            rgb_feats.append(tmp[1])
            flow_feats.append(tmp[2])
            video_names.append(tmp[3])
        captions = torch.stack(tuple(captions), dim=0).to(self.device)
        rgb_feats = torch.stack(tuple(rgb_feats), dim=0).to(self.device)
        flow_feats = torch.stack(tuple(flow_feats), dim=0).to(self.device)
        return captions, rgb_feats, flow_feats, video_names


class MSRVTTDataset(Dataset):

    def __init__(self, device, cap_path, trainval_meta_path, test_meta_path,
                 video_path, start_token, end_token, pad_token, mode):
        self.device = device
        self.start_token = start_token
        self.end_token = end_token
        self.pad_token = pad_token
        self.specials = ['<unk>', pad_token, start_token, end_token]

        # Caption Data
        tv_f = open(trainval_meta_path, 'r')
        t_f = open(test_meta_path)
        self.tvds = json.load(tv_f)
        self.tds = json.load(t_f)
        tv_f.close()
        t_f.close()
        if mode == 'train' or mode == 'validate':
            self.ds = self.tvds
        elif mode == 'test':
            self.ds = self.tds
        all_tvcap = []
        all_tcap = []
        for tv_cap, t_cap in zip(self.tvds['sentences'], self.tds['sentences']):
            all_tvcap.append(tv_cap['caption'])
            all_tcap.append(t_cap['caption'])
        self.all_cap = all_tvcap + all_tcap
        self.cap_ds = pd.read_csv(cap_path, names=[0,1])
        # cap_ds = pd.read_csv(cap_path, names=[0,1])
        
        self.token_cap, self.all_vocab = generate_token(self.cap_ds[1], self.all_cap, self.specials)
        # self.all_token_cap, self.all_vocab = generate_token(cap_ds[1], self.specials)
        self.voc_size = len(self.all_vocab)
        self.pad_idx = self.all_vocab.get_stoi()[pad_token]
        self.start_idx = self.all_vocab.get_stoi()[start_token]
        self.end_idx = self.all_vocab.get_stoi()[end_token]
        self.train_video_ids = []
        self.val_video_ids = []
        self.test_video_ids = []
        for sp in self.ds['videos']:
            if sp['split'] == 'train':
                self.train_video_ids.append(sp['video_id'])
            elif sp['split'] == 'validate':
                self.val_video_ids.append(sp['video_id'])
            elif sp['split'] == 'test':
                self.test_video_ids.append(sp['video_id'])
        if mode == 'train':
            self.video_list = self.train_video_ids
        elif mode == 'validate':
            self.video_list = self.val_video_ids
        elif mode == 'test':
            self.video_list = self.test_video_ids

        # RGB and Flow feature Data
        video_feat = h5py.File(video_path)
        if (mode == 'train') or (mode == 'validate'):
            self.rgb_feat = video_feat['msrvtt/trainval/rgb']
            self.flow_feat = video_feat['msrvtt/trainval/flow']
        elif mode == 'test':
            self.rgb_feat = video_feat['msrvtt/test/rgb']
            self.flow_feat = video_feat['msrvtt/test/flow']
        rgb_stack, flow_stack = [], []
        for v_name in self.video_list:
            rgb = torch.from_numpy(self.rgb_feat[f'{v_name}/rgb_feat'][:]).float()
            flow = torch.from_numpy(self.flow_feat[f'{v_name}/flow_feat'][:]).float()
            # rgb = torch.from_numpy(self.rgb_feat[f'{v_name}/rgb_feat'][:]).float().to(self.device)
            # flow = torch.from_numpy(self.flow_feat[f'{v_name}/flow_feat'][:]).float().to(self.device)
            rgb_stack.append(rgb)
            flow_stack.append(flow)
        self.rgb_stack = pad_sequence(rgb_stack, batch_first=True, padding_value=self.pad_idx)
        self.flow_stack = pad_sequence(flow_stack, batch_first=True, padding_value=0)
        print(f'Created MSR-VTT {mode} Dataset.')

    def __getitem__(self, index):
        video_name = self.cap_ds[0][index]
        caption = self.token_cap[index]
        feat_index = self.video_list.index(video_name)
        rgb_feat = self.rgb_stack[feat_index].to(self.device)
        flow_feat = self.flow_stack[feat_index].to(self.device)
        return caption, rgb_feat, flow_feat, video_name

    def __len__(self):
        return len(self.cap_ds)

    def collate_batch(self, batch):
        captions, rgb_feats, flow_feats, video_names = [], [], [], []
        for tmp in batch:
            captions.append(tmp[0])
            rgb_feats.append(tmp[1])
            flow_feats.append(tmp[2])
            video_names.append(tmp[3])
        # captions = torch.stack(tuple(captions), dim=0)
        # rgb_feats = torch.stack(tuple(rgb_feats), dim=0)
        # flow_feats = torch.stack(tuple(flow_feats), dim=0)
        captions = torch.stack(tuple(captions), dim=0).to(self.device)
        rgb_feats = torch.stack(tuple(rgb_feats), dim=0).to(self.device)
        flow_feats = torch.stack(tuple(flow_feats), dim=0).to(self.device)
        return captions, rgb_feats, flow_feats, video_names
