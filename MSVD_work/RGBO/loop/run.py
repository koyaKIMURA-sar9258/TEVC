import os
import json
from tqdm import tqdm
import numpy as np
import torch
import spacy
from time import time, strftime, localtime

from model.transformer import mask
from evaluate.evaluate import calc_metrics, calculate_all_metrics
from utils.utils import HiddenPrints

def greedy_decoder(model, rgb_src, flow_src, max_len, start_idx, end_idx, pad_idx):
    completeness_mask = torch.zeros(len(rgb_src), 1).byte().to(rgb_src.device)
    with torch.no_grad():
        B, S = rgb_src.size(0), rgb_src.size(1)
        trg = (torch.ones(B, 1) * start_idx).type_as(rgb_src).long()

        while (trg.size(-1) <= max_len) and (not completeness_mask.all()):
            rgb_masks = mask(rgb_src[:, :, 0], trg, pad_idx)
            flow_masks = mask(flow_src[:, :, 0], trg, pad_idx)
            masks = (rgb_masks[0], flow_masks[0], rgb_masks[1])
            preds = model(rgb_src, flow_src, trg, masks)
            next_word = preds[:, -1].max(dim=-1)[1].unsqueeze(1)
            trg = torch.cat([trg, next_word], dim=-1)
            # sum two masks (or adding 1s where the ending token occured)
            completeness_mask = completeness_mask | torch.eq(next_word, end_idx).byte()
    
    return trg

def rgb_greedy_decoder(model, rgb_src, max_len, start_idx, end_idx, pad_idx):
    completeness_mask = torch.zeros(len(rgb_src), 1).byte().to(rgb_src.device)
    with torch.no_grad():
        B, S = rgb_src.size(0), rgb_src.size(1)
        trg = (torch.ones(B, 1) * start_idx).type_as(rgb_src).long()

        while (trg.size(-1) <= max_len) and (not completeness_mask.all()):
            rgb_masks = mask(rgb_src[:, :, 0], trg, pad_idx)
            # masks = (rgb_masks[0], flow_masks[0], rgb_masks[1])
            preds = model(rgb_src, trg, rgb_masks[0], rgb_masks[1])
            next_word = preds[:, -1].max(dim=-1)[1].unsqueeze(1)
            trg = torch.cat([trg, next_word], dim=-1)
            # sum two masks (or adding 1s where the ending token occured)
            completeness_mask = completeness_mask | torch.eq(next_word, end_idx).byte()
    
    return trg

def predict_TBoard(video_list, loader, model, vocab, use_flow, max_len=30):
    model.eval()
    meta = loader.dataset.cap_ds
    ds = loader.dataset
    start_idx = loader.dataset.start_idx
    end_idx = loader.dataset.end_idx
    pad_idx = loader.dataset.pad_idx
    text = ''
    texts = []
    for video_name in video_list:
        text += f'\t {video_name} \n'
        vid_index = meta[0][meta[0]==video_name].index[0]
        _, rgb_feat, flow_feat, v_name = ds[vid_index]
        assert v_name == video_name
        rgb_feat = rgb_feat.unsqueeze(0)
        flow_feat = flow_feat.unsqueeze(0)
        if use_flow == True:
            trg_ints = greedy_decoder(model, rgb_feat, flow_feat, max_len,
                                        start_idx, end_idx, pad_idx)
        elif use_flow == False:
            trg_ints = rgb_greedy_decoder(model, rgb_feat, max_len, start_idx,
                                          end_idx, pad_idx)
        trg_ints = trg_ints.cpu().numpy()[0]
        trg_words = [vocab.get_itos()[i] for i in trg_ints]
        en_sent = ' '.join(trg_words)
        texts.append(en_sent)
        text += f'\t P sent: {en_sent}'
        text += '\t  \n'
    return text, texts


# training function
def training(model, loader, loss_compute, lr_scheduler, epoch, TBoard, dataset_name, use_flow):

    model.train()
    losses = []
    time = strftime('%X', localtime())
    train_results_path = f'./results/model_param/{dataset_name}/training/train_{epoch}.pth'
    count = 0

    for i, batch in enumerate(tqdm(loader, desc=f'{time} train ({epoch})')):
        # if count == 10:
        #     break
        captions, rgb_feats, flow_feats, _ = batch
        captions, captions_idxy = captions[:, :-1], captions[:, 1:]
        rgb_masks = mask(rgb_feats[:, :, 0], captions, loader.dataset.pad_idx)
        n_token = (captions[:, 1:] != loader.dataset.pad_idx).sum()
        if use_flow is True:
            flow_masks = mask(flow_feats[:, :, 0], captions, loader.dataset.pad_idx)
            # (src_rgb_mask, src_flow_mask, trg_mask)
            masks = (rgb_masks[0], flow_masks[0], rgb_masks[1])
            pred = model(rgb_feats, flow_feats, captions, masks)
        elif use_flow is False:
            pred = model(rgb_feats, captions, rgb_masks[0], rgb_masks[1])
        
        loss_iter = loss_compute(pred, captions_idxy, n_token)
        loss_iter_norm = loss_iter / n_token
        losses.append(loss_iter_norm.item())

        if TBoard is not None:
            step_num = epoch * len(loader) + i
            TBoard.add_scalar('train/Loss_iter', loss_iter_norm.item(), step_num)
            TBoard.add_scalar('debug/lr', lr_scheduler.get_lr(), step_num)
        
        # count += 1
        

    
    loss_total_norm = np.sum(losses) / len(loader)

    if TBoard is not None:
        TBoard.add_scalar('debug/train_loss_epoch', loss_total_norm, epoch)
    # if epoch % 5 == 0:
    torch.save(model.state_dict(), train_results_path)

# validation function
def validation(model, loader, loss_compute, lr_scheduler, epoch,
               TBoard, videos_to_monitor, dataset_name, vocab, use_flow):

    model.eval()
    losses = []
    time = strftime('%X', localtime())
    # val_results_path = f'./results/model_param/{dataset_name}/validation/val_{epoch}.pth'
    count = 0
    max_meteor_batch, ave_meteor_batch = [], []
    for i, batch in enumerate(tqdm(loader, desc=f'{time} eval ({epoch})')):
        # if count == 10:
        #     break
        captions, rgb_feats, flow_feats, video_names = batch
        captions, captions_idxy = captions[:, :-1], captions[:, 1:]
        rgb_masks = mask(rgb_feats[:, :, 0], captions, loader.dataset.pad_idx)
        n_token = (captions_idxy != loader.dataset.pad_idx).sum()
        flow_masks = mask(flow_feats[:, :, 0], captions, loader.dataset.pad_idx)
        masks = (rgb_masks[0], flow_masks[0], rgb_masks[1])
        max_meteors, ave_meteors = [], []

        with torch.no_grad():
            if use_flow is True:
                pred = model(rgb_feats, flow_feats, captions, masks)
            elif use_flow is False:
                pred = model(rgb_feats, captions, rgb_masks[0], rgb_masks[1])
            loss_iter = loss_compute.criterion(pred, captions_idxy)
            loss_iter_norm = loss_iter / n_token
            losses.append(loss_iter_norm.item())

            # text, texts = predict_TBoard(video_names, loader, model)
            # meteor_scores = calculate_metrics(texts, loader, video_names)
            # max_meteor, ave_meteor = meteor_scores
            # max_meteors.append(max_meteor)
            # ave_meteors.append(ave_meteor)

            if TBoard is not None:
                step_num = epoch * len(loader)
                TBoard.add_scalar(f'debug/validation/loss_iter', loss_iter_norm.item(), step_num)
                text, texts = predict_TBoard(video_names, loader, model, vocab, use_flow)
                max_meteor, ave_meteor = calc_metrics(texts, loader, video_names)
                max_meteor_batch.append(max_meteor)
                ave_meteor_batch.append(ave_meteor)
        # count += 1
      
    loss_total_norm = np.sum(losses) / len(loader)
    max_meteor_epoch = max(max_meteor_batch)
    ave_meteor_epoch = np.sum(max_meteor_batch) / len(max_meteor_batch)
    if TBoard is not None:
        TBoard.add_scalar('validation/Loss_epoch', loss_total_norm, epoch)
        TBoard.add_scalar('validation/max_meteor', max_meteor_epoch*100, epoch)
        TBoard.add_scalar('validation/ave_meteor', ave_meteor_epoch*100, epoch)
        text, _ = predict_TBoard(videos_to_monitor, loader, model, vocab, use_flow)
        TBoard.add_text('prediction_val', text, epoch)
        print(f'Max meteor epoch{epoch}: ', max_meteor_epoch*100)
        print(f'Average meteor epoch{epoch}: ', ave_meteor_epoch*100)

    # torch.save(model.state_dict(), val_results_path)

    return loss_total_norm, max_meteor_epoch, ave_meteor_epoch


def test(model, loader, dataset_name, val_avemeteor_epoch, TBoard, vocab, use_flow, best_epoch):
    
    # best_param_epoch = val_avemeteor_epoch.index(max(val_avemeteor_epoch))
    best_model_path = f'./results/model_param/{dataset_name}/training/train_{best_epoch}.pth'
    time = strftime('%X', localtime())
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    max_meteors, max_bleu3s, max_bleu4s = [], [], []
    ave_meteors, ave_bleu3s, ave_bleu4s = [], [], []
    count = 0

    for i, batch in enumerate(tqdm(loader, desc=f'{time} test ')):
        #if count == 30:
        #    break
        _, _, _, video_names = batch
        # captions, rgb_feats, flow_feat, video_names = batch
        with torch.no_grad():
            text, texts = predict_TBoard(video_names, loader, model, vocab, use_flow)
            meteor_scores, bleu3_scores, bleu4_scores = calculate_all_metrics(texts, loader, video_names)
            max_meteor = max(meteor_scores)
            ave_meteor = np.sum(meteor_scores) / len(meteor_scores)
            max_bleu3 = max(bleu3_scores)
            ave_bleu3 = np.sum(bleu3_scores) / len(bleu3_scores)
            max_bleu4 = max(bleu4_scores)
            ave_bleu4 = np.sum(bleu4_scores) / len(bleu4_scores)
            max_meteors.append(max_meteor)
            ave_meteors.append(ave_meteor)
            max_bleu3s.append(max_bleu3)
            ave_bleu3s.append(ave_bleu3)
            max_bleu4s.append(max_bleu4)
            ave_bleu4s.append(ave_bleu4)
        if TBoard is not None:
            TBoard.add_text('test prediction text', text)
            TBoard.add_scalar('test/max_meteor', max_meteor*100, i)
            TBoard.add_scalar('test/ave_meteor', ave_meteor*100, i)
            TBoard.add_scalar('test/max_bleu3', max_bleu3*100, i)
            TBoard.add_scalar('test/ave_bleu3', ave_bleu3*100, i)
            TBoard.add_scalar('test/max_bleu4', max_bleu4*100, i)
            TBoard.add_scalar('test/ave_bleu4', ave_bleu4*100, i)
        #count += 1
    
    # Max and Average metrics for test dataset.
    all_max_meteor = max(max_meteors)
    all_max_bleu3 = max(max_bleu3s)
    all_max_bleu4 = max(max_bleu4s)
    # max_cider = max(max_ciders)
    all_ave_meteor = np.sum(ave_meteors) / len(ave_meteors)
    all_ave_bleu3 = np.sum(ave_bleu3s) / len(ave_bleu3s)
    all_ave_bleu4 = np.sum(ave_bleu4s) / len(ave_bleu4s)
    # ave_cider = np.sum(ave_ciders) / len(ave_ciders)
    
    max_metrics = (all_max_meteor, all_ave_bleu3, all_max_bleu4)
    ave_metrics = (all_ave_meteor, all_ave_bleu3, all_ave_bleu4)

    if TBoard is not None:
        TBoard.add_scalar(f'test/max/meteor', all_max_meteor * 100, 0)
        TBoard.add_scalar(f'test/max/bleu3',  all_ave_bleu3 * 100, 0)
        TBoard.add_scalar(f'test/max/bleu4',  all_max_bleu4 * 100, 0)
        # TBoard.add_scalar(f'test/max/cider',  max_metrics[2] * 100, 0)
        TBoard.add_scalar(f'test/average/meteor', all_ave_meteor * 100, 0)
        TBoard.add_scalar(f'test/average/bleu3', all_ave_bleu3 * 100, 0)
        TBoard.add_scalar(f'test/average/bleu4', all_ave_bleu4 * 100, 0)
        # TBoard.add_scalar(f'test/average/cider', ave_metrics[2] * 100, 0)
    
    return max_metrics, ave_metrics 
        
        
    
