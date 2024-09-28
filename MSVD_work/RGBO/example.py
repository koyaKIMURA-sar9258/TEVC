import os
import argparse
from time import strftime, localtime
from shutil import copytree, ignore_patterns

import numpy as np
import h5py
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.tensorboard.writer import SummaryWriter

from utils.utils import msvd_dataset, msrvtt_dataset
from model.transformer import VideoCaptionTransoformer
from loss.loss import LabelSmoothing, SimpleLossCompute
from scheduler.lr_scheduler import SimpleScheduler
from model.transformer import mask
from loop.run import training_loop, save_model, predict_1by1_for_TBoard, greedy_decoder

from utils.utils import timer
from tqdm import tqdm

class Config(object):
    
    def __init__(self, args):
        self.device = args.device
        self.dataset_name = args.dataset_name
        self.msvd_path = args.msvd_path
        self.msvd_cap_path = args.msvd_cap_path
        self.msrvtt_path = args.msrvtt_path
        self.msrvtt_trainval_meta_path = args.msrvtt_trainval_meta_path
        self.msrvtt_test_meta_path = args.msrvtt_test_meta_path
        self.start_token = args.start_token
        self.end_token = args.end_token
        self.pad_token = args.pad_token
        self.max_len = args.max_len
        self.min_freq = args.min_freq
        self.dout_p = args.dout_p
        self.N = args.N
        self.d_model = args.d_model
        self.H = args.H
        self.d_ff = args.d_ff
        self.B = args.B
        self.inf_B_coeff = args.inf_B_coeff
        self.start_epoch = args.start_epoch
        self.epoch_num = args.epoch_num
        # self.one_by_one_starts_at = args.one_by_one_start_at
        self.early_stop_after = args.early_stop_after
        self.criterion = args.criterion
        self.smoothing = args.smoothing
        self.optimizer = args.optimizer
        self.betas = args.betas
        self.eps = args.eps
        self.scheduler = args.scheduler
        self.lr_coeff = args.lr_coeff
        self.to_log = args.to_log
        self.log_dir = args.log_dir
        self.lr = args.lr
        self.msvd_videos_to_monitor = args.msvd_videos_to_monitor

        self.curr_time = strftime('%y%m%d%H%M%S', localtime())
        if args.to_log:
            self.log_dir = args.log_dir
            self.checkpoint_dir = self.log_dir
            exper_name = self.curr_time[2:]
            self.log_path = os.path.join(self.log_dir, exper_name)
            self.model_checkpoint_path = os.path.join(self.checkpoint_dir, exper_name)
        else:
            self.log_dir = None
            self.log_path = None
    
    def get_params(self, out_type):
        
        if out_type == 'md_table':
            table  = '| Parameter | Value | \n'
            table += '|-----------|-------| \n'

            for par, val in vars(self).items():
                table += f'| {par} | {val}| \n'

            return table
        
        elif out_type == 'dict':
            params_to_filter = [
                'model_checkpoint_path', 'log_path', 'comment', 'curr_time', 
                'checkpoint_dir', 'log_dir', 'videos_to_monitor', 'to_log', 
                'verbose_evaluation', 'tIoUs', 'reference_paths', 
                'one_by_one_starts_at', 'device', 'device_ids', 'pad_token',
                'end_token', 'start_token', 'val_1_meta_path', 'video_feature_name',
                'val_2_meta_path', 'train_meta_path', 'betas', 'path'
            ]
            dct = vars(self)
            dct = {k: v for k, v in dct.items() if (k not in params_to_filter) and (v is not None)}
            
            return dct
    
    def self_copy(self):
        
        if self.to_log:
            # let it be in method's arguments (for TBoard)
            self.path = os.path.realpath(__file__)
            pwd = os.path.split(self.path)[0]
            cp_path = os.path.join(self.model_checkpoint_path, 'wdir_copy')
            copytree(pwd, cp_path, ignore=ignore_patterns('todel', 'submodules', '.git'))





def main(cfg):

    if cfg.dataset_name == 'msvd':
        train_ds, test_ds, train_loader, test_loader = msvd_dataset(cfg)
    elif cfg.dataset_name == 'msrvtt':
        train_ds, test_ds, train_loader, test_loader = msrvtt_dataset(cfg)

    model = VideoCaptionTransoformer(
        cfg.d_model, cfg.dout_p, train_ds.voc_size,
        cfg.H, cfg.d_ff, cfg.N
    )
    
    creterion = LabelSmoothing(cfg.smoothing, train_ds.pad_idx)
    optimizer = torch.optim.Adam(
        model.parameters(), 0, (cfg.betas[0], cfg.betas[1]), cfg.eps
    )
    lr_scheduler = SimpleScheduler(optimizer, cfg.lr)
    loss_compute = SimpleLossCompute(creterion, lr_scheduler)

    model.to(cfg.device)
    param_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Param Num: {param_num}')

    if cfg.to_log:
        os.makedirs(cfg.log_path)
        os.makedirs(cfg.model_checkpoint_path, exist_ok=True)# handles the case when model_checkpoint_path = log_path
        TBoard = SummaryWriter(log_dir=cfg.log_path)
        TBoard.add_text('config', cfg.get_params('md_table'), 0)
        TBoard.add_scalar('debug/param_number', param_num, 0)
    else:
        TBoard = None

    # keeping track of the best model 
    best_metric = 0
    # "early stopping" thing
    num_epoch_best_metric_unchanged = 0

    count = 0
    for epoch in range(cfg.start_epoch, cfg.epoch_num):
        if count == 5:
            break
        num_epoch_best_metric_unchanged += 1
        if (num_epoch_best_metric_unchanged == cfg.early_stop_after) or (timer(cfg.curr_time) > 67):
            print(f'Early stop at {epoch}: unchanged for {num_epoch_best_metric_unchanged} epochs')
            print(f'Current timer: {timer(cfg.curr_time)}')
            break
        
        # Train
        training_loop(model, train_loader, loss_compute, lr_scheduler, epoch, TBoard)
        

        ##############################
        # Eval
        # print('---------Eval---------')
        # model.eval()
        # losses = []

        # for i, batch in enumerate(tqdm(test_loader, desc=f'{strftime("%X", localtime())} valid ({epoch})')):
        #     video_names, captions, rgb_feats, flow_feats = batch
        #     n_tokens = (captions != test_loader.dataset.pad_idx).sum()
        #     masks = mask(rgb_feats[:, :, 0], captions, test_loader.dataset.pad_idx)
            
        #     with torch.no_grad():
        #         pred = model(rgb_feats, captions, masks)
        #         loss_iter = loss_compute.criterion(pred, captions)
        #         loss_iter_norm = loss_iter / n_tokens
        #         losses.append(loss_iter_norm.item())

        # loss_total_norm = np.sum(losses) / len(test_loader)
        # text, preds = predict_1by1_for_TBoard(
        #     cfg.msvd_videos_to_monitor, test_loader,
        #     greedy_decoder, model, cfg.max_len
        # )
        # print(preds)
        # print(preds[:, -1].max(dim=-1)[1].unsqueeze(1))
        # count += 1
    torch.save(model.state_dict(), './model_weight.pth')
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run experiment')
    parser.add_argument(
        '--device', default=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    )
    parser.add_argument(
        '--dataset_name', required=True
    )
    parser.add_argument(
        '--msvd_path', default='./data/msvd/msvd.h5'
    )
    parser.add_argument(
        '--msvd_cap_path', default='./data/msvd/captions.csv'
    )
    parser.add_argument(
        '--msrvtt_path', default='./data/msrvtt/msrvtt.h5'
    )
    parser.add_argument(
        '--msrvtt_trainval_meta_path', default='./data/msrvtt/train_val_videodatainfo.json'
    )
    parser.add_argument(
        '--msrvtt_test_meta_path', default='./data/msrvtt/test_videodatainfo.json'
    )
    parser.add_argument(
        '--start_token', dest='start_token', default='<start>',
    )
    parser.add_argument(
        '--end_token', dest='end_token', default='<end>',
    )
    parser.add_argument(
        '--pad_token', dest='pad_token', default='<pad>',
    )
    parser.add_argument(
        '--max_len', type=int, default=50,
        help='maximum size of 1by1 prediction'
    )
    parser.add_argument(
        '--min_freq', type=int, default=1,
        help='to be in the vocab a word should appear min_freq times in train dataset'
    )
    parser.add_argument('--dout_p', type=float, default=0.1)
    parser.add_argument('--N', type=int, default=1, help='number of layers in a model')
    parser.add_argument(
        '--d_model', type=int, default=1024,
        help='If use_linear_embedder is true, this is going to be the d_model size for video model'
    )
    parser.add_argument(
        '--H', type=int, default=4,
        help='number of heads in multiheaded attention in Transformer'
    )
    parser.add_argument(
        '--d_ff', type=int, default=2048,
        help='size of the internal layer of PositionwiseFeedForward net in Transformer (Video)'
    )
    parser.add_argument(
        '--B', type=int, default=28,
        help='batch size per a device'
    )
    parser.add_argument(
        '--inf_B_coeff', type=int, default=2,
        help='the batch size on inference is inf_B_coeff times the B'
    )
    parser.add_argument(
        '--start_epoch', type=int, default=0, choices=[0],
        help='the epoch number to start training (if specified, pretraining a net from start_epoch epoch)'
    )
    parser.add_argument(
        '--epoch_num', type=int, default=45,
        help='number of epochs to train'
    )
    parser.add_argument(
        '--one_by_one_starts_at', type=int, default=0,
        help='number of epochs to skip before starting 1-by-1 validation'
    )
    parser.add_argument(
        '--early_stop_after', type=int, default=50,
        help='number of epochs to wait for best metric to change before stopping'
    )
    parser.add_argument(
        '--criterion', type=str, default='label_smoothing', choices=['label_smoothing'],
        help='criterion to measure the loss'
    )
    parser.add_argument(
        '--smoothing', type=float, default=0.7,
        help='smoothing coeff (= 0 cross ent loss; -> 1 more smoothing, random labels) must be in [0, 1]'
    )
    parser.add_argument(
        '--optimizer', type=str, default='adam', choices=['adam'],
        help='optimizer'
    )
    parser.add_argument(
        '--betas', type=float, nargs=2, default=[0.9, 0.98],
        help='beta 1 and beta 2 parameters in adam'
    )
    parser.add_argument(
        '--eps', type=float, default=1e-8,
        help='eps parameter in adam'
    )
    parser.add_argument(
        '--scheduler', type=str, default='constant', choices=['attention_is_all_you_need', 'constant'], 
        help='lr scheduler'
    )
    parser.add_argument(
        '--lr_coeff', type=float,  
        help='lr scheduler coefficient (if scheduler is attention_is_all_you_need)'
    )
    parser.add_argument(
        '--dont_log', dest='to_log', action='store_false', 
        help='Prevent logging in the experiment.'
    )
    parser.add_argument(
        '--msvd_videos_to_monitor', type=str, nargs='+',
        default=['zv2RIbUsnSw_335_341', 'zxB4dFJhHR8_1_9', 'zzit5b_-ukg_5_20'],
        help='the videos to monitor on validation loop with 1 by 1 prediction'
    )
    parser.add_argument('--log_dir', type=str, default='./log/')
    parser.add_argument('--lr', type=float, default=1e-5, help='lr (if scheduler is constant)')
    parser.set_defaults(to_log=True)
    args = parser.parse_args()
    cfg = Config(args)
    main(cfg)
