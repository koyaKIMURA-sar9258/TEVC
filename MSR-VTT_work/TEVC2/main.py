import os
import argparse
import json
from time import strftime, localtime
from shutil import copytree, ignore_patterns

import numpy as np
import h5py
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from dataset.dataset import MSVDDataset, MSRVTTDataset
from model.transformer import TEVCTransformer, RGBOnlyTransformer
from loss.loss import LabelSmoothing, SimpleLossCompute
from scheduler.lr_scheduler import SimpleScheduler
from loop.run import training, validation, test

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
        self.msrvtt_train_cap_path = args.msrvtt_train_cap_path
        self.msrvtt_val_cap_path = args.msrvtt_val_cap_path
        self.msrvtt_test_cap_path = args.msrvtt_test_cap_path
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
        self.msvd_val_videos_to_monitor = args.msvd_val_videos_to_monitor
        self.msrvtt_val_videos_to_monitor = args.msrvtt_val_videos_to_monitor
        self.use_flow = args.use_flow
        self.interruption = args.interruption

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
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    torch.cuda.set_device(cfg.device)
    if cfg.dataset_name == 'msvd':
        train_ds = MSVDDataset(
            cfg.device, cfg.msvd_cap_path, cfg.start_token, cfg.end_token,
            cfg.pad_token, 55884, 71860, cfg.msvd_path, 'train'
        )
        val_ds = MSVDDataset(
            cfg.device, cfg.msvd_cap_path, cfg.start_token, cfg.end_token,
            cfg.pad_token, 55884, 71860, cfg.msvd_path, 'validate'
        )
        test_ds = MSVDDataset(
            cfg.device, cfg.msvd_cap_path, cfg.start_token, cfg.end_token,
            cfg.pad_token, 55884, 71860, cfg.msvd_path, 'test'
        )
        train_loader = DataLoader(train_ds, cfg.B, shuffle=True, collate_fn=train_ds.collate_batch)
        # val_loader = DataLoader(val_ds, collate_fn=val_ds.collate_batch)
        # test_loader = DataLoader(test_ds, collate_fn=test_ds.collate_batch)
        val_loader = DataLoader(val_ds, cfg.B, collate_fn=train_ds.collate_batch)
        test_loader = DataLoader(test_ds, cfg.B, collate_fn=train_ds.collate_batch)

    elif cfg.dataset_name == 'msrvtt':
        train_ds = MSRVTTDataset(
            cfg.device, cfg.msrvtt_train_cap_path, cfg.msrvtt_trainval_meta_path, cfg.msrvtt_test_meta_path,
            cfg.msrvtt_path, cfg.start_token, cfg.end_token, cfg.pad_token, 'train'
        )
        val_ds = MSRVTTDataset(
            cfg.device, cfg.msrvtt_val_cap_path, cfg.msrvtt_trainval_meta_path, cfg.msrvtt_test_meta_path,
            cfg.msrvtt_path, cfg.start_token, cfg.end_token, cfg.pad_token, 'validate'
        )
        test_ds = MSRVTTDataset(
            cfg.device, cfg.msrvtt_test_cap_path, cfg.msrvtt_trainval_meta_path, cfg.msrvtt_test_meta_path,
            cfg.msrvtt_path, cfg.start_token, cfg.end_token, cfg.pad_token, 'test'
        )
        train_loader = DataLoader(train_ds, cfg.B, shuffle=True, collate_fn=train_ds.collate_batch)
        # val_loader = DataLoader(val_ds, collate_fn=train_ds.collate_batch)
        # test_loader = DataLoader(test_ds, collate_fn=train_ds.collate_batch)
        val_loader = DataLoader(val_ds, cfg.B, collate_fn=train_ds.collate_batch)
        test_loader = DataLoader(test_ds, cfg.B, collate_fn=train_ds.collate_batch)

    if cfg.use_flow:
        model = TEVCTransformer(
            train_ds.voc_size, cfg.d_model, cfg.H, cfg.N,
            cfg.d_ff, cfg.dout_p, seq_len=100
        )
    else:
        model = RGBOnlyTransformer(
            train_ds.voc_size, cfg.d_model, cfg.H, cfg.N,
            cfg.d_ff, cfg.dout_p, seq_len=100
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
        os.makedirs(cfg.model_checkpoint_path, exist_ok=True)
        TBoard = SummaryWriter(log_dir=cfg.log_path)
        TBoard.add_text('config', cfg.get_params('md_table'), 0)
        TBoard.add_scalar('debug/param_number', param_num, 0)
    else:
        TBoard = None
    
    ave_best_metric = 0.
    max_best_metric = 0.
    ave_best_epoch = 0
    max_best_epoch = 0
    num_epoch_best_metric_unchanged = 0
    val_epoch_losses = []
    val_avemeteor_epoch = []
    val_maxmeteor_epoch = []
    count = 0

    if cfg.interruption is True:
        model.load_state_dict(torch.load('./results/model_param/msrvtt/training/train_5.pth'))
        vocab = torch.load(f'./results/{cfg.dataset_name}_vocab.pt')
    else:
        vocab = train_ds.all_vocab
        torch.save(vocab, f'./results/{cfg.dataset_name}_vocab.pt')
    
    # for epoch in range(cfg.start_epoch, cfg.epoch_num):
    #     # if count == 3:
    #     #     break
    #     print(f'####### Epoch {epoch} #######')
    #     num_epoch_best_metric_unchanged += 1
    #     if (num_epoch_best_metric_unchanged == cfg.early_stop_after) or (timer(cfg.curr_time) > 67):
    #         print(f'Early stop at {epoch}: unchanged for {num_epoch_best_metric_unchanged} epochs')
    #         print(f'Current timer: {timer(cfg.curr_time)}')
    #         break
        
    #     # Train
    #     print('--- Training ---')
    #     training(model, train_loader, loss_compute, lr_scheduler,
    #              epoch, TBoard, cfg.dataset_name, cfg.use_flow)

    #     # Validation
    #     print('--- Validation ---')
    #     if cfg.dataset_name == 'msvd':
    #         val_loss, ave_meteor, max_meteor = validation(model, val_loader, loss_compute, lr_scheduler,epoch, TBoard,
    #                                                   cfg.msvd_val_videos_to_monitor, cfg.dataset_name, vocab, cfg.use_flow)
    #     elif cfg.dataset_name == 'msrvtt':
    #         val_loss, ave_meteor, max_meteor = validation(model, val_loader, loss_compute, lr_scheduler,epoch, TBoard,
    #                                                   cfg.msrvtt_val_videos_to_monitor, cfg.dataset_name, vocab, cfg.use_flow)

        # if ave_best_metric < ave_meteor:
        #     ave_best_metric = ave_meteor
        #     ave_best_epoch = epoch
        #     best_model_path = f'./results/model_param/{cfg.dataset_name}/validation/val_{epoch}.pth'
        #     torch.save(model.state_dict(), best_model_path)
        # if max_best_metric < max_meteor:
        #     max_best_metric = max_meteor
        #     max_best_epoch = epoch
        # val_epoch_losses.append(val_loss)
        # val_avemeteor_epoch.append(ave_meteor)
        # val_maxmeteor_epoch.append(max_meteor)
        # count += 1

    print('----- Test Start -----')
    max_metrics , ave_metrics = test(model, test_loader, cfg.dataset_name, 0, TBoard, vocab, cfg.use_flow, ave_best_epoch)
    print('\nTest Dataset results')
    print('Max Meteor: ', max_metrics[0]*100)
    print('Max Bleu@3: ', max_metrics[1]*100)
    print('Max Bleu@4: ', max_metrics[2]*100)
    print('Average Meteor: ', ave_metrics[0]*100)
    print('Average Bleu@3: ', ave_metrics[1]*100)
    print('Average Bleu@4: ', ave_metrics[2]*100)
    
    # if TBoard is not None:
    #     TBoard.add_scalar(f'test/max/meteor', max_metrics[0] * 100, 0)
    #     TBoard.add_scalar(f'test/max/bleu3',  max_metrics[1] * 100, 0)
    #     TBoard.add_scalar(f'test/max/bleu4',  max_metrics[2] * 100, 0)
    #     TBoard.add_scalar(f'test/max/cider',  max_metrics[3] * 100, 0)
    #     TBoard.add_scalar(f'test/average/meteor',  ave_metrics[0] * 100, 0)
    #     TBoard.add_scalar(f'test/average/meteor', ave_metrics[0] * 100, 0)
    #     TBoard.add_scalar(f'test/average/meteor', ave_metrics[0] * 100, 0)
    #     TBoard.add_scalar(f'test/average/meteor', ave_metrics[0] * 100, 0)


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
        '--msrvtt_train_cap_path', default='./data/msrvtt/train_caption.csv'
    )
    parser.add_argument(
        '--msrvtt_val_cap_path', default='./data/msrvtt/val_caption.csv'
    )
    parser.add_argument(
        '--msrvtt_test_meta_path', default='./data/msrvtt/test_videodatainfo.json'
    )
    parser.add_argument(
        '--msrvtt_test_cap_path', default='./data/msrvtt/test_caption.csv'
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
        '--max_len', type=int, default=20,
        help='maximum size of 1by1 prediction'
    )
    parser.add_argument(
        '--min_freq', type=int, default=1,
        help='to be in the vocab a word should appear min_freq times in train dataset'
    )
    parser.add_argument('--dout_p', type=float, default=0.1)
    parser.add_argument('--N', type=int, default=6, help='number of layers in a model')
    parser.add_argument(
        '--d_model', type=int, default=1024,
        help='If use_linear_embedder is true, this is going to be the d_model size for video model'
    )
    parser.add_argument(
        '--H', type=int, default=8,
        help='number of heads in multiheaded attention in Transformer'
    )
    parser.add_argument(
        '--d_ff', type=int, default=2048,
        help='size of the internal layer of PositionwiseFeedForward net in Transformer (Video)'
    )
    parser.add_argument(
        '--B', type=int, default=128,
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
        '--epoch_num', type=int, default=50,
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
        '--msvd_val_videos_to_monitor', type=str, nargs='+',
        default=['hNOzHvsEmg4_31_36', 'hNPZmTlY_3Q_0_8', 'hPyU5KjpWVc_0_35'],
        help='the videos to monitor on validation loop with 1 by 1 prediction'
    )
    parser.add_argument(
        '--msrvtt_val_videos_to_monitor', type=str, nargs='+',
        default=['video6513', 'video6514', 'video6515'],
        help='the videos to monitor on validation loop with 1 by 1 prediction'
    )
    parser.add_argument(
        '--use_flow', action='store_false'
    )
    parser.add_argument(
        '--interruption', action='store_true'
    )
    parser.add_argument('--log_dir', type=str, default='./log/')
    parser.add_argument('--lr', type=float, default=1e-5, help='lr (if scheduler is constant)')
    parser.set_defaults(to_log=True)
    args = parser.parse_args()
    cfg = Config(args)
    main(cfg)