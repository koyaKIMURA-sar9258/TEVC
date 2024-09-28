import h5py
import pandas as pd
from time import strptime, localtime, mktime
import sys
import os

from torch.utils.data import DataLoader
from dataset.dataset import MSVDDataset, MSRVTTDataset

def timer(timer_started_at):
    # 20190911133748 (YmdHMS) -> struct time
    timer_started_at = strptime(timer_started_at, '%y%m%d%H%M%S')
    # struct time -> secs from 1900 01 01 etc
    timer_started_at = mktime(timer_started_at)
    
    now = mktime(localtime())
    timer_in_hours = (now - timer_started_at) / 3600
    
    return round(timer_in_hours, 2)
    
class HiddenPrints:
    '''
    Used in 1by1 validation in order to block printing of the enviroment 
    which is surrounded by this class 
    '''
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def msvd_dataset(args):
    # MSVD Dataset
    print('create MSVD dataset and dataloader')
    v_feat = h5py.File(args.msvd_path)
    rgb_data = v_feat['msvd/rgb']
    flow_data = v_feat['msvd/flow']
    msvd_video_list = list(rgb_data.keys())
    msvd_video_name_list = msvd_video_list
    for key in msvd_video_list:
        if len(rgb_data[f'{key}/rgb_feat']) == 0:
            msvd_video_name_list.remove(key)
            print(f'remove {key}')
    video_num = len(msvd_video_name_list)
    print(f'Video num: {video_num}')
    print(f'List num: {len(msvd_video_name_list)}')
    rate = int(video_num*0.8)

    # train and test video names list
    train_video = msvd_video_name_list[:rate]
    test_video = msvd_video_name_list[rate:]

    # dataset
    train_ds = MSVDDataset(
        train_video, rgb_data, flow_data, args.msvd_cap_path, args.start_token,
        args.end_token, args.pad_token, args.dataset_name, args.device
    )
    test_ds = MSVDDataset(
        test_video, rgb_data, flow_data, args.msvd_cap_path,args.start_token,
        args.end_token, args.pad_token, args.dataset_name, args.device
    )

    # DataLoader
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, collate_fn=train_ds.collate_batch)
    test_loader  = DataLoader(test_ds, batch_size=128, shuffle=True, collate_fn=test_ds.collate_batch)
    print('Finished !')

    return train_ds, test_ds, train_loader, test_loader

def msrvtt_dataset(args):
    print('create MSR-VTT dataset and dataloader')
    train_ds = MSRVTTDataset(
        args.msrvtt_path, args.msrvtt_trainval_meta_path, 'trainval', 
        args.start_token, args.end_token, args.pad_token, args.dataset_name
    )
    test_ds = MSRVTTDataset(
        args.msrvtt_path, args.msrvtt_test_meta_path, 'test', 
        args.start_token, args.end_token, args.pad_token, args.dataset_name
    )
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, collate_fn=train_ds.collate_batch)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=True, collate_fn=test_ds.collate_batch)
    print('Finished !')
    
    return train_ds, test_ds, train_loader, test_loader
  