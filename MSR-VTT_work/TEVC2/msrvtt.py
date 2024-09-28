import pandas as pd
import json

train_path = './data/msrvtt/train_val_videodatainfo.json'
test_path = './data/msrvtt/test_videodatainfo.json'

f = open(train_path, 'r')
ds = json.load(f)
f.close()

train_video_ids = []
val_video_ids = []
for sp in ds['videos']:
    if sp['split'] == 'train':
        train_video_ids.append(sp['video_id'])
    elif sp['split'] == 'validate':
        val_video_ids.append(sp['video_id'])

# print(train_video_ids)

train_data = []
for id in train_video_ids:
    for sentence in ds['sentences']:
        if sentence['video_id'] == id:
            train_data.append([id, sentence['caption']])
val_data = []
for id in val_video_ids:
    for sentence in ds['sentences']:
        if sentence['video_id'] == id:
            val_data.append([id, sentence['caption']])

train_df = pd.DataFrame(data=train_data)
val_df = pd.DataFrame(data=val_data)
train_df.to_csv('./data/msrvtt/train_caption.csv')
val_df.to_csv('./data/msrvtt/val_caption.csv')
print(train_df)
print('----------------------------------------------')
print(val_df)

s = open(test_path, 'r')
test_ds = json.load(s)
s.close()

test_video_ids = []
for sp in test_ds['videos']:
    test_video_ids.append(sp['video_id'])

test_data = []
for id in test_video_ids:
    for sentence in test_ds['sentences']:
        if sentence['video_id'] == id:
            test_data.append([id, sentence['caption']])

test_df = pd.DataFrame(data=test_data)
test_df.to_csv('./data/msrvtt/test_caption.csv')
print('----------------------------------------------')
print(test_df)

