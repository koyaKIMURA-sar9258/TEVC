import sys
from tqdm import tqdm
sys.path.insert(0, './submodules/')

from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
import numpy as np

# The function is called in validation.
def calc_metrics(pred_texts, loader, video_list):
    dict_gt, dict_pred = {}, {}
    meta = loader.dataset.cap_ds
    for video_name, text in zip(video_list, pred_texts):
        text = text.replace('<start>', '')
        text = text.replace('<end>', '')
        gt_caps = list(meta[1][meta[0]==video_name])
        dict_gt[video_name] = gt_caps
        dict_pred[video_name] = [text]
    score, scores = Meteor().compute_score(dict_gt, dict_pred)
    try:
        max_meteor = max(scores) # max score in batch
        ave_meteor = sum(scores) / len(scores) # average score in batch
    except ValueError as e:
        print(e)
        pass
    return max_meteor, ave_meteor

def calculate_all_metrics(texts, loader, monitor_to_videos):
    dict_gt, dict_pred = {}, {}
    meta = loader.dataset.cap_ds
    
    for video_name, text in zip(monitor_to_videos, texts):
        text = text.replace('<start>', '')
        text = text.replace('<end>', '')
        gt_caps = list(meta[1][meta[0]==video_name])
        dict_gt[video_name] = gt_caps
        dict_pred[video_name] = [text]

    meteor_score, meteor_scores = Meteor().compute_score(dict_gt, dict_pred)
    bleu_score, bleu_scores = Bleu(4).compute_score(dict_gt, dict_pred)
    return meteor_scores, bleu_scores[2], bleu_scores[3]
