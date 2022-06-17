import re
import json
import os

import torch
import torch.distributed as dist

import utils

def pre_caption(caption,max_words=50):
    caption = re.sub(
        r"([.!\"()*#:;~])",       
        ' ',
        caption.lower(),
    )
    caption = re.sub(
        r"\s{2,}",
        ' ',
        caption,
    )
    caption = caption.rstrip('\n') 
    caption = caption.strip(' ')

    #truncate caption
    caption_words = caption.split(' ')
    if len(caption_words)>max_words:
        caption = ' '.join(caption_words[:max_words])
            
    return caption

def pre_question(question,max_ques_words=50):
    question = re.sub(
        r"([.!\"()*#:;~])",
        '',
        question.lower(),
    ) 
    question = question.rstrip(' ')
    
    #truncate question
    question_words = question.split(' ')
    if len(question_words)>max_ques_words:
        question = ' '.join(question_words[:max_ques_words])
            
    return question


def save_result(result, result_dir, filename, remove_duplicate=''):
    result_file = os.path.join(result_dir, '%s_rank%d.json'%(filename,utils.get_rank()))
    final_result_file = os.path.join(result_dir, '%s.json'%filename)
    
    json.dump(result, open(result_file,'w'))

#    dist.barrier()

    if utils.is_main_process():   
        # combine results from all processes
        result = []

        for rank in range(utils.get_world_size()):
            result_file = os.path.join(result_dir, '%s_rank%d.json'%(filename,rank))
            res = json.load(open(result_file,'r'))
            result += res

        if remove_duplicate:
            result_new = []
            id_list = []    
            for res in result:
                if res[remove_duplicate] not in id_list:
                    id_list.append(res[remove_duplicate])
                    result_new.append(res)
            result = result_new             
                
        json.dump(result,open(final_result_file,'w'))            
        print('result file saved to %s'%final_result_file)

    return final_result_file


import re
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
from torchvision.datasets.utils import download_url


def remove_URL(text):
    """Remove URLs from a sample string"""
    return re.sub(r"http\S+", "<URL>", text)


def prepare_json(image_root, split):
    imgs = []
    annotations = []
    dataset = os.path.basename(image_root)
    if dataset == 'PCCD':
        aspects = ['general_impression', 'subject_of_photo', 'composition',
                   'use_of_camera', 'depth_of_field', 'color_lighting', 'focus',
                   'description']
        for i, s in enumerate(json.load(open(image_root+"/processed_%s.json" % split, 'r'))):
            caption = remove_URL(s[aspects[i%7]])
            if len(caption) == 0:
                offset = 1
                while len(caption) == 0:
                    caption = remove_URL(s[aspects[(i+offset)%7]])
                    offset += 1
            annotations.append({'image_id': i,
                                'caption': caption,
                                'id': i})
            imgs.append({'id': i})
    elif dataset == 'reddit':
        for i, s in enumerate(json.load(open(image_root+"/processed_%s.json" % split, 'r'))):
            caption = remove_URL(s['first_level_comments_values'][0])
            annotations.append({'image_id': i,
                                'caption': caption,
                                'id': i})
            imgs.append({'id': i})
    elif dataset == 'AVA':
        for i, s in enumerate(json.load(open(image_root+"/processed_%s.json" % split, 'r'))):
            caption = remove_URL(s['comments'][0])
            annotations.append({'image_id': i,
                                'caption': caption,
                                'id': i})
            imgs.append({'id': i})
    return {'images': imgs, 'annotations': annotations}


def coco_caption_eval(coco_gt_root, results_file, split):
    annotation_file = prepare_json(coco_gt_root, split)
    with open("annotation/tmp.json", 'w') as f:
        json.dump(annotation_file, f)

    # create coco object and coco_result object
    coco = COCO("annotation/tmp.json")
    coco_result = coco.loadRes(results_file)

    # create coco_eval object by taking coco and coco_result
    coco_eval = COCOEvalCap(coco, coco_result)

    # evaluate on a subset of images by setting
    # coco_eval.params['image_id'] = coco_result.getImgIds()
    # please remove this line when evaluating the full validation set
    # coco_eval.params['image_id'] = coco_result.getImgIds()

    # evaluate results
    # SPICE will take a few minutes the first time, but speeds up due to caching
    coco_eval.evaluate()

    # print output evaluation scores
    for metric, score in coco_eval.eval.items():
        print(f'{metric}: {score:.3f}')
    
    return coco_eval
