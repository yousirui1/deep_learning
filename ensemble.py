import os, sys, argparse
sys.path.append(parentdir)
import dataloaders
import models
from utilities import *
from traintest import validate
import numpy as np
from scipy import stats
import torch

def get_ensemble_res(mdl_list, base_path, dataset='audioset'):
    num_class = 527 if dataset == 'audioset' else 200
    ensemble_res = np.zeros([len(mdl_list) + 1, 3])
    if os.path.exists(base_path) == False:
        os.mkdir(base_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for model_idx, mdl in enumerate(mdl_list):
        print('now loading model {:d}: {:s}'.format(model_idx, mdl))

        sd = torch.load(mdl, map_location = device)
        if 'module.effnet._fc.weight' in sd.keys():
            del sd['module.effnet._fc.weight']
            del sd['module.effnet._fc.bias']
            torch.save(sd, mdl)
        audio_model = models.EffNetAttention(label_dim = num_class, b = 2, pretrain=False,
                head_num = 4)
        audio_model = torch.nn.DataParallel(audio_model)
        audio_model.load_state_dict(sd, strict=True)
        
        args.exp_dir = base_path

        stats, _ = validate(audio_model, eval_loader, args, model_idx)
        mAP = np.mean([stat['AP'] for stat in stats])
        mAUC = np.mean([stat['auc'] for stat in stats])
        dprime = d_prime(mAUC)
        ensemble_res[model_idx, :] = [mAP, mAUC, dprime]
        #print('Model {:d} {:s} mAP: {:.6f}, AUC:]]]]]]]]]]]]]]]]]]]]]]]]]]]]][][][][][][][][][][

    #calculate the ensemble result 

