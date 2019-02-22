import os,time,scipy.io

import numpy as np
import rawpy
import glob

import torch
import torch.nn as nn
import torch.optim as optim

from model import Net
from dataset import LearningToSeeInTheDarkDatasetTest
from utils import *


result_dir = './test_result_sony/'
if not os.path.isdir(result_dir):
    os.makedirs(result_dir)


def main():

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    t_file_names, t_ids = getInputImagesList()
    gt_file_names, gt_ids = getGroundtruthImagesList()

    dataset =  LearningToSeeInTheDarkDatasetTest(t_ids, 
                                            './dataset/sony/short/', 
                                            './dataset/sony/long/')
    
    # Load trained model
    model = Net()
    model_state_dict = torch.load( './res/sony_epoch_4000.pth')['model_state']
    model.load_state_dict(model_state_dict)

    for index in range( len(t_ids) ):

        test_id, t_full, gt_full = dataset[index]

        for i in range( len(t_full) ):

            input_full, scale_full, ratio = t_full[i]

            in_img = torch.from_numpy(input_full).permute(0,3,1,2).to(device)
            out_img = model(in_img)
            output = out_img.permute(0, 2, 3, 1).cpu().data.numpy()

            output = np.minimum(np.maximum(output,0),1)

            output = output[0,:,:,:]
            gt_full = gt_full[0,:,:,:]
            scale_full = scale_full[0,:,:,:]
            origin_full = scale_full
            scale_full = scale_full*np.mean(gt_full)/np.mean(scale_full)

            scipy.misc.toimage(origin_full*255,  high=255, low=0, cmin=0, cmax=255).save(result_dir + '%5d_00_%d_ori.png'%(test_id,ratio))
            scipy.misc.toimage(output*255,  high=255, low=0, cmin=0, cmax=255).save(result_dir + '%5d_00_%d_out.png'%(test_id,ratio))
            scipy.misc.toimage(scale_full*255,  high=255, low=0, cmin=0, cmax=255).save(result_dir + '%5d_00_%d_scale.png'%(test_id,ratio))
            scipy.misc.toimage(gt_full*255,  high=255, low=0, cmin=0, cmax=255).save(result_dir + '%5d_00_%d_gt.png'%(test_id,ratio))


if __name__ == '__main__':
    main()
    print("Testing done", file=sys.stderr, flush=True)
