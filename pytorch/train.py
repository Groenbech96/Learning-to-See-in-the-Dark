from __future__ import division
import os, sys, time, scipy.io
from torch.utils.data import DataLoader
import argparse
import numpy as np
import rawpy
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from model import Net
from dataset import LearningToSeeInTheDarkDataset
from utils import *
from collections import OrderedDict

print("Training Model...", file=sys.stderr, flush=True)

result_dir = './result/'

parser = argparse.ArgumentParser(description='Learning to see in the dark - PyTorch')
parser.add_argument('--load', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--epochs', default=0, type=int, metavar='N',
                    help='number of total epochs to run')

def main():
    

    start_epoch = 1
    save_freq = 250

    model = None
    losses = []

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()
    print(args)

    t_file_names, t_ids = getInputImagesList()
    gt_file_names, gt_ids = getGroundtruthImagesList()

    dataset =  LearningToSeeInTheDarkDataset(t_ids, 
                                            './dataset/sony/short/', 
                                            './dataset/sony/long/')
    
    dataLoader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=8)
    
    learning_rate = 1e-4

    model = Net().cuda()
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)
    
    if args.load:
        if os.path.isdir(args.load):
            name = ''
            if args.epochs:
                name = 'sony_epoch_%04d.pth' % args.epochs
                print("Loading checkpoint '{}'".format(args.load), file=sys.stderr, flush=True)
                checkpoint = torch.load(args.load + name)
                start_epoch = checkpoint['epoch']
                
                # Correct for nn.parallel model loading error
                state_dict = checkpoint['model_state']
                state_dict_new = OrderedDict()
                for k, v in state_dict.items():
                    k_new = k[7:]
                    state_dict_new[k_new] = v

                model.load_state_dict(state_dict_new)
                optimizer.load_state_dict(checkpoint['optimizer_state'])
            
                print("Loaded checkpoint '{}' (epoch {})"
                    .format(args.load, checkpoint['epoch']), file=sys.stderr, flush=True)

            else:
                print("Epochs not specified", file=sys.stderr, flush=True)

    if torch.cuda.device_count() > 1:
       print("Running in parrallel: " + str(torch.cuda.device_count()) + " GPU's",file=sys.stderr, flush=True )
       model = nn.DataParallel(model,  device_ids=[0, 1]).cuda()

    model.to(device)
    st = time.time()

    print("Starting training", file=sys.stderr, flush=True)
    for epoch in range(start_epoch , 4001):
        
        if epoch > 2000:
            for g in optimizer.param_groups:
                g['lr'] = 1e-5
        
        g_loss = []
        for t_id, t_patch, gt_patch in dataLoader:
            
            # Get the only element in the batch
            t_patch = t_patch[0]
            gt_patch = gt_patch[0]

            in_img = t_patch.permute(0,3,1,2).to(device)
            gt_img = gt_patch.permute(0,3,1,2).to(device)

            model.zero_grad()
            out_img = model(in_img)

            loss = torch.abs(out_img - gt_img).mean()
            loss.backward()

            optimizer.step()
            g_loss.append(loss.item())
        
        
        with open(result_dir + 'log.txt', 'a+') as f:
            f.write("%d Mean Loss=%.3f Time=%.3f \n" % (epoch, np.mean(g_loss), time.time() - st))
        
        if epoch % save_freq == 0:
            # create dir
            if not os.path.isdir(result_dir + '%04d'%epoch):
                os.makedirs(result_dir + '%04d'%epoch)

                output = out_img.permute(0, 2, 3, 1).cpu().data.numpy()
                output = np.minimum(np.maximum(output,0),1)

                temp = np.concatenate((gt_patch[0,:,:,:], output[0,:,:,:]),axis=1)
                scipy.misc.toimage(temp*255,  high=255, low=0, cmin=0, cmax=255).save(result_dir + '%04d/%05d_00_train.jpg'%(epoch, t_id))

                torch.save({
                        'model_state': model.state_dict(),
                        'epoch' : epoch,
                        'optimizer_state': optimizer.state_dict()
                    }, result_dir + 'sony_epoch_%04d.pth' % epoch)

if __name__ == '__main__':
    main()
    print("Training done", file=sys.stderr, flush=True)