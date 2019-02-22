from __future__ import division
import os, sys, time, scipy.io
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

print("Training Model...", file=sys.stderr, flush=True)

#input_dir = './dataset/Sony/short/'
#gt_dir = './dataset/Sony/long/'
#checkpoint_dir = './result_Sony/'
#result_dir = './result_Sony/'
#model_dir = './saved_model/'

#device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Get all images that are ground truth
#train_fns = glob.glob(gt_dir + '0*.ARW')

# get train IDs
#train_ids = [int(os.path.basename(train_fn)[0:5]) for train_fn in train_fns]

# Raw data takes long time to load. Keep them in memory after loaded.
# gt_images = [None] * 6000
# input_images = {}
# input_images['300'] = [None] * len(train_ids)
# input_images['250'] = [None] * len(train_ids)
# input_images['100'] = [None] * len(train_ids)

# g_loss = torch.from_numpy(np.zeros((5000, 1))).float().to(device)

# allfolders = glob.glob('./result/*0')
# lastepoch = 0
# for folder in allfolders:
#     lastepoch = np.maximum(lastepoch, int(folder[-4:]))

# learning_rate = 1e-4
# model = Net().to(device)

# opt = optim.Adam(model.parameters(), lr = learning_rate)

# for epoch in range(lastepoch , 4001):
    
#     if os.path.isdir("result/%04d"%epoch):
#         continue
#     cnt=0
#     if epoch > 2000:
#         for g in opt.param_groups:
#             g['lr'] = 1e-5

#     for ind in np.random.permutation(len(train_ids)):

#         # get the path from image id
#         train_id = train_ids[ind]
#         in_files = glob.glob(input_dir + '%05d_00*.ARW'%train_id)
#         if len(in_files)-1 == 0:
#               in_path = in_files[0]
#         else:
#             in_path = in_files[np.random.randint(0,len(in_files)-1)]
        
#         _, in_fn = os.path.split(in_path)

#         gt_files = glob.glob(gt_dir + '%05d_00*.ARW'%train_id)
#         gt_path = gt_files[0]
#         _, gt_fn = os.path.split(gt_path)
#         in_exposure =  float(in_fn[9:-5])
#         gt_exposure =  float(gt_fn[9:-5])
#         ratio = min(gt_exposure/in_exposure, 300)

#         st = time.time()
#         cnt += 1

#         if input_images[str(ratio)[0:3]][ind] is None:
            
            
#             raw = rawpy.imread(in_path)
#             input_images[str(ratio)[0:3]][ind] = np.expand_dims(pack_raw(raw),axis=0) *ratio

#             gt_raw = rawpy.imread(gt_path)
#             im = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
#             gt_images[ind] = np.expand_dims(np.float32(im/65535.0),axis = 0)

#         # crop
#         H = input_images[str(ratio)[0:3]][ind].shape[1]
#         W = input_images[str(ratio)[0:3]][ind].shape[2]

#         xx = np.random.randint(0, W - ps)
#         yy = np.random.randint(0, H - ps)

#         input_patch = input_images[str(ratio)[0:3]][ind][:, yy:yy + ps, xx:xx + ps, :]
#         gt_patch = gt_images[ind][:, yy * 2:yy * 2 + ps * 2, xx * 2:xx * 2 + ps * 2, :]

#         if np.random.randint(2, size=1)[0] == 1:  # random flip
#             input_patch = np.flip(input_patch, axis=1)
#             gt_patch = np.flip(gt_patch, axis=1)
#         if np.random.randint(2, size=1)[0] == 1:
#             input_patch = np.flip(input_patch, axis=2)
#             gt_patch = np.flip(gt_patch, axis=2)
#         if np.random.randint(2, size=1)[0] == 1:  # random transpose
#             input_patch = np.transpose(input_patch, (0, 2, 1, 3))
#             gt_patch = np.transpose(gt_patch, (0, 2, 1, 3))

#         input_patch = np.minimum(input_patch,1.0)
#         gt_patch = np.maximum(gt_patch, 0.0)

#         in_img = torch.from_numpy(input_patch).permute(0,3,1,2).to(device)
#         gt_img = torch.from_numpy(gt_patch).permute(0,3,1,2).to(device)

#         model.zero_grad()
#         out_img = model(in_img)

#         loss = reduce_mean(out_img, gt_img)
#         loss.backward()

#         opt.step()
#         g_loss[ind]=loss.data

#         if epoch%save_freq==0:
#             print("%d %d Loss=%.3f Time=%.3f" % (epoch, cnt, np.mean(g_loss[np.where(g_loss)]), time.time() - st))
#             if not os.path.isdir(result_dir + '%04d'%epoch):
#                 os.makedirs(result_dir + '%04d'%epoch)
#             output = out_img.permute(0, 2, 3, 1).cpu().data.numpy()
#             output = np.minimum(np.maximum(output,0),1)

#             temp = np.concatenate((gt_patch[0,:,:,:], output[0,:,:,:]),axis=1)              
#             scipy.misc.toimage(temp*255,  high=255, low=0, cmin=0, cmax=255).save(result_dir + '%04d/%05d_00_train_%d.jpg'%(epoch,train_id,ratio))

#             torch.save(model.state_dict(), checkpoint_dir+'checkpoint_sony_e%04d.pth'%epoch)



result_dir = './result/'

parser = argparse.ArgumentParser(description='Learning to see in the dark - PyTorch')
parser.add_argument('--load', default=result_dir, type=str, metavar='PATH',
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

    t_file_names, t_ids = getInputImagesList()
    gt_file_names, gt_ids = getGroundtruthImagesList()

    dataset =  LearningToSeeInTheDarkDataset(t_ids, 
                                            './dataset/sony/short/', 
                                            './dataset/sony/long/')
    
    learning_rate = 1e-4

    model = Net()
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)

    if args.load:
        if os.path.isdir(args.load):
            name = ''
            if args.epochs:
                name = 'sony_epoch_%04d.pth' % args.epochs
                print("Loading checkpoint '{}'".format(args.load), file=sys.stderr, flush=True)
                checkpoint = torch.load(args.load + name)
                start_epoch = checkpoint['epoch']
                model.load_state_dict(checkpoint['model_state'])
                optimizer.load_state_dict(checkpoint['optimizer_state'])
            
                print("Loaded checkpoint '{}' (epoch {})"
                    .format(args.load, checkpoint['epoch']), file=sys.stderr, flush=True)

            else:
                print("Epochs not specified", file=sys.stderr, flush=True)

    if torch.cuda.device_count() > 1:
        print("Running in parrallel: " + str(torch.cuda.device_count()) + "GPU's",file=sys.stderr, flush=True )
        model = nn.DataParallel(model)

    model.to(device)
    st = time.time()

    for epoch in range(start_epoch , 4001):
        
        if epoch > 2000:
            for g in optimizer.param_groups:
                g['lr'] = 1e-5
        
        losses = []
        for index in np.random.permutation(len(t_ids)):
            
            t_patch, gt_patch = dataset[index]
            
            in_img = torch.from_numpy(t_patch).permute(0,3,1,2).to(device)
            gt_img = torch.from_numpy(gt_patch).permute(0,3,1,2).to(device)

            model.zero_grad()
            out_img = model(in_img)

            loss = torch.abs(out_img - gt_img).mean()
            loss.backward()

            optimizer.step()
            losses.append(loss.item())
        
        with open(result_dir + 'log.txt', 'a+') as f:
            f.write("%d Loss=%.3f Time=%.3f \n" % (epoch, np.mean(losses), time.time() - st))
            
        if epoch % save_freq == 0:
            # create dir
            if not os.path.isdir(result_dir + '%04d'%epoch):
                os.makedirs(result_dir + '%04d'%epoch)

                output = out_img.permute(0, 2, 3, 1).cpu().data.numpy()
                output = np.minimum(np.maximum(output,0),1)

                temp = np.concatenate((gt_patch[0,:,:,:], output[0,:,:,:]),axis=1)
                scipy.misc.toimage(temp*255,  high=255, low=0, cmin=0, cmax=255).save(result_dir + '%04d/%05d_00_train_%d.jpg'%(epoch,train_id,ratio))

                torch.save({
                        'model_state': model.state_dict(),
                        'epoch' : epoch,
                        'optimizer_state': optimizer.state_dict()
                    }, result_dir + 'sony_epoch_%04d.pth' % epoch)

if __name__ == '__main__':
    main()
    print("Training done", file=sys.stderr, flush=True)