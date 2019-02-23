from torch.utils.data import Dataset
import glob
import os, sys
import rawpy
import numpy as np
from utils import pack_raw


class LearningToSeeInTheDarkDataset(Dataset):

    def __init__(self, t_ids, t_paths, gt_paths, patch_size=512, transforms=None):
        super(LearningToSeeInTheDarkDataset, self).__init__()
        
        # We only look at 00 pictures
        self.t_ids = np.unique(t_ids)
        
        self.t_paths = t_paths
        self.gt_paths = gt_paths
        
        self.gt_images = np.array([None] * 6000)
        #self.gt_images 
        self.t_images = {}
        self.t_images['100'] = np.array([None] * len(self.t_ids))
        #self.t_images['100'][:] = None
        self.t_images['250'] = np.array([None] * len(self.t_ids))
        #self.t_images['250'][:] = None
        self.t_images['300'] = np.array([None] * len(self.t_ids))
        #self.t_images['300'][:] = None

        print("Loading data into memory", file=sys.stderr, flush=True)
        
        for index, t_id in enumerate(self.t_ids):
            print(t_id)
            t_files = glob.glob(self.t_paths + '%05d_00*.ARW'%t_id)
            gt_files = glob.glob(self.gt_paths + '%05d_00*.ARW'%t_id)

            gt_path = gt_files[0]
            _, gt_fn = os.path.split(gt_path)
            gt_exposure =  float(gt_fn[9:-5])

            for t_path in t_files:
                _, t_fn = os.path.split(t_path)
                t_exposure =  float(t_fn[9:-5])

                ratio = min(gt_exposure/t_exposure, 300)
                ratio_key = str(ratio)[0:3]

                t_raw = rawpy.imread(t_path)
                self.t_images[ratio_key][index] = np.expand_dims(pack_raw(t_raw) , axis=0) * ratio

            gt_raw = rawpy.imread(gt_path)
            im = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
            self.gt_images[index] = np.expand_dims( np.float32(im/65535.0) , axis = 0)

            if index % 10 == 0:
                print("Loaded %d ids "%index, file=sys.stderr, flush=True)

        print("Completed loading data into memory", file=sys.stderr, flush=True)
        
        self.patch_size = patch_size
        self.transforms = transforms

    def __getitem__(self, index):
        
        t_id = self.t_ids[index]
        
        t_files = glob.glob(self.t_paths + '%05d_00*.ARW'%t_id)
        gt_files = glob.glob(self.gt_paths + '%05d_00*.ARW'%t_id)
        
        # Randomly input select image
        if len(t_files) == 1:
              t_path = t_files[0]
        else:
            t_path = t_files[np.random.randint(0,len(t_files)-1)]

        # Only one ground truth image
        gt_path = gt_files[0]

        _, t_fn = os.path.split(t_path)
        _, gt_fn = os.path.split(gt_path)

        t_exposure =  float(t_fn[9:-5])
        gt_exposure =  float(gt_fn[9:-5])
        
        # Ratio is either 300/250/100
        # Se Table 1
        ratio = min(gt_exposure/t_exposure, 300)
        ratio_key = str(ratio)[0:3]

        #if self.t_images[ratio_key][index] is None:
            
            #t_raw = rawpy.imread(t_path)
            #self.t_images[ratio_key][index] = np.expand_dims(pack_raw(t_raw) , axis=0) * ratio
            
            #gt_raw = rawpy.imread(gt_path)
            #im = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
            #self.gt_images[index] = np.expand_dims( np.float32(im/65535.0) , axis = 0)

        #t_raw = rawpy.imread(t_path)
        #t_image = np.expand_dims(pack_raw(t_raw) , axis=0) * ratio
        
        #gt_raw = rawpy.imread(gt_path)
        #im = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
        #gt_image = np.expand_dims( np.float32(im/65535.0) , axis = 0)

        # crop
        H = self.t_images[ratio_key][index].shape[1]
        W = self.t_images[ratio_key][index].shape[2]

        #H = t_image.shape[1]
        #W = t_image.shape[2]

        xx = np.random.randint(0, W - self.patch_size)
        yy = np.random.randint(0, H - self.patch_size)

        t_patch = self.t_images[ratio_key][index][:, yy:yy + self.patch_size, xx:xx + self.patch_size, :]
        gt_patch = self.gt_images[index][:, yy * 2:yy * 2 + self.patch_size * 2, xx * 2:xx * 2 + self.patch_size * 2, :]
        #t_patch = t_image[:, yy:yy + self.patch_size, xx:xx + self.patch_size, :]
        #gt_patch = gt_image[:, yy * 2:yy * 2 + self.patch_size * 2, xx * 2:xx * 2 + self.patch_size * 2, :]

        if np.random.randint(2) == 1:  # random flip
            t_patch = np.flip(t_patch, axis=1)
            gt_patch = np.flip(gt_patch, axis=1)
        if np.random.randint(2) == 1:
            t_patch = np.flip(t_patch, axis=2)
            gt_patch = np.flip(gt_patch, axis=2)
        if np.random.randint(2) == 1:  # random transpose
            t_patch = np.transpose(t_patch, (0, 2, 1, 3))
            gt_patch = np.transpose(gt_patch, (0, 2, 1, 3))

        t_patch = np.minimum(t_patch,1.0)
        gt_patch = np.maximum(gt_patch, 0.0)

        return t_id, t_patch, gt_patch
        
    def __len__(self):
        return len(self.t_ids)

