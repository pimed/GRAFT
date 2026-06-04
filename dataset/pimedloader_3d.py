""" dataloader is quite self-explanatory. All the data loading and data preprocessing classes and functions live here. """

from operator import getitem
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import SimpleITK as sitk

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import os,fnmatch
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
# from util_funcs.utils import crop_or_pad, resample_im,Recenter,bbox_3D, cropping, resample_im_to_size, sitk_crop, sitk_crop_label,resample_for_training
import copy 
import json
from scipy import ndimage
from tqdm import tqdm
from torchio import RandomElasticDeformation
import pandas as pd
# from guided_diffusion_3cohorts.compute_stats import *


class RandomFlipTransform_2D_sample(object):
    """ Applies random horizontal flip transform to the entire sample,
    i.e. t2, adc, dwi, mask and label"""
    
    def __init__(self, atlas = False,prob_atlas = False,p=0.5):
        self.p = p
        self.atlas = atlas
        self.prob_atlas = prob_atlas

    def __call__(self, sample, verbose=False):
        
        x = random.random()
        
        if x > 1 - self.p:
            
            if verbose:
                print('Horizontal flip applied.')     
                
            new_sample = copy.copy(sample)

            new_sample['t2'] = (sample['t2'][:,::-1,:]).copy()
            new_sample['adc'] = (sample['adc'][:,::-1,:]).copy()
            new_sample['dwi'] = (sample['dwi'][:,::-1,:]).copy()
            new_sample['mask'] = (sample['mask'][:,::-1,]).copy()
            new_sample['label'] = (sample['label'][:,::-1,]).copy()
            if self.atlas:
                new_sample['atlas'] = (sample['atlas'][:,::-1,]).copy()
                # if self.prob_atlas:
                #     new_sample['prob_atlas'] = (sample['prob_atlas'][:,::-1,]).copy()
            return new_sample
        
        else:
            return sample


class RandomRotationTransform_2D_sample(object):
    """ Applies rotation by a random angle transform to the entire sample,
    i.e. t2, adc, dwi, mask and label"""
    
    def __init__(self, degrees, atlas = False, prob_atlas = False):
        self.degrees = degrees
        self.atlas = atlas
        self.prob_atlas = prob_atlas

    def myrotate_ndimage(self, im, angle):
        return ndimage.rotate(im, angle, axes=(1,0), reshape=False, order=0, mode='nearest')
    
    def myrotate_ndimage_mri(self, im, angle):
        return ndimage.rotate(im, angle, axes=(1,0), reshape=False, mode='nearest')
    
    def __call__(self, sample, verbose=False):
        
        angle = random.randrange(self.degrees[0], self.degrees[1])
        
        if verbose:
            print('Rotation angle=',angle)
        
        new_sample = copy.copy(sample)
        
        new_sample['t2'] = (self.myrotate_ndimage_mri(sample['t2'], angle)).copy()
        new_sample['adc'] = (self.myrotate_ndimage_mri(sample['adc'], angle)).copy()
        new_sample['dwi'] = (self.myrotate_ndimage_mri(sample['dwi'], angle)).copy()
        
        new_sample['mask'] = (self.myrotate_ndimage(sample['mask'], angle)).copy()
        new_sample['label'] = (self.myrotate_ndimage(sample['label'], angle)).copy()
        if self.atlas:
            new_sample['atlas'] = (self.myrotate_ndimage(sample['atlas'], angle)).copy()
            # if self.prob_atlas:
            #     new_sample['prob_atlas'] = (self.myrotate_ndimage(sample['prob_atlas'], angle)).copy()
        return new_sample

class RandomAffineTransform_3D_sample(object):
    """ Applies affine transformation by a random 3D Euler transform to the entire sample,
    i.e. t2, adc, dwi, mask and label
    RandomTranslation in 10 pixels +
    3D rotation center at (128,128,10) of image, rotation angle (-pi/(5*degrees),pi/(5*degrees))
    """
    def __init__(self, degrees):
        self.degrees = degrees

    def myaffine(self):
        rotation_center = (128, 128, 10)
        translation = (np.random.randint(10),np.random.randint(10),np.random.randint(10))
        x_rad = np.random.randint(10)
        y_rad = np.random.randint(10)
        z_rad = np.random.randint(10)
        if x_rad >= 5:
            x_rad = (x_rad-4)*self.degrees
        else:
            x_rad = (x_rad-5)*self.degrees
        
        if y_rad >= 5:
                y_rad = (y_rad-4)*self.degrees
        else:
            y_rad = (y_rad-5)*self.degrees
            
        if z_rad >= 5:
                z_rad = (z_rad-4)*self.degrees
        else:
            z_rad = (z_rad-5)*self.degrees
        theta_x = -np.pi/float(x_rad)
        theta_y = -np.pi/float(y_rad)
        theta_z = -np.pi/float(z_rad)

        similarity = sitk.Euler3DTransform(rotation_center,theta_x,theta_y,theta_z,translation)
        return  similarity
        
    def __call__(self, t2,adc,dwi,mask, verbose=False):
        trans = self.myaffine()
        
        t2_transformed = sitk.Resample(t2, trans, sitk.sitkLinear, 0.0, t2.GetPixelID())
        adc_transformed = sitk.Resample(adc, trans, sitk.sitkLinear, 0.0, adc.GetPixelID())
        dwi_transformed = sitk.Resample(dwi, trans, sitk.sitkLinear, 0.0, dwi.GetPixelID())
        mask_transformed = sitk.Resample(mask, trans, sitk.sitkNearestNeighbor, 0.0, mask.GetPixelID())
        
        return t2_transformed,adc_transformed,dwi_transformed,mask_transformed,trans
    
    

class NormalizeTransform_2D_sample(object):
    """Normalizes t2, adc, and dwi images wrt training cohort;
    Z-score such that resulting images have mean 0, std 1."""

    def __init__(self, stats, stats_per_case=False,stats_per_case_file = None, atlas=False, prob_atlas=False):
        self.stats = stats
        self.atlas = atlas
        self.prob_atlas = prob_atlas
        if stats_per_case:
            self.normalize_per_case = True
            # Check if stats file exists, otherwise compute on-the-fly
            if stats_per_case_file and os.path.exists(stats_per_case_file):
                with open(stats_per_case_file) as json_file:
                    self.stats_per_case = json.load(json_file)
                self.compute_on_fly = False
            else:
                print("📊 Stats file not found - will compute normalization statistics on-the-fly within mask regions")
                self.stats_per_case = {}
                self.compute_on_fly = True
        else:
            self.normalize_per_case = False

    def __call__(self, sample, verbose=False):
        case_id = sample["case_id"]
        if self.normalize_per_case:
            if self.compute_on_fly:
                # Compute statistics on-the-fly within mask region
                mask = sample["mask"]
                t2_img = sample["t2"]
                adc_img = sample["adc"] 
                dwi_img = sample["dwi"]
                
                # Extract pixels within mask > 0
                mask_region = mask > 0
                
                # Compute mean and std within mask region for each modality
                t2_masked = t2_img[mask_region]
                adc_masked = adc_img[mask_region]
                dwi_masked = dwi_img[mask_region]
                
                t2_mean = np.mean(t2_masked) if len(t2_masked) > 0 else 0.0
                t2_std = np.std(t2_masked) if len(t2_masked) > 0 else 1.0
                adc_mean = np.mean(adc_masked) if len(adc_masked) > 0 else 0.0
                adc_std = np.std(adc_masked) if len(adc_masked) > 0 else 1.0
                dwi_mean = np.mean(dwi_masked) if len(dwi_masked) > 0 else 0.0
                dwi_std = np.std(dwi_masked) if len(dwi_masked) > 0 else 1.0
                
                # Cache computed stats for potential reuse
                self.stats_per_case[case_id] = [t2_mean, t2_std, adc_mean, adc_std, dwi_mean, dwi_std]
                
                if verbose:
                    print(f"🧮 Computed stats for {case_id}: T2({t2_mean:.2f}±{t2_std:.2f}), ADC({adc_mean:.2f}±{adc_std:.2f}), DWI({dwi_mean:.2f}±{dwi_std:.2f})")
            else:
                # Use pre-computed stats from file
                stats = self.stats_per_case[case_id]
                t2_mean = stats[0]
                t2_std = stats[1]
                adc_mean = stats[2]
                adc_std = stats[3]
                dwi_mean = stats[4]
                dwi_std = stats[5]
            
            # Ensure no zero std to avoid division by zero
            if dwi_std == 0.0:
                dwi_std = 1.0
            if t2_std == 0.0:
                t2_std = 1.0
            if adc_std == 0.0:
                adc_std = 1.0
        else:
            if self.atlas:
                # if self.prob_atlas:
                #     t2_mean, t2_std, adc_mean, adc_std, dwi_mean, dwi_std, atlas_mean, atlas_std, prob_atlas_mean, prob_atlas_std = self.stats
                # else:
                (
                    t2_mean,
                    t2_std,
                    adc_mean,
                    adc_std,
                    dwi_mean,
                    dwi_std,
                    # atlas_mean,
                    # atlas_std,
                ) = self.stats
            else:
                t2_mean, t2_std, adc_mean, adc_std, dwi_mean, dwi_std = self.stats
        if verbose:
            print(
                "Normalized with cohort stats: t2 mean {:.2f}, std {:.2f}; \
            adc mean {:.2f}, std {:.2f}, dwi mean {:.2f}, std {:.2f}".format(
                    t2_mean, t2_std, adc_mean, adc_std, dwi_mean, dwi_std
                )
            )

        new_sample = copy.copy(sample)

        new_sample["t2"] = ((sample["t2"] - t2_mean) / t2_std).copy()
        new_sample["adc"] = ((sample["adc"] - adc_mean) / adc_std).copy()
        new_sample["dwi"] = ((sample["dwi"] - dwi_mean) / dwi_std).copy()
        if self.atlas:
            # new_sample["atlas"] = ((sample["atlas"] - atlas_mean) / atlas_std).copy()
            new_sample["atlas"] = sample["atlas"]
            # if self.prob_atlas:
            #     new_sample['prob_atlas'] = ((sample['prob_atlas'] - prob_atlas_mean)/prob_atlas_std).copy()
        del new_sample["case_id"]
        return new_sample

class ProstateMaskTransform_2D_sample(object):
    """ Applies prostate mask to the T2, ADC, and DWI inputs."""
    
    def __init__(self):
        pass
    
    def __call__(self, sample, verbose=False):
        """ Takes input 'sample', which is a dict:
                {'t2': stacked_t2 (3,256,256), 
                'adc': stacked_adc (3,256,256), 
                'mask':mask_np (256,256), 
                'label':label_np (256,256), 
                'volume': Sitk volume, 
                'case_id':case_id str} """ 
        
        new_sample = copy.copy(sample)

        prostate_mask_np = sample['mask']
        
        new_sample['t2'][prostate_mask_np == 0] = 0
        new_sample['adc'][prostate_mask_np == 0] = 0
        new_sample['dwi'][prostate_mask_np == 0] = 0
        
        return new_sample
    


class PIMEDDataset_3D(Dataset):
    """3D SPCNet dataset, one datapoint=one volume"""

    def __init__(self,path_dict,pred_type = 'binary',region_based = False,use_T2 = True,use_ADC = True,use_DWI = True,
                 transform=None, stats_per_case=True,stats_per_case_file = None,fold_cases=None, case_id2cohort = None,savefolder_weights = None,
                model_name = 'original', apply_affine = False,test_flag = False):
        """
        Args:
            path_dict (dict): Path dictionary for all images and their correspoding labels.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.path_dict = path_dict
        self.pred_type = pred_type
        self.region_based = region_based

        self.use_T2 = use_T2
        self.use_ADC = use_ADC
        self.use_DWI = use_DWI
        self.savefolder_weights = savefolder_weights
        self.test_flag = test_flag
        self.case_ids = fold_cases
        self.hist_plot = fold_cases

        self.affine_degree = 30
        self.apply_affine = apply_affine
        self.case_id2cohort = case_id2cohort
        self.case_n_slices = []
        self.stats_per_case = stats_per_case
        self.stats_per_case_file = stats_per_case_file
        self.case_slices = {}
        self.idx_to_case_id = {}
        self.select_slices_start_dict = {}  # Will be populated lazily
        self.mask_start_idx_dict = {}       # Will be populated lazily  
        self.mask_end_idx_dict = {}         # Will be populated lazily
        self.case_id2count = {}
        self.case_count2id = {}
        self.affine_trans = RandomAffineTransform_3D_sample(self.affine_degree)
        self.model_name = model_name

        # Skip populate_self_dicts() for faster initialization
        # Mask indices will be computed lazily in __getitem__
        print(f"✅ Fast initialization: {len(self.case_ids)} cases ready for lazy loading")
            
        self.apply_transform = False
        
        # if stats is None:
        #         if not self.val:
        #             self.stats = self.compute_stats(stats_path)
        # else:
        #     self.stats = stats
        self.stats = None
            
        if transform:
            self.transform = transform
        else:
            if self.model_name =='atlas':
                    self.transform = NormalizeTransform_2D_sample(self.stats,self.stats_per_case,self.stats_per_case_file)
            else:
                self.transform = NormalizeTransform_2D_sample(self.stats,self.stats_per_case,self.stats_per_case_file)
            # if val:
            #     if self.model_name =='atlas':
            #         self.transform = NormalizeTransform_2D_sample(self.stats,self.stats_per_case,True)
            #     else:
            #         self.transform = NormalizeTransform_2D_sample(self.stats,self.stats_per_case)
            # else:
            #     if self.model_name =='atlas':
            #         self.transform = transforms.Compose(
            #                 [ RandomRotationTransform_2D_sample(degrees=(-15,15),atlas =True,prob_atlas = self.use_prob_atlas),
            #                 RandomFlipTransform_2D_sample(True,self.use_prob_atlas),
            #                 NormalizeTransform_2D_sample(self.stats,self.stats_per_case,True,self.use_prob_atlas) ] )
            #     else:
            #         self.transform = transforms.Compose(
            #                 [ RandomRotationTransform_2D_sample(degrees=(-15,15)),
            #                 RandomFlipTransform_2D_sample(),
            #                 NormalizeTransform_2D_sample(self.stats,self.stats_per_case) ] )

        self.apply_transform = True
        
    def __len__(self):
        """ Number of 2D datapoints. """
        # return np.sum([x['n_slices'] for x in self.case_slices.values()]).astype('int32')
        # return 1
        return len(self.case_ids)
        # return 20

        
    def return_mask_start_idx(self):
        """dictionary containing the index of the starting slice of prostate mask 
            in the whole image {id: idx}. Used for evaluation only to stack slice predictions back to a whole image
        """
        return self.mask_start_idx_dict
    
    def return_mask_end_idx(self):
        """dictionary containing the index of the ending slice of prostate mask 
            in the whole image {id: idx}. Used for evaluation only to stack slice predictions back to a whole image
        """
        return self.mask_end_idx_dict
    
    def return_case_count2id(self):
        """
        return: the dictionary with 
                key: count of case_id inside the whole data
                value: case_id
        """
        return self.case_count2id

    def return_stats(self):
        """
        return training stats computed during training, used for evaluation dataloader.
        """
        return self.stats

    def return_case_slices(self):
        """
        return dict self.case_slices
        key: [Str] case_id,
        value: dict{'n_slices':prostate slices, 'slice_0_idx': i, starting index of this case in idx_to_case_id}
        """
        return self.case_slices

    def return_case_n_slices(self):
        return self.case_n_slices


    def plot_hist(self, im_np, mask_np, save_name,save_path,file_kind):
        """plot histogram of given image, also save it to save_path"""
        if file_kind =='t2':
            max_hist = 500
            min_hist = 0
        elif file_kind =='adc':
            max_hist = 3000
            min_hist = 0
        else:
            max_hist = 50
            min_hist = 0
        save_hist_path = os.path.join(save_path,'hist')
        if not os.path.exists(save_hist_path):
            os.makedirs(save_hist_path)
            
        im_np = im_np[mask_np>0]

        plt.hist(im_np.ravel(), max_hist+1, (min_hist, max_hist))
        plt.show()
        plt.savefig(os.path.join(save_hist_path,save_name+'_'+file_kind+'.png'))
        plt.close()
        
    
    def read_t2_adc_mask_from_disk(self,cohort, t2_file, adc_file,dwi_file, mask_file, case_id = None):
        """ Reads the t2, adc, dwi and prostate mask images from disk.
            If apply_affine = True, apply affine transformation to t2, adc, dwi, and mask here 
            and return the transformation.
        """
        
        # Read images from disk
        if self.region_based:
            t2 = sitk.ReadImage(os.path.join(self.path_dict['t2'], t2_file))
            adc = sitk.ReadImage(os.path.join(self.path_dict['adc'], adc_file))
            dwi = sitk.ReadImage(os.path.join(self.path_dict['dwi'], dwi_file))
            mask = sitk.ReadImage(os.path.join(self.path_dict['prostate'], mask_file))
        else:
            t2 = sitk.ReadImage(os.path.join(self.path_dict[cohort]['t2'], t2_file))
            adc = sitk.ReadImage(os.path.join(self.path_dict[cohort]['adc'], adc_file))
            dwi = sitk.ReadImage(os.path.join(self.path_dict[cohort]['dwi'], dwi_file))
            mask = sitk.ReadImage(os.path.join(self.path_dict[cohort]['prostate'], mask_file))
        
        mask_np = sitk.GetArrayFromImage(mask)
    
        trans = None
        if self.apply_affine:
            t2,adc,dwi,mask,trans = self.affine_trans(t2,adc,dwi,mask)
        volume = mask

        t2_np = sitk.GetArrayFromImage(t2)
        adc_np = sitk.GetArrayFromImage(adc)
        dwi_np = sitk.GetArrayFromImage(dwi)

        mask_np = sitk.GetArrayFromImage(mask).astype('int8')
        mask_np[mask_np > 0] = 1
        
        return t2_np, adc_np,dwi_np, mask_np, volume,trans
    
    def read_atlas(self,atlas_path,atlas_file,case_id):
        atlas = sitk.ReadImage(os.path.join(atlas_path,atlas_file))
        atlas_np = sitk.GetArrayFromImage(atlas)
        return atlas_np
    
    def read_prob_atlas(self,atlas_agg_file,atlas_norm_file,mask_np):
        atlas_agg = sitk.ReadImage(os.path.join(os.path.join(self.path_dict['prob_atlas'],'prob_agg_atlas'),atlas_agg_file))
        atlas_agg_np = sitk.GetArrayFromImage(atlas_agg)
        atlas_norm = sitk.ReadImage(os.path.join(os.path.join(self.path_dict['prob_atlas'],'prob_norm_atlas'),atlas_norm_file))
        atlas_norm_np = sitk.GetArrayFromImage(atlas_norm)
        # atlas_agg_np = np.where(atlas_agg_np<10**(-5),10**(-5),atlas_agg_np)
        # atlas_norm_np = np.where(atlas_norm_np<10**(-5),10**(-5),atlas_norm_np)
        return atlas_agg_np,atlas_norm_np


    def read_labels_from_disk(self, label_file,mask_np,case_id,cohort):
        """ Only used for evaluation.
            Reads cancer label image files from disk;
            Constructs label_np array where each pixel value is
            0=normal, 1=cs cancer.
            If there is an affine transformation, then affine_trans will be applied to lesions image.
            if there is missing lesion image, a zero-value array will be used.
            Also gets mask_start_idx and mask_end_idx for constructing 2 dict return_mask_start_idx and return_mask_end_idx:
            mask_start_idx is the beginning slice index of prostate mask
            mask_start_idx is the ending slice index of prostate mask
        """
        cohort = self.case_id2cohort[case_id]
        label_np = None
        
        # Check if label file exists
        if self.region_based:
            label_path = os.path.join(self.path_dict['cancer'], label_file)
        else:
            label_path = os.path.join(self.path_dict[cohort]['cancer'], label_file)
            
        if os.path.exists(label_path):
            all_label = sitk.ReadImage(label_path)
            label_np = sitk.GetArrayFromImage(all_label).astype('float32')
            if not self.region_based:
                if self.pred_type =='3class':
                    if cohort == 'stanford':
                        label_np = np.where(label_np>1,2,label_np)
                    elif cohort == 'ucla':
                        label_np = np.where(label_np>1,2,label_np)
                    else:
                        label_np[label_np > 0] = 2
                elif self.pred_type =='binary':
                    csPCa_label = np.zeros_like(label_np)
                    if cohort == 'stanford':
                        csPCa_label[label_np>1] = 1
                    elif cohort == 'ucla':
                        csPCa_label[label_np>1] = 1
                    else:
                        csPCa_label[label_np>0] = 1
                    label_np = csPCa_label

        if label_np is None:
            label_np = np.zeros((mask_np.shape))

        coor = np.nonzero(mask_np)[0]
        prostate = np.unique(coor)
        # get the start and end index of prostate
        mask_start_idx = prostate[0]
        mask_end_idx = prostate[-1]
        return label_np,prostate,mask_start_idx,mask_end_idx
    


    def _compute_case_mask_indices(self, case_id):
        """
        Lazily compute mask indices for a single case.
        This replaces the heavy populate_self_dicts() method.
        """
        try:
            cohort = self.case_id2cohort[case_id]
            mask_pattern = case_id + '*'
            
            # Find mask file
            if self.region_based:
                mask_files = fnmatch.filter(os.listdir(self.path_dict['prostate']), mask_pattern)
            else:
                mask_files = fnmatch.filter(os.listdir(self.path_dict[cohort]['prostate']), mask_pattern)
                
            if not mask_files:
                print(f'Skipping {case_id}: no prostate mask file found')
                # Set default values to prevent re-computation
                self.select_slices_start_dict[case_id] = 0
                self.mask_start_idx_dict[case_id] = "0"
                self.mask_end_idx_dict[case_id] = "19"
                return
                
            mask_file = mask_files[0]
            
            # Read mask to get indices
            if self.region_based:
                mask = sitk.ReadImage(os.path.join(self.path_dict['prostate'], mask_file))
            else:
                mask = sitk.ReadImage(os.path.join(self.path_dict[cohort]['prostate'], mask_file))
            mask_np = sitk.GetArrayFromImage(mask).astype('int8')
            mask_np[mask_np > 0] = 1
            
            # Compute prostate boundaries
            coor = np.nonzero(mask_np)[0]
            if len(coor) > 0:
                prostate = np.unique(coor)
                mask_start_idx = prostate[0]
                mask_end_idx = prostate[-1]
            else:
                # Empty mask - use defaults
                mask_start_idx = 0
                mask_end_idx = mask_np.shape[0] - 1
                
            # Cache the computed values
            self.select_slices_start_dict[case_id] = mask_start_idx
            self.mask_start_idx_dict[case_id] = str(mask_start_idx)
            self.mask_end_idx_dict[case_id] = str(mask_end_idx)
            
        except Exception as e:
            print(f'Error computing mask indices for {case_id}: {str(e)}')
            # Set safe defaults
            self.select_slices_start_dict[case_id] = 0
            self.mask_start_idx_dict[case_id] = "0"
            self.mask_end_idx_dict[case_id] = "19"
    
    def populate_self_dicts(self):
        """ Sweeps through all cases and constructs the following dictionaries:
            self.case_slices: dict 
                key: case_id, 
                value: dict{'n_slices':prostate shape, 'slice_0_idx': i, starting index of this case in idx_to_case_id}
                
            self.idx_to_case_id: dict 
                key: i, the index of a datapoint in the Dataset object 
                value: case_id to which this datapoint corresponds"""
        i = 0
        case_idx_count = 0
        for case_id in tqdm(self.case_ids):
            # self.idx_to_case_id[case_idx_count] = case_id
            # case_idx_count += 1
            cohort = self.case_id2cohort[case_id]
            mask_pattern = case_id+'*'
            lesion_pattern = case_id+'*'

            try:
                # Try to find mask file - skip case if not found
                mask_files = fnmatch.filter(os.listdir(self.path_dict[cohort]['prostate']), mask_pattern)
                if not mask_files:
                    print(f'Skipping {case_id}: no prostate mask file found')
                    continue
                mask_file = mask_files[0]
                
                # Try to find label file
                try:
                    label_file = fnmatch.filter(os.listdir(self.path_dict[cohort]['cancer']), lesion_pattern)[0]
                except: 
                    print(case_id+' has no lesion or prostate mask')
                    label_file = 'no_label'
                    
                mask = sitk.ReadImage(os.path.join(self.path_dict[cohort]['prostate'], mask_file))
            except Exception as e:
                print(f'Skipping {case_id}: error accessing files - {str(e)}')
                continue

            mask_np = sitk.GetArrayFromImage(mask).astype('int8')
            mask_np[mask_np > 0] = 1
            try:
                label_np,prostate,mask_start_idx,mask_end_idx = self.read_labels_from_disk(label_file,mask_np, case_id,cohort)
                # print('prostate',prostate[0])
                # print('prostate.shape[0]',prostate.shape[0])
                self.mask_start_idx_dict[case_id] = str(mask_start_idx)
                self.mask_end_idx_dict[case_id] = str(mask_end_idx)
                self.case_slices[case_id] = {'n_slices':prostate.shape[0], 'slice_0_idx':i}
                # self.cases_labels[case_id] = label_np

                self.case_id2count[case_id] = case_idx_count
                self.case_count2id[case_idx_count] = case_id
                case_idx_count +=1

                for j in range(i, i+self.case_slices[case_id]['n_slices']):
                    self.idx_to_case_id[j] = case_id  
                i += self.case_slices[case_id]['n_slices']
                # print('mask_np.shape',mask_np.shape)
                # mask_center_slice = mask_np.shape[0]//2
                # select_slices_start = mask_center_slice-10
                coor = np.nonzero(mask_np)[0]
                prostate = np.unique(coor)
                # get the start and end index of prostate
                mask_start_idx = prostate[0]
                self.select_slices_start_dict[case_id] = mask_start_idx
            except:
                print(f'==================={case_id} error in mask ==============')



    
    def __getitem__(self, idx, verbose=False):
        """ Grabs one sample from the dataset.
        Sample consists of:
                {'t2': stacked_t2 (3,256,256), 
                'adc': stacked_adc (3,256,256), 
                'dwi': stacked_dwi (3,256,256), 
                'mask':mask_np (256,256), 
                'label':label_np (256,256), 
                'volume': Sitk volume, 
                'case_id':case_id str
                'slice_idx':slice idx of this sampled slice in the case (int)
                (Optional)'final_decision':lesion decision ground truth label of the slice (int)
                (Optional)'clinical_decision': clinical decision ground truth label of the slice (int)

                }

        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # idx = 11
        # For the 2D dataset, idx corresponds to a single slice from a case

        # For 3D training, idx directly maps to case_id
        case_id = self.case_ids[idx]
        
        # Lazy loading: compute mask indices on first access
        if case_id not in self.select_slices_start_dict:
            self._compute_case_mask_indices(case_id)
            
        select_start_idx = self.select_slices_start_dict[case_id]
        
        cohort = self.case_id2cohort[case_id]
        mask_pattern = case_id+'*'
        lesion_pattern = case_id+'*'
        
        t2_pattern = case_id+'*'
        adc_pattern = case_id+'*'
        dwi_pattern = case_id+'*'
    
        try:
            if self.region_based:
                t2_pattern = case_id + '_0000*'
                adc_pattern = case_id + '_0001*'
                dwi_pattern = case_id + '_0002*'

                mask_files = fnmatch.filter(os.listdir(self.path_dict['prostate']), mask_pattern)
                adc_files = fnmatch.filter(os.listdir(self.path_dict['adc']), adc_pattern)
                t2_files = fnmatch.filter(os.listdir(self.path_dict['t2']), t2_pattern)
                dwi_files = fnmatch.filter(os.listdir(self.path_dict['dwi']), dwi_pattern)
            else:
                mask_files = fnmatch.filter(os.listdir(self.path_dict[cohort]['prostate']), mask_pattern)
                adc_files = fnmatch.filter(os.listdir(self.path_dict[cohort]['adc']), adc_pattern)
                t2_files = fnmatch.filter(os.listdir(self.path_dict[cohort]['t2']), t2_pattern)
                dwi_files = fnmatch.filter(os.listdir(self.path_dict[cohort]['dwi']), dwi_pattern)
                
            if not mask_files:
                raise FileNotFoundError(f"No mask file found for {case_id}")
            if not adc_files:
                raise FileNotFoundError(f"No ADC file found for {case_id}")
            if not t2_files:
                raise FileNotFoundError(f"No T2 file found for {case_id}")
            if not dwi_files:
                raise FileNotFoundError(f"No DWI file found for {case_id}")
                
            mask_file = mask_files[0]
            adc_file = adc_files[0]
            t2_file = t2_files[0]
            dwi_file = dwi_files[0]
        except Exception as e:
            print(f'Error accessing files for {case_id}: {str(e)}')
            # Return a dummy sample or handle error appropriately
            raise e

        t2_np, adc_np,dwi_np, mask_np, volume,affine_trans = self.read_t2_adc_mask_from_disk(cohort,t2_file, adc_file, dwi_file, mask_file,case_id) 

        if np.sum(mask_np)==0:
            mask_np = np.ones(t2_np.shape)
        
        if verbose:
            print('Input shapes t2, adc, mask:',t2_np.shape, adc_np.shape, mask_np.shape)

        try:
            if self.region_based:
                label_file = fnmatch.filter(os.listdir(self.path_dict['cancer']), lesion_pattern)[0]
            else:
                label_file = fnmatch.filter(os.listdir(self.path_dict[cohort]['cancer']), lesion_pattern)[0]
            
        except:
            label_file = 'no_label'
            print('------ WARNING: no label for case_id:',case_id,'------')

        label_np, cancer, mask_start_idx,mask_end_idx= self.read_labels_from_disk(label_file,mask_np, case_id,cohort)
        z_size = mask_end_idx - mask_start_idx +1
        
        # STEP 1: Compute normalization stats on FULL prostate region (before crop/pad)
        # This ensures all prostate tissue is used for statistics, even if we later center-crop
        prostate_mask = mask_np[mask_start_idx:mask_end_idx+1, :, :]
        t2_prostate = t2_np[mask_start_idx:mask_end_idx+1, :, :]
        adc_prostate = adc_np[mask_start_idx:mask_end_idx+1, :, :]
        dwi_prostate = dwi_np[mask_start_idx:mask_end_idx+1, :, :]
        
        # Compute stats within mask region
        mask_region = prostate_mask > 0
        
        t2_masked = t2_prostate[mask_region]
        adc_masked = adc_prostate[mask_region]
        dwi_masked = dwi_prostate[mask_region]
        
        t2_mean = np.mean(t2_masked) if len(t2_masked) > 0 else 0.0
        t2_std = np.std(t2_masked) if len(t2_masked) > 0 else 1.0
        adc_mean = np.mean(adc_masked) if len(adc_masked) > 0 else 0.0
        adc_std = np.std(adc_masked) if len(adc_masked) > 0 else 1.0
        dwi_mean = np.mean(dwi_masked) if len(dwi_masked) > 0 else 0.0
        dwi_std = np.std(dwi_masked) if len(dwi_masked) > 0 else 1.0
        
        # Ensure no zero std
        if t2_std == 0.0:
            t2_std = 1.0
        if adc_std == 0.0:
            adc_std = 1.0
        if dwi_std == 0.0:
            dwi_std = 1.0
        
        # STEP 2: Crop or pad to 20 slices
        if z_size >= 20:
            # Center crop to 20 slices
            crop_start = mask_start_idx + (z_size - 20) // 2
            crop_end = crop_start + 20
            
            t2_np = t2_np[crop_start:crop_end, :, :]
            adc_np = adc_np[crop_start:crop_end, :, :]
            dwi_np = dwi_np[crop_start:crop_end, :, :]
            mask_np = mask_np[crop_start:crop_end, :, :]
            label_np = label_np[crop_start:crop_end, :, :]
        else:
            # Crop to prostate region first, then pad to 20 slices
            t2_np = t2_np[mask_start_idx:mask_end_idx+1, :, :]
            adc_np = adc_np[mask_start_idx:mask_end_idx+1, :, :]
            dwi_np = dwi_np[mask_start_idx:mask_end_idx+1, :, :]
            mask_np = mask_np[mask_start_idx:mask_end_idx+1, :, :]
            label_np = label_np[mask_start_idx:mask_end_idx+1, :, :]
            
            # Pad to 20 slices
            padding_need = 20 - z_size
            pad_before = padding_need // 2
            pad_after = padding_need - pad_before
            
            t2_np = np.pad(t2_np, ((pad_before, pad_after), (0, 0), (0, 0)), mode='constant', constant_values=0)
            adc_np = np.pad(adc_np, ((pad_before, pad_after), (0, 0), (0, 0)), mode='constant', constant_values=0)
            dwi_np = np.pad(dwi_np, ((pad_before, pad_after), (0, 0), (0, 0)), mode='constant', constant_values=0)
            mask_np = np.pad(mask_np, ((pad_before, pad_after), (0, 0), (0, 0)), mode='constant', constant_values=0)
            label_np = np.pad(label_np, ((pad_before, pad_after), (0, 0), (0, 0)), mode='constant', constant_values=0)
        
        # Verify shape is now [20, H, W]
        assert t2_np.shape[0] == 20, f"Expected z-dimension to be 20, got {t2_np.shape[0]}"
        
        # STEP 3: Apply normalization using pre-computed stats from full prostate
        t2_np = (t2_np - t2_mean) / t2_std
        adc_np = (adc_np - adc_mean) / adc_std
        dwi_np = (dwi_np - dwi_mean) / dwi_std
        
        # Stack modalities into final image tensor [3, 20, 256, 256]
        img = np.stack([t2_np, adc_np, dwi_np]).astype('float32')
        
        # Label tensor [20, 256, 256]
        label = np.array(label_np, dtype=np.int_)
        
        # Prostate mask tensor [20, 256, 256] - binary mask indicating prostate region
        prostate_mask = np.array(mask_np > 0, dtype=np.float32)

        return img, label, prostate_mask, mask_start_idx, mask_end_idx
    
    
    
                
                
    def compute_stats(self, savepath= None ):
        """ Computes statistics for the entire dataset.
        Returns (t2_mean, t2_std, adc_mean, adc_std, dwi_mean, dwi_std)"""
        
        t2_mean, t2_std, _ = get_dataset_stats(image_type = 't2',path_dict =  self.path_dict,case_ids = self.case_ids,case_id2cohort = self.case_id2cohort)
        adc_mean, adc_std, _ = get_dataset_stats(image_type = 'adc',path_dict = self.path_dict,case_ids = self.case_ids,case_id2cohort = self.case_id2cohort)
        dwi_mean, dwi_std, _ = get_dataset_stats(image_type = 'dwi', path_dict = self.path_dict,case_ids = self.case_ids,case_id2cohort = self.case_id2cohort)
        stats = [t2_mean, t2_std, adc_mean, adc_std, dwi_mean, dwi_std]
        if self.model_name =='atlas':
            atlas_mean, atlas_std, _ = get_dataset_stats(image_type = 'atlas', path_dict = self.path_dict,case_ids = self.case_ids,case_id2cohort = self.case_id2cohort)
            # if self.use_prob_atlas==True:
            #     prob_atlas_mean, prob_atlas_std, _ = get_dataset_stats(image_type = 'prob_atlas', path_dict = self.path_dict,case_ids = self.case_ids,case_id2cohort = self.case_id2cohort)
            #     stats = [t2_mean, t2_std, adc_mean, adc_std, dwi_mean, dwi_std, atlas_mean, atlas_std,prob_atlas_mean,prob_atlas_std]
            # else:
            stats = [t2_mean, t2_std, adc_mean, adc_std, dwi_mean, dwi_std, atlas_mean, atlas_std]
        print('stats',stats)
        if savepath:
            with open(savepath, 'w') as fp:
                json.dump(stats, fp)
        
        return stats


class PIMEDwrapper:
    def __init__(self,path_dict,train_3d = False, cancer_only=True, 
                 transform=None, stats=None, stats_per_case=False,stats_per_case_file = None, case_id2cohort = None,val = False,savefolder_weights = None,cropping = False,pred = False,
                 id_names_thresed = None, model_name = 'original',clinical_info_path = None, apply_affine = False,stats_path = None, use_prob_atlas = False,test_flag = False):
        self.path_dict = path_dict
        self.train_3d = train_3d
        self.cancer_only = cancer_only
        self.transform = transform
        self.stats = stats
        self.stats_per_case = stats_per_case
        self.stats_per_case_file = stats_per_case_file
        self.case_id2cohort = case_id2cohort
        self.val = val
        self.savefolder_weights = savefolder_weights
        self.cropping = cropping
        self.pred = pred
        self.id_names_thresed = id_names_thresed
        self.model_name = model_name
        self.clinical_info_path = clinical_info_path
        self.apply_affine = apply_affine
        self.stats_path = stats_path
        self.use_prob_atlas = use_prob_atlas
        self.test_flag = test_flag
    def return_pimed_loader(self,fold_cases):
        return PIMEDDataset_2D(path_dict = self.path_dict,train_3d =self.train_3d, cancer_only=self.cancer_only, 
                 transform=self.transform, stats=self.stats, stats_per_case=self.stats_per_case,stats_per_case_file = self.stats_per_case_file, fold_cases=fold_cases, case_id2cohort = self.case_id2cohort,val = self.val,savefolder_weights = self.savefolder_weights,cropping = self.cropping,pred = self.pred,
                 id_names_thresed = self.id_names_thresed, model_name = self.model_name,clinical_info_path = self.clinical_info_path, apply_affine = self.apply_affine,stats_path = self.stats_path, use_prob_atlas = self.use_prob_atlas,test_flag = self.test_flag)


def my_collate(batch):
    """ Custom collate function for SPCNet data.
        Return: 
        List(
            data: Tuple( t2 (torch.Tensor[batch_size,3,256,256]),
                         adc (torch.Tensor[batch_size,3,256,256]),
                         dwi (torch.Tensor[batch_size,3,256,256]))
            target: Tuple( [Ground truth lesion(torch.LongTensor[batch_size,256,256])] *13 )
            mask: mask label(torch.LongTensor[batch_size,256,256])
            case_slice_list: List(List[case_id],List[slice_idx]) content correspond to slices in data, target, mask
        }
    """
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    t2_list = []
    adc_list = []
    dwi_list = []
    label_list = []
    mask_list = []
    case_slice_list = [[],[]]
    
    for sample in batch:
        t2_np = np.moveaxis(sample['t2'],-1,0)
        adc_np = np.moveaxis(sample['adc'],-1,0)
        dwi_np = np.moveaxis(sample['dwi'],-1,0)

        t2_list.append(torch.tensor(t2_np))
        adc_list.append(torch.tensor(adc_np))
        dwi_list.append(torch.from_numpy(dwi_np))
        label_list.append(torch.tensor(sample['label'].copy()))
        mask_list.append(torch.tensor(sample['mask'].copy()))
        
        case_slice_list[0].append(sample['case_id'])
        case_slice_list[1].append(sample['slice_idx'])
        
    t2 = np.concatenate(t2_list, axis=0)
    
    adc = np.concatenate(adc_list,axis=0)
    
    
    dwi = np.concatenate(dwi_list,axis=0)
    
    
    y = np.concatenate(label_list, axis=0)
    
    
    mask = np.concatenate(mask_list,axis=0)
    
    data = (t2, adc, dwi)

    target = tuple([y]*13)

    return [data, target,mask,case_slice_list]

def my_collate_decision(batch):
    """ Custom collate function for SPCNet-decision data.
        Return: 
        List(
            data: Tuple( t2 (torch.Tensor[batch_size,3,256,256]),
                         adc (torch.Tensor[batch_size,3,256,256]),
                         dwi (torch.Tensor[batch_size,3,256,256]))
            target: Tuple( [Ground truth lesion(torch.LongTensor[batch_size,256,256])] *13 )
            final_decisions: slice-level ground truth (torch.LongTensor[batch_size,1])
            mask: mask label(torch.LongTensor[batch_size,256,256])
            case_slice_list: List(List[case_id],List[slice_idx]) content correspond to slices in data, target, mask
        )
    """
        
    t2_list = []
    adc_list = []
    dwi_list = []
    label_list = []
    mask_list = []
    case_slice_list = [[],[]]
    final_decisions = []
    
    for sample in batch:
        # t2_np = np.moveaxis(sample['t2'],-1,0)
        # adc_np = np.moveaxis(sample['adc'],-1,0)
        # dwi_np = np.moveaxis(sample['dwi'],-1,0)

        # t2_list.append(t2_np)
        # adc_list.append(adc_np)
        # dwi_list.append(dwi_np)
        # label_list.append(sample['label'].copy())
        # mask_list.append(sample['mask'].copy())
        
        # case_slice_list[0].append(sample['case_id'])
        # case_slice_list[1].append(sample['slice_idx'])
        # final_decisions.append(sample['final_decision'])
        t2_np = np.moveaxis(sample['t2'],-1,0)
        adc_np = np.moveaxis(sample['adc'],-1,0)
        dwi_np = np.moveaxis(sample['dwi'],-1,0)

        t2_list.append(torch.tensor(t2_np))
        adc_list.append(torch.tensor(adc_np))
        dwi_list.append(torch.from_numpy(dwi_np))
        label_list.append(torch.tensor(sample['label'].copy()))
        mask_list.append(torch.tensor(sample['mask'].copy()))
        
        case_slice_list[0].append(sample['case_id'])
        case_slice_list[1].append(sample['slice_idx'])
        final_decisions.append(torch.tensor(sample['final_decision']))
        
    t2 = torch.stack(t2_list, dim=0)
    t2 = t2.type(torch.FloatTensor).to('cuda')
    
    adc = torch.stack(adc_list,dim=0)
    adc = adc.type(torch.FloatTensor).to('cuda')
    
    dwi = torch.stack(dwi_list,dim=0)
    dwi = dwi.type(torch.FloatTensor).to('cuda')
    
    y = torch.stack(label_list, dim=0)
    y = y.type(torch.LongTensor).to('cuda')
    
    mask = torch.stack(mask_list,dim=0)
    mask = mask.type(torch.LongTensor).to('cuda')
    #print(t2.shape, adc.shape, y.shape)
    final_decisions = torch.stack(final_decisions,dim=0)
    final_decisions = final_decisions.type(torch.LongTensor).to('cuda')
    # # data = (t2, adc, dwi)
    # t2 = np.concatenate(t2_list, axis=0)
    
    # adc = np.concatenate(adc_list,axis=0)
    
    
    # dwi = np.concatenate(dwi_list,axis=0)
    
    
    # y = np.concatenate(label_list, axis=0)
    
    
    # mask = np.concatenate(mask_list,axis=0)
    
    # final_decisions = np.vstack(final_decisions,axis=0)
    

    data = (t2, adc, dwi)

    target = tuple([y]*13)

    return [data, target,final_decisions,mask,case_slice_list]


def my_collate_clinical(batch):
    """ Custom collate function for SPCNet-clinical data.
        Return: 
        List(
            data: Tuple( t2 (torch.Tensor[batch_size,3,256,256]),
                         adc (torch.Tensor[batch_size,3,256,256]),
                         dwi (torch.Tensor[batch_size,3,256,256]))
            target: Tuple( [Ground truth lesion(torch.LongTensor[batch_size,256,256])] *13 )
            clinical_decisions: slice-level clinical ground truth (torch.LongTensor[batch_size,1])
            mask: mask label(torch.LongTensor[batch_size,256,256])
            case_slice_list: List(List[case_id],List[slice_idx]) content correspond to slices in data, target, mask
        )
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    t2_list = []
    adc_list = []
    dwi_list = []
    label_list = []
    mask_list = []
    case_slice_list = [[],[]]
    clinical_decisions = []
    
    for sample in batch:
        t2_np = np.moveaxis(sample['t2'],-1,0)
        adc_np = np.moveaxis(sample['adc'],-1,0)
        dwi_np = np.moveaxis(sample['dwi'],-1,0)

        t2_list.append(torch.tensor(t2_np))
        adc_list.append(torch.tensor(adc_np))
        dwi_list.append(torch.from_numpy(dwi_np))
        label_list.append(torch.tensor(sample['label'].copy()))
        mask_list.append(torch.tensor(sample['mask'].copy()))
        
        case_slice_list[0].append(sample['case_id'])
        case_slice_list[1].append(sample['slice_idx'])
        clinical_decisions.append(torch.tensor(sample['clinical_decision']))
        
    t2 = torch.stack(t2_list, dim=0)
    t2 = t2.type(torch.FloatTensor).to(device)
    
    adc = torch.stack(adc_list,dim=0)
    adc = adc.type(torch.FloatTensor).to(device)
    
    dwi = torch.stack(dwi_list,dim=0)
    dwi = dwi.type(torch.FloatTensor).to(device)
    
    y = torch.stack(label_list, dim=0)
    y = y.type(torch.LongTensor).to(device)
    
    mask = torch.stack(mask_list,dim=0)
    mask = mask.type(torch.LongTensor).to(device)
    clinical_decisions = torch.stack(clinical_decisions,dim=0)
    clinical_decisions = clinical_decisions.type(torch.LongTensor).to(device)
    
    data = (t2, adc, dwi)
    target = tuple([y]*13)

    return [data, target,clinical_decisions,mask,case_slice_list]

def my_collate_pred(batch):
    """ Custom collate function for SPCNet data at eval time only (all models).
        Return: 
        List(
            data: Tuple( t2 (torch.Tensor[batch_size,3,256,256]),
                         adc (torch.Tensor[batch_size,3,256,256]),
                         dwi (torch.Tensor[batch_size,3,256,256]))
            target: Tuple( [Ground truth lesion(torch.LongTensor[batch_size,256,256])] *13 )
            mask: mask label(torch.LongTensor[batch_size,256,256])
            case_slice_list: List(List[case_id],List[slice_idx]) content correspond to slices in data, target, mask
        )
    """

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    t2_list = []
    adc_list = []
    dwi_list = []
    mask_list = []
    case_slice_list = [[],[]]
    target = None
    
    for sample in batch:
        t2_np = np.moveaxis(sample['t2'],-1,0)
        adc_np = np.moveaxis(sample['adc'],-1,0)
        dwi_np = np.moveaxis(sample['dwi'],-1,0)

        t2_list.append(torch.tensor(t2_np))
        adc_list.append(torch.tensor(adc_np))
        dwi_list.append(torch.from_numpy(dwi_np))
        mask_list.append(torch.tensor(sample['mask'].copy()))
        
        case_slice_list[0].append(sample['case_id'])
        case_slice_list[1].append(sample['slice_idx'])
        
    t2 = torch.stack(t2_list, dim=0)
    t2 = t2.type(torch.FloatTensor).to(device)
    
    adc = torch.stack(adc_list,dim=0)
    adc = adc.type(torch.FloatTensor).to(device)
    
    dwi = torch.stack(dwi_list,dim=0)
    dwi = dwi.type(torch.FloatTensor).to(device)

    mask = torch.stack(mask_list,dim=0)
    mask = mask.type(torch.LongTensor).to(device)
    
    data = (t2, adc, dwi)

    print(t2.shape)
    return [data, target,mask,case_slice_list]

### ~~~~~~~~~~~~~~~ UTILITIES ~~~~~~~~~~~~~~~

def visualize_batch(data, target):
    """ Visualize one entire batch of data. """
    t2 = data[0].cpu().numpy()
    adc = data[1].cpu().numpy()
    dwi = data[2].cpu().numpy()
    y = target[0].cpu().numpy()
    
    n_samples = t2.shape[0]
    
    fig,ax = plt.subplots(n_samples, 9, figsize=(20,n_samples*6))
    
    print('Above each plot are the (min, max) pixel values; \
    for label maps, above each plot are the unique label values appearing in that map')
    
    for i in range(n_samples):
        ax[i,0].imshow(t2[i,0,:,:], cmap='gray')
        ax[i,0].set_title('{:.2f}, {:.2f}'.format(np.min(t2[i,0,:,:]),np.max(t2[i,0,:,:])) )
        
        ax[i,1].imshow(t2[i,1,:,:], cmap='gray')
        ax[i,1].set_title('{:.2f}, {:.2f}'.format(np.min(t2[i,1,:,:]),np.max(t2[i,1,:,:])) )
        
        ax[i,2].imshow(t2[i,2,:,:], cmap='gray')
        ax[i,2].set_title('{:.2f}, {:.2f}'.format(np.min(t2[i,2,:,:]),np.max(t2[i,2,:,:])) )
        
        ax[i,3].imshow(adc[i,0,:,:], cmap='gray')
        ax[i,3].set_title('{:.2f}, {:.2f}'.format(np.min(adc[i,0,:,:]),np.max(adc[i,0,:,:])) )
        
        ax[i,4].imshow(adc[i,1,:,:], cmap='gray')
        ax[i,4].set_title('{:.2f}, {:.2f}'.format(np.min(adc[i,0,:,:]),np.max(adc[i,0,:,:])) )
        
        ax[i,5].imshow(adc[i,2,:,:], cmap='gray')
        ax[i,5].set_title('{:.2f}, {:.2f}'.format(np.min(adc[i,0,:,:]),np.max(adc[i,0,:,:])) )
        
        ax[i,6].imshow(dwi[i,2,:,:], cmap='gray')
        ax[i,6].set_title('{:.2f}, {:.2f}'.format(np.min(dwi[i,0,:,:]),np.max(dwi[i,0,:,:])) )
        
        ax[i,7].imshow(y[i,0,:,:])
        ax[i,7].set_title(str(np.unique(y[i,0,:,:])))
        
        ax[i,8].imshow(y[i,1,:,:])
        ax[i,8].set_title(str(np.unique(y[i,1,:,:])))
        
        ax[i,9].imshow(y[i,2,:,:])
        ax[i,9].set_title(str(np.unique(y[i,2,:,:])))
        
        for j in range(10):
            axis=ax[i][j]
            axis.get_xaxis().set_visible(False)
            axis.get_yaxis().set_visible(False)
