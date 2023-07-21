import os
import json
import numpy as np
from PIL import Image
import torch
import copy
from torch.utils.data import Dataset
from torchvision import transforms

def get_metadata(dataset_name):
    if dataset_name == 'aid':
        meta = {
            'num_classes': 17,
            'path_to_dataset': '/work/src/data/aid',
            'path_to_images': '/work/dataset/AID_multi'
        }
    
    elif dataset_name == 'mlrsnet':
        meta = {
            'num_classes': 60,
            'path_to_dataset': '/work/src/data/mlrsnet',
            'path_to_images': '/work/dataset/MLRSNet/Images'
        }
    
    else:
        raise NotImplementedError('Metadata dictionary not implemented.')
    
    return meta

def get_transforms(img_size):
    '''
    Returns image transforms.
    '''
    
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    tx = {}
    
    tx['train'] = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])
    tx['val'] = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])
    tx['test'] = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])
    
    return tx

def load_data(base_path, P):
    data = {}
    for phase in ['train', 'val', 'test']:
        data[phase] = {}
        data[phase]['labels'] = np.load(os.path.join(base_path, 'formatted_{}_labels.npy'.format(phase)), allow_pickle=True)
        data[phase]['labels_obs'] = np.load(os.path.join(base_path, 'formatted_{}_labels_obs.npy'.format(phase)), allow_pickle=True)
        data[phase]['images'] = np.load(os.path.join(base_path, 'formatted_{}_images.npy'.format(phase)), allow_pickle=True)
        data[phase]['feats'] = np.load(P['{}_feats_file'.format(phase)], allow_pickle=True) if P['use_feats'] else []
    return data

'''
def parse_categories(categories):
    category_list = []
    id_to_index = {}
    for i in range(len(categories)):
        category_list.append(categories[i]['name'])
        id_to_index[categories[i]['id']] = i
    return (category_list, id_to_index)

def get_category_list(P):
    if P['dataset'] == 'pascal':
        catName_to_catID = {
            'aeroplane': 0,
            'bicycle': 1,
            'bird': 2,
            'boat': 3,
            'bottle': 4,
            'bus': 5,
            'car': 6,
            'cat': 7,
            'chair': 8,
            'cow': 9,
            'diningtable': 10,
            'dog': 11,
            'horse': 12,
            'motorbike': 13,
            'person': 14,
            'pottedplant': 15,
            'sheep': 16,
            'sofa': 17,
            'train': 18,
            'tvmonitor': 19
        }
        return {catName_to_catID[k]: k for k in catName_to_catID}
    
    elif P['dataset'] == 'coco':
        load_path = 'data/coco'
        meta = {}
        meta['category_id_to_index'] = {}
        meta['category_list'] = []

        with open(os.path.join(load_path, 'annotations', 'instances_train2014.json'), 'r') as f:
            D = json.load(f)

        (meta['category_list'], meta['category_id_to_index']) = parse_categories(D['categories'])
        return meta['category_list']

    elif P['dataset'] == 'nuswide':
        pass # TODO
    
    elif P['dataset'] == 'cub':
        pass # TODO
    
    elif P['dataset'] == 'mlrsnet':
        pass # TODO
'''

class multilabel:
    def __init__(self, P, tx):
        # get dataset metadata:
        meta = get_metadata(P['dataset'])
        self.base_path = meta['path_to_dataset']
        
        # load data:
        source_data = load_data(self.base_path, P)
        
        # define train set:
        self.train = ds_multilabel(
            P['dataset'],
            source_data['train']['images'],
            source_data['train']['labels'],
            source_data['train']['labels_obs'],
            source_data['train']['feats'] if P['use_feats'] else [],
            tx['train'],
            P['use_feats']
        )
        
        # define val set:
        self.val = ds_multilabel(
            P['dataset'],
            source_data['val']['images'],
            source_data['val']['labels'],
            source_data['val']['labels_obs'],
            source_data['val']['feats'] if P['use_feats'] else [],
            tx['val'],
            P['use_feats']
        )
        
        # define test set:
        self.test = ds_multilabel(
            P['dataset'],
            source_data['test']['images'],
            source_data['test']['labels'],
            source_data['test']['labels_obs'],
            source_data['test']['feats'],
            tx['test'],
            P['use_feats']
        )
        
        # define dict of dataset lengths:
        self.lengths = {'train': len(self.train), 'val': len(self.val), 'test': len(self.test)}
    
    def get_datasets(self):
        return {'train': self.train, 'val': self.val, 'test': self.test}


class ds_multilabel(Dataset):
    def __init__(self, dataset_name, image_ids, label_matrix, label_matrix_obs, feats, tx, use_feats):
        meta = get_metadata(dataset_name)
        self.dataset_name = dataset_name
        self.num_classes = meta['num_classes']
        self.path_to_images = meta['path_to_images']
        
        self.image_ids = image_ids
        self.label_matrix = label_matrix
        self.label_matrix_obs = label_matrix_obs
        self.feats = feats
        self.tx = tx
        self.use_feats = use_feats

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self,idx):
        if self.use_feats:
            # Set I to be a feature vector:
            I = torch.FloatTensor(np.copy(self.feats[idx, :]))
        else:
            # Set I to be an image:
            if self.dataset_name == 'mlrsnet':
                image_path = os.path.join(self.path_to_images, self.image_ids[idx][:-6], self.image_ids[idx] + '.jpg')
            elif self.dataset_name == 'aid':
                image_path = self.image_ids[idx]
            with Image.open(image_path) as I_raw:
                I = self.tx(I_raw.convert('RGB'))
        
        out = {
            'image': I,
            'label_vec_obs': torch.FloatTensor(np.copy(self.label_matrix_obs[idx, :])),
            'label_vec_true': torch.FloatTensor(np.copy(self.label_matrix[idx, :])),
            'idx': idx,
            'image_path': image_path # added for CAM visualization purpose
        }
        
        return out


def get_data(P):
    '''
    Given a parameter dictionary P, initialize and return the specified dataset.
    '''
    
    # define transforms:
    tx = get_transforms(P['img_size'])
    
    # select and return the right dataset:
    if P['dataset'] == 'mlrsnet':
        ds = multilabel(P, tx).get_datasets()
    elif P['dataset'] == 'aid':
        ds = multilabel(P, tx).get_datasets()
    else:
        raise ValueError('Unknown dataset.')
    
    # Optionally overwrite the observed training labels with clean labels:
    if P['train_set_variant'] == 'clean':
        print('Using clean labels for training.')
        ds['train'].label_matrix_obs = copy.deepcopy(ds['train'].label_matrix)
    else:
        print('Using single positive labels for training.')
    
    # Optionally overwrite the observed val labels with clean labels:
    assert P['val_set_variant'] in ['clean', 'observed']
    if P['val_set_variant'] == 'clean':
        print('Using clean labels for validation.')
        ds['val'].label_matrix_obs = copy.deepcopy(ds['val'].label_matrix)
    else:
        print('Using single positive labels for validation.')
    
    # We always use a clean test set:
    ds['test'].label_matrix_obs = copy.deepcopy(ds['test'].label_matrix)
    
    return ds