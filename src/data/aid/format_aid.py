import json
import os
import re
import argparse
import numpy as np
import pandas as pd

pp = argparse.ArgumentParser(description='Format aid metadata.')
pp.add_argument('--load-path', type=str, default='/work/dataset/AID_multi', help='Path to a directory containing a copy of the aid dataset.')
pp.add_argument('--csv-path', type=str, default='/work/src/data/aid')
pp.add_argument('--save-path', type=str, default='/work/src/data/aid', help='Path to output directory.')
args = pp.parse_args()

label_dict = {'airport': 'Airport',
              'bareland': 'BareLand',
              'baseballfield': 'BaseballField',
              'beach': 'Beach',
              'bridge': 'Bridge',
              'center': 'Center',
              'church': 'Church',
              'commercial': 'Commercial',
              'denseresidential': 'DenseResidential',
              'desert': 'Desert',
              'farmland': 'Farmland',
              'forest': 'Forest',
              'industrial': 'Industrial',
              'meadow': 'Meadow',
              'mediumresidential': 'MediumResidential',
              'mountain': 'Mountain',
              'park': 'Park',
              'parking': 'Parking',
              'playground': 'Playground',
              'pond': 'Pond',
              'port': 'Port',
              'railwaystation': 'RailwayStation',
              'resort': 'Resort',
              'river': 'River',
              'school': 'School',
              'sparseresidential': 'SparseResidential',
              'square': 'Square',
              'stadium': 'Stadium',
              'storagetanks': 'StorageTanks',
              'viaduct': 'Viaduct'}

for split in ['train', 'val']:
    # load csv file
    df = pd.read_csv(
        os.path.join(args.csv_path, 'aidmultilabel_' + split + '.csv'),
        delimiter = '\t'
        )
    
    df_label = df['IMAGE\LABEL'].replace(r"[^a-zA-Z]", "", regex=True)
    df_label.update(df_label.map(label_dict))
    df['IMAGE\LABEL'] = '/work/dataset/AID_multi/images_tr/' + df_label + '/' + df['IMAGE\LABEL'] + '.jpg'
    
    # get list of images
    np.save(os.path.join(args.save_path, 'formatted_' + split + '_images.npy'), np.array(df['IMAGE\LABEL'].values)) # image name, shape : (num of image)
    
    # get labels
    np.save(os.path.join(args.save_path, 'formatted_' + split + '_labels.npy'), np.array(df.iloc[:, 1:].values)) # one-hot vector, shape : (num of image, num of label)


for split in ['test']:
    # load csv file
    df = pd.read_csv(
        os.path.join(args.csv_path, 'aidmultilabel_' + split + '.csv'),
        delimiter = '\t'
        )
    
    df_label = df['IMAGE\LABEL'].replace(r"[^a-zA-Z]", "", regex=True)
    df_label.update(df_label.map(label_dict))
    df['IMAGE\LABEL'] = '/work/dataset/AID_multi/images_test/' + df_label + '/' + df['IMAGE\LABEL'] + '.jpg'
    
    # get list of images
    np.save(os.path.join(args.save_path, 'formatted_' + split + '_images.npy'), np.array(df['IMAGE\LABEL'].values)) # image name, shape : (num of image)
    
    # get labels
    np.save(os.path.join(args.save_path, 'formatted_' + split + '_labels.npy'), np.array(df.iloc[:, 1:].values)) # one-hot vector, shape : (num of image, num of label)
