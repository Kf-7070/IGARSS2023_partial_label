import json
import os
import argparse
import numpy as np
import pandas as pd

pp = argparse.ArgumentParser(description='Format mlrsnet metadata.')
pp.add_argument('--load-path', type=str, default='/work/dataset/MLRSNet', help='Path to a directory containing a copy of the mlrsnet dataset.')
pp.add_argument('--csv-path', type=str, default='/work/src/data/mlrsnet')
pp.add_argument('--save-path', type=str, default='/work/src/data/mlrsnet', help='Path to output directory.')
args = pp.parse_args()


for split in ['train', 'val', 'test']:
    # load csv file
    df = pd.read_csv(
        os.path.join(args.csv_path, 'mlrsnet_' + split + '.csv'),
        delimiter = '\t'
        )
    
    # get list of images
    np.save(os.path.join(args.save_path, 'formatted_' + split + '_images.npy'), np.array(df['image'].values)) # image name, shape : (num of image)
    
    # get labels
    np.save(os.path.join(args.save_path, 'formatted_' + split + '_labels.npy'), np.array(df.iloc[:, 1:].values)) # one-hot vector, shape : (num of image, num of label)
