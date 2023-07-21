import os, datetime, argparse, json
from munch import Munch as mch


_DATASET = ('mlrsnet', 'aid')
_TRAIN_SET_VARIANT = ('observed', 'clean')
_OPTIMIZER = ('adam', 'sgd')
_LOSS = ('BCE, asymmetric')
_SCHEMES = ('AN' ,'LL-R', 'LL-Ct', 'LL-Cp')
_LOOKUP = {
    'feat_dim': {
        'resnet50': 2048
    },
    'num_classes': {
        'mlrsnet': 60,
        'aid': 17
    }}

def set_dir(args):
    save_path = os.path.join(args.save_path, args.dataset + '_' + args.loss_function + '_' + args.mod_scheme, datetime.datetime.now().strftime('%Y_%m_%d_%H_%M'))
    os.makedirs(save_path, exist_ok=True)
    return save_path

def set_follow_up_configs(args):
    args.feat_dim = _LOOKUP['feat_dim'][args.arch]
    args.num_classes = _LOOKUP['num_classes'][args.dataset]
    args.save_path = set_dir(args)
    
    if args.param_path == '':
        args.param_path = args.save_path

    if args.delta_rel != 0:
        args.delta_rel /= 100
        
    args.clean_rate = 1

    return args

def get_configs():
    parser = argparse.ArgumentParser()
    
    # machine settings
    parser.add_argument('--gpu_num', type=str, default='1')
    parser.add_argument('--num_workers', type=int, default=8)
    
    # dataset setting
    parser.add_argument('--dataset', type=str, default='mlrsnet', choices=_DATASET)
    parser.add_argument('--img_size', type=int, default=256)
    
    # path settings
    parser.add_argument('--save_path', type=str, default='/work/src/log/example')
    parser.add_argument('--exp_name', type=str, default='exp_default')
    parser.add_argument('--param_path', type=str, default='')
    
    # experiment settings
    parser.add_argument('--test_mode', action='store_true',
                        help='evaluate test dataset')
    parser.add_argument('--ss_frac_train', type=float, default=1.0,
                        help='fraction of training set to subsample')
    parser.add_argument('--ss_frac_val', type=float, default=1.0,
                        help='fraction of val set to subsample')
    
    parser.add_argument('--train_set_variant', type=str, default='observed',
                        choices=_TRAIN_SET_VARIANT)
    parser.add_argument('--val_set_variant', type=str, default='clean')
    
    parser.add_argument('--use_feats', type=bool, default=False,
                        help='False if end-to-end training, True if linear training')
    parser.add_argument('--freeze_feature_extractor', type=bool, default=False)
    
    # loss function / mode
    parser.add_argument('--loss_function', type=str, default='BCE', choices=_LOSS)
    parser.add_argument('--mod_scheme', type=str, default='AN', choices=_SCHEMES)
    parser.add_argument('--delta_rel', type=float, default=0.1)
    
    parser.add_argument('--eps', default=1e-5, type=float,
                        help='eps for focal loss (default: 1e-5)')
    parser.add_argument('--dtgfl', action='store_true', default=False,
                        help='disable_torch_grad_focal_loss in asl')
    parser.add_argument('--gamma_pos', default=0, type=float,
                        metavar='gamma_pos', help='gamma pos for simplified asl loss')
    parser.add_argument('--gamma_neg', default=2, type=float,
                        metavar='gamma_neg', help='gamma neg for simplified asl loss')
    parser.add_argument('--loss_clip', default=0.0, type=float,
                        help='scale factor for clip')
    
    # model
    parser.add_argument('--arch', type=str, default='resnet50')
    parser.add_argument('--use_pretrained', type=bool, default=True)
    
    # training
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--bsize', type=int, default=128)
    parser.add_argument('--val_interval', type=int, default=1,
                        help='interval of validation')
    
    # optimizer
    parser.add_argument('--optimizer', type=str, default='adam', choices=_OPTIMIZER)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--lr_mult', type=float, default=10)
    
    # seed
    parser.add_argument('--ss_seed', type=int, default=999,
                        help='seed fo subsampling')
    parser.add_argument('--split_seed', type=int, default=1200)
    
    
    args = parser.parse_args()
    args = set_follow_up_configs(args)
    
    # save config
    if args.test_mode:
        # load training config
        path = os.path.join(args.param_path, "config.json")
        with open(path, 'r') as f:
            args_train = json.load(f)
        
        for k, v in args_train.items():
            if k not in ['gpu_num', 'num_workers', 'save_path', 'exp_name', 'test_mode']:
                setattr(args, k, v)
        
        # save testing config
        path = os.path.join(args.save_path, "config.json")
        with open(path, 'w') as f:
            json.dump(vars(args), f, indent=2)
        print("Full test config saved to {}".format(path))
        
    else:
        # save training config
        path = os.path.join(args.save_path, "config.json")
        with open(path, 'w') as f:
            json.dump(vars(args), f, indent=2)
        print("Full training config saved to {}".format(path))
    
    args = mch(**vars(args))
    
    return args