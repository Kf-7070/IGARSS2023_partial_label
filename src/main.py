import os
import glob
import json
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter

from config import get_configs
import dataset.datasets as datasets
import model.models as models
from utils.instrumentation import compute_metrics
from utils.losses import compute_batch_loss, loss_an


def main():
    # load configs
    P = get_configs()
    
    # set gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = P['gpu_num']
    print('Using GPU : {}'.format(P['gpu_num']))
    
    # Tensorboard
    summary_writer = SummaryWriter(log_dir=P['save_path'])
    
    # dataloader
    if P['dataset'] in ['mlrsnet', 'aid']:
        dataset = datasets.get_data(P)

    else:
        raise NotImplementedError
    
    dataloader = {}
    if P['test_mode']:
        phase_list =['test']
    else:
        phase_list = ['train', 'val', 'test']
    for phase in phase_list:
        dataloader[phase] = torch.utils.data.DataLoader(
            dataset[phase],
            batch_size = P['bsize'],
            shuffle = phase == 'train',
            sampler = None,
            num_workers = P['num_workers'],
            drop_last = False,
            pin_memory = True
        )
    
    # model
    model = models.ImageClassifier(P)
    
    feature_extractor_params = [param for param in list(model.feature_extractor.parameters()) if param.requires_grad]
    linear_classifier_params = [param for param in list(model.linear_classifier.parameters()) if param.requires_grad]
    opt_params = [
        {'params': feature_extractor_params, 'lr' : P['lr']},
        {'params': linear_classifier_params, 'lr' : P['lr_mult'] * P['lr']}
    ]
    
    # optimizer
    if P['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(opt_params, lr=P['lr'])
    elif P['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(opt_params, lr=P['lr'], momentum=0.9, weight_decay=0.001)
    
    # criterion
    criterion = compute_batch_loss
    
    # Train / Test
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if P['test_mode']:
        test(P, dataset, dataloader, model, optimizer, criterion, device)
    
    else:
        train(P, dataset, dataloader, model, optimizer, criterion, device, summary_writer)
        test(P, dataset, dataloader, model, optimizer, criterion, device)
    
    if summary_writer:
        summary_writer.close()
    
    return


def train(P, dataset, dataloader, model, optimizer, criterion, device, summary_writer):
    model.to(device)
    bestmap_val = 0
    stop_flag = False

    for epoch in range(1, P['num_epochs']+1):
        log = ''
        for phase in ['train', 'val']:
            loss_list = []
            if phase == 'train':
                model.train()
            else:
                if (epoch-1) % P['val_interval'] == 0:
                    model.eval()
                    y_pred = np.zeros((len(dataset[phase]), P['num_classes']))
                    y_true = np.zeros((len(dataset[phase]), P['num_classes']))
                    batch_stack = 0
                else:
                    break
            
            with torch.set_grad_enabled(phase == 'train'):
                for batch in tqdm(dataloader[phase]):
                    image = batch['image'].to(device, non_blocking=True)
                    label_vec_obs = batch['label_vec_obs'].to(device, non_blocking=True)
                    label_vec_true = batch['label_vec_true'].to(device, non_blocking=True)
                    label_vec_true_np = batch['label_vec_true'].clone().numpy()
                    idx = batch['idx']

                    # forward pass
                    optimizer.zero_grad()

                    logits = model(image)

                    if logits.dim() == 1:
                        logits = torch.unsqueeze(logits, 0)
                    preds = torch.sigmoid(logits)

                    if phase == 'train':
                        loss, correction_idx = criterion(logits, label_vec_obs, P)
                        loss.backward()
                        optimizer.step()
                        
                        loss_list.append(loss.item())

                        if P['mod_scheme'] is 'LL-Cp' and correction_idx is not None:
                            dataset[phase].label_matrix_obs[idx[correction_idx[0].cpu()], correction_idx[1].cpu()] = 1.0

                    else:
                        loss = F.binary_cross_entropy_with_logits(logits, label_vec_true)
                        loss_list.append(loss.item())
                        
                        preds_np = preds.cpu().numpy()
                        this_batch_size = preds_np.shape[0]
                        y_pred[batch_stack : batch_stack+this_batch_size] = preds_np
                        y_true[batch_stack : batch_stack+this_batch_size] = label_vec_true_np
                        batch_stack += this_batch_size
            
            if phase == 'train':
                summary_writer.add_scalar('train_loss', np.array(loss_list).mean(), epoch)
                #summary_writer.add_scalar('train_mAP', mAP, epoch)
            else:
                summary_writer.add_scalar('val_loss', np.array(loss_list).mean(), epoch)

        metrics = compute_metrics(y_pred, y_true)
        del y_pred
        del y_true
        map_val = metrics['map']
        rec_at_1 = metrics['rec_at_1']
        rec_at_3 = metrics['rec_at_3']
        rec_at_5 = metrics['rec_at_5']
        prec_at_1 = metrics['prec_at_1']
        prec_at_3 = metrics['prec_at_3']
        prec_at_5 = metrics['prec_at_5']
        top_at_1 = metrics['top_1']
        top_at_3 = metrics['top_3']
        top_at_5 = metrics['top_5']
        
        summary_writer.add_scalar('val_mAP', map_val, epoch)
        summary_writer.add_scalar('rec_at_1', rec_at_1, epoch)
        summary_writer.add_scalar('rec_at_3', rec_at_3, epoch)
        summary_writer.add_scalar('rec_at_5', rec_at_5, epoch)
        summary_writer.add_scalar('prec_at_1', prec_at_1, epoch)
        summary_writer.add_scalar('prec_at_3', prec_at_3, epoch)
        summary_writer.add_scalar('prec_at_5', prec_at_5, epoch)
        summary_writer.add_scalar('top_1', top_at_1, epoch)
        summary_writer.add_scalar('top_3', top_at_3, epoch)
        summary_writer.add_scalar('top_5', top_at_5, epoch)
        
        P['clean_rate'] -= P['delta_rel']
        
        print(f"Epoch {epoch} : val mAP {map_val:.3f}")
        log += f"Epoch {epoch} : val mAP {map_val:.3f}" + '\n'
        
        # check best mAP and early stop
        if bestmap_val < map_val:
            bestmap_val = map_val
            bestmap_epoch = epoch
            
            print(f'Saving model weight for best val mAP {bestmap_val:.3f}')
            log += f'Saving model weight for best val mAP {bestmap_val:.3f}' + '\n'
            path = os.path.join(P['save_path'], 'bestmodel_epoch_{}.pt'.format(epoch))
            torch.save((model.state_dict(), P), path)
        
        elif bestmap_val - map_val > 3:
            print('Early stopped.')
            log += 'Early stopped.'
            stop_flag = True
        
        with open(os.path.join(P['save_path'], 'log.txt'), 'a') as f:
            f.write(log)
        
        if stop_flag:
            break
    return


def test(P, dataset, dataloader, model, optimizer, criterion, device):
    # load model
    l = glob.glob(os.path.join(P['param_path'], 'bestmodel_epoch_*.pt'))
    l.sort(reverse=True)
    path = l[0]
    model_state, _ = torch.load(path)
    model.load_state_dict(model_state)

    # evaluate test dataset
    phase = 'test'
    model.eval()
    y_pred = np.zeros((len(dataset[phase]), P['num_classes']))
    y_true = np.zeros((len(dataset[phase]), P['num_classes']))
    batch_stack = 0
    with torch.set_grad_enabled(phase == 'train'):
        for batch in tqdm(dataloader[phase]):
            image = batch['image'].to(device, non_blocking=True)
            label_vec_obs = batch['label_vec_obs'].to(device, non_blocking=True)
            label_vec_true = batch['label_vec_true'].clone().numpy()
            idx = batch['idx']

            # forward pass
            optimizer.zero_grad()

            logits = model(image)
            
            if logits.dim() == 1:
                logits = torch.unsqueeze(logits, 0)
            preds = torch.sigmoid(logits)
            
            preds_np = preds.cpu().numpy()
            this_batch_size = preds_np.shape[0]
            y_pred[batch_stack : batch_stack+this_batch_size] = preds_np
            y_true[batch_stack : batch_stack+this_batch_size] = label_vec_true
            batch_stack += this_batch_size

    metrics = compute_metrics(y_pred, y_true)
    map_test = metrics['map']
    
    print(f'Test mAP : {map_test:.3f}')
    
    with open(os.path.join(P['save_path'], 'best_map.txt'), 'w') as f:
        f.write('Config path : {}\nparm_path : {}\nTest mAP : {:.3f}'.format(os.path.join(P['save_path'], "config.json"),
                                                                              path,
                                                                              map_test))
    
    with open(os.path.join(P['save_path'], 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    return

if __name__ == '__main__':
    main()