import os
import torch
import importlib
import torch.distributed as dist
import numpy as np


def relative_bias_interpolate(checkpoint,config):
    for k in list(checkpoint['model']):
        if 'relative_position_index' in k:
            del checkpoint['model'][k]
        if 'relative_position_bias_table' in k:
            relative_position_bias_table = checkpoint['model'][k]
            cls_bias = relative_position_bias_table[:1,:]
            relative_position_bias_table = relative_position_bias_table[1:,:]
            size = int(relative_position_bias_table.shape[0]**0.5)
            img_size = (size+1)//2
            if 'stage_3' in k:
                downsample_ratio = 16
            elif 'stage_4' in k:
                downsample_ratio = 32
            new_img_size = config.DATA.IMG_SIZE//downsample_ratio
            new_size = 2*new_img_size-1
            if new_size == size:
                continue
            relative_position_bias_table = relative_position_bias_table.reshape(size,size,-1)
            relative_position_bias_table = relative_position_bias_table.unsqueeze(0).permute(0,3,1,2)#bs,nhead,h,w
            relative_position_bias_table = torch.nn.functional.interpolate(
                relative_position_bias_table, size=(new_size, new_size), mode='bicubic', align_corners=False)
            relative_position_bias_table = relative_position_bias_table.permute(0,2,3,1)
            relative_position_bias_table = relative_position_bias_table.squeeze(0).reshape(new_size*new_size,-1)
            relative_position_bias_table = torch.cat((cls_bias,relative_position_bias_table),dim=0)
            checkpoint['model'][k] = relative_position_bias_table
    return checkpoint


def load_pretrained(config,model,logger=None,strict=False):
    if logger is not None:
        logger.info(f"-=-=+ Pretraining from: '{config.MODEL.PRETRAINED}' +=-=-")
    checkpoint = torch.load(config.MODEL.PRETRAINED, map_location='cpu')
    if 'model' not in checkpoint:
        if 'state_dict_ema' in checkpoint:
            checkpoint['model'] = checkpoint['state_dict_ema']
        else:
            checkpoint['model'] = checkpoint
    if config.MODEL.DROP_HEAD:
        if 'head.weight' in checkpoint['model'] and 'head.bias' in checkpoint['model']:
            if logger is not None:
                logger.info(f"-=-=+ Dropping Head +=-=-")
            del checkpoint['model']['head.weight']
            del checkpoint['model']['head.bias']
        if 'head.fc.weight' in checkpoint['model'] and 'head.fc.bias' in checkpoint['model']:
            if logger is not None:
                logger.info(f"-=-=+ Dropping Head +=-=-")
            del checkpoint['model']['head.fc.weight']
            del checkpoint['model']['head.fc.bias']
    if config.MODEL.DROP_META:
        if logger is not None:
            logger.info(f"-=-=+ Dropping Meta Head +=-=-")
        for k in list(checkpoint['model']):
            if 'meta' in k:
                del checkpoint['model'][k]
            
    checkpoint = relative_bias_interpolate(checkpoint,config)
    if 'point_coord' in checkpoint['model']:
        if logger is not None:
            logger.info(f"-=-=+ Dropping Point Coords +=-=-")
        del checkpoint['model']['point_coord']
    msg = model.load_state_dict(checkpoint['model'], strict=strict)
    del checkpoint
    torch.cuda.empty_cache()


def load_checkpoint(config, model, optimizer, lr_scheduler, logger, scaler):
    logger.info(f"==============> Resuming from: '{config.MODEL.RESUME}' <==============")
    if config.MODEL.RESUME.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            config.MODEL.RESUME, map_location='cpu', check_hash=True)
    else:
        checkpoint = torch.load(config.MODEL.RESUME, map_location='cpu')
    if 'model' not in checkpoint:
        if 'state_dict_ema' in checkpoint:
            checkpoint['model'] = checkpoint['state_dict_ema']
        else:
            checkpoint['model'] = checkpoint
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    logger.info(msg)
    max_accuracy = 0.0
    if not config.EVAL_MODE and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        config.defrost()
        config.TRAIN.START_EPOCH = checkpoint['epoch'] + 1
        config.freeze()
        if 'amp' in checkpoint and config.AMP_OPT_LEVEL != "O0" and checkpoint['config'].AMP_OPT_LEVEL != "O0":
            scaler.load_state_dict(checkpoint['amp'])
            logger.info(f"Loaded Scaler: {scaler.state_dict()}")
        logger.info(f"===> Loaded successfully: '{config.MODEL.RESUME}' (Epoch {(checkpoint['epoch'] + 1)}) <===")
        if 'max_accuracy' in checkpoint:
            max_accuracy = checkpoint['max_accuracy']

    del checkpoint
    torch.cuda.empty_cache()
    return max_accuracy, scaler


def save_checkpoint(config, epoch, model, max_accuracy, optimizer, lr_scheduler, logger, scaler):
    save_state = {'model': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'lr_scheduler': lr_scheduler.state_dict(),
                  'max_accuracy': max_accuracy,
                  'epoch': epoch,
                  'config': config}
    if config.AMP_OPT_LEVEL != "O0":
        save_state['amp'] = scaler.state_dict()
        logger.info(f"Saved Scaler: {scaler.state_dict()}")

    save_path = os.path.join(config.OUTPUT, f'ckpt_epoch_{(epoch + 1)}.pth')
    logger.info(f"===> Saving checkpoint: '{save_path}' <===")
    torch.save(save_state, save_path)
    logger.info(f"===> Success <===")
    
    
    lastest_save_path = os.path.join(config.OUTPUT, f'latest.pth')
    logger.info(f"===> Saving checkpoint: '{lastest_save_path}' <===")
    torch.save(save_state, lastest_save_path)
    logger.info(f"===> Success <===")


def get_grad_norm(parameters, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm


def auto_resume_helper(output_dir, logger):
    checkpoints = os.listdir(output_dir)
    checkpoints = [ckpt for ckpt in checkpoints if ckpt.endswith('pth')]
    logger.info(f"-=-=+ All checkpoints found in '{output_dir}': {checkpoints} +=-=-")
    if len(checkpoints) > 0:
        latest_checkpoint = max([os.path.join(output_dir, d) for d in checkpoints], key=os.path.getmtime)
        logger.info(f"-=-=+ The latest checkpoint found: '{latest_checkpoint}' +=-=-")
        resume_file = latest_checkpoint
    else:
        resume_file = None
    return resume_file


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt


def load_ext(name, funcs):
    ext = importlib.import_module(name)
    for fun in funcs:
        assert hasattr(ext, fun), f'{fun} missing in module {name}'
    return ext


# Many-Medium-Few-Shot Top-k Accuracy (from Balanced Softmax Loss) # top-k hacked by orumetsu
def shot_acc(preds, labels, train_data, many_shot_thr=100, low_shot_thr=20, acc_per_cls=False):
    
    if isinstance(train_data, np.ndarray):
        training_labels = np.array(train_data).astype(int)
    else:
        training_labels = np.array(train_data.dataset.labels).astype(int)

    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
    elif isinstance(preds, np.ndarray):
        pass
    else:
        raise TypeError('Type ({}) of preds not supported'.format(type(preds)))
    train_class_count = []
    test_class_count = []
    class_correct = []
    for l in np.unique(labels):
        train_class_count.append(len(training_labels[training_labels == l]))
        test_class_count.append(len(labels[labels == l]))
        # hack for top-k, could break
        cur_labels = labels[labels == l]
        topk_preds = preds[labels == l]
        summation = 0
        for i in range(len(cur_labels)):
            if isinstance(topk_preds[i], np.floating): # top-1 goes here
                summation += 1 if cur_labels[i] == topk_preds[i] else 0
            elif isinstance(topk_preds[i], np.ndarray): # # top-k goes here
                summation += 1 if cur_labels[i] in topk_preds[i] else 0
        class_correct.append(summation)

    many_shot = []
    median_shot = []
    low_shot = []
    many_shot_len = []
    median_shot_len = []
    low_shot_len = []
    for i in range(len(train_class_count)):
        if train_class_count[i] > many_shot_thr:
            many_shot.append((class_correct[i] / test_class_count[i]))
            many_shot_len.append(test_class_count[i])
        elif train_class_count[i] < low_shot_thr:
            low_shot.append((class_correct[i] / test_class_count[i]))
            low_shot_len.append(test_class_count[i])
        else:
            median_shot.append((class_correct[i] / test_class_count[i])) 
            median_shot_len.append(test_class_count[i])   
 
    if len(many_shot) == 0:
        many_shot.append(-1)
    if len(median_shot) == 0:
        median_shot.append(-1)
    if len(low_shot) == 0:
        low_shot.append(-1)

    # needed for AverageMeter
    shot_len = (sum(many_shot_len), sum(median_shot_len), sum(low_shot_len))
    
    if acc_per_cls:
        class_accs = [c / cnt for c, cnt in zip(class_correct, test_class_count)]
        return np.mean(many_shot), np.mean(median_shot), np.mean(low_shot), class_accs, shot_len
    else:
        return np.mean(many_shot), np.mean(median_shot), np.mean(low_shot), shot_len