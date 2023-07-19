import os
import time
import argparse
import datetime
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler


from torch.utils.tensorboard import SummaryWriter
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import accuracy, AverageMeter

from config import get_config
from models import build_model
from data import build_loader
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
from logger import create_logger
from utils import load_checkpoint, save_checkpoint, get_grad_norm, auto_resume_helper, reduce_tensor, load_pretrained, shot_acc
from loss.BalancedSoftmaxLoss import BalancedSoftmax


def parse_option():
    parser = argparse.ArgumentParser('MetaFG training and evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs.",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--data-path',default='./imagenet', type=str, help='path to dataset')
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')
    
    parser.add_argument('--num-workers', type=int, 
                        help="num of workers on dataloader ")
    
    parser.add_argument('--lr', type=float, metavar='LR',
                        help='learning rate')
    parser.add_argument('--weight-decay', type=float,
                        help='weight decay (default: 0.05 for adamw)')
    
    parser.add_argument('--min-lr', type=float,
                        help='learning rate')
    parser.add_argument('--warmup-lr', type=float,
                        help='warmup learning rate')
    parser.add_argument('--epochs', type=int,
                        help="epochs")
    parser.add_argument('--warmup-epochs', type=int,
                        help="epochs")
    
    parser.add_argument('--dataset', type=str,
                        help='dataset')
    parser.add_argument('--lr-scheduler-name', type=str,
                        help='lr scheduler name: cosine, linear, step')
    
    parser.add_argument('--pretrain', type=str,
                        help='pretrain')
    
    parser.add_argument('--tensorboard', action='store_true', help='using tensorboard')

    # Balanced Softmax Loss
    parser.add_argument('--use-balanced-softmax-loss', action='store_true', help='Using Balanced Softmax Loss')
    
    
    # distributed training
    # parser.add_argument("--local-rank", type=int, required=True, help='local rank for DistributedDataParallel')

    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config


def main(config):
    dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn = build_loader(config)
    
    # for shot acc
    training_labels = np.array([info["target"] for info in dataset_train.images_info])
    
    logger.info(f"==> Creating model: {config.MODEL.TYPE} / {config.MODEL.NAME} <==")
    model = build_model(config)
    model.cuda()
    logger.info(str(model))

    optimizer = build_optimizer(config, model)
    if config.AMP_OPT_LEVEL != "O0":
        scaler = GradScaler()
        print(f"-= Initialized Scaler: {scaler.state_dict()} =-")
    else:
        scaler = None
        print("-= Amp not used =-")
        
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.LOCAL_RANK], broadcast_buffers=False)
    model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Number of params: {n_parameters}")
    if hasattr(model_without_ddp, 'flops'):
        flops = model_without_ddp.flops()
        logger.info(f"Number of GFLOPs: {flops / 1e9}")
    lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))
    
    if config.USE_BALANCED_SOFTMAX_LOSS:
        logger.info("-= Using Balanced Softmax Loss =-")
        criterion = BalancedSoftmax("./datasets/inaturalist2018/train2018_class_freq.json")
    elif config.AUG.MIXUP > 0.:
        # smoothing is handled with mixup label transform
        logger.info("-= Using Soft Target Cross Entropy Loss =-")
        criterion = SoftTargetCrossEntropy()
    elif config.MODEL.LABEL_SMOOTHING > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=config.MODEL.LABEL_SMOOTHING)
    else:
        criterion = torch.nn.CrossEntropyLoss()
    
    max_accuracy = 0.0
    if config.MODEL.PRETRAINED:
        load_pretrained(config,model_without_ddp,logger)
        if config.EVAL_MODE:
            acc1 = validate(config, data_loader_val, model, training_labels)
            logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.3f}%")
            return

    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT, logger)
        if resume_file:
            if config.MODEL.RESUME:
                logger.warning(f"[Auto-Resume] Changing resume file from '{config.MODEL.RESUME}' to '{resume_file}'")
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f"Auto-resuming from: '{resume_file}'")
        else:
            logger.info(f"No checkpoint found in '{config.OUTPUT}', ignoring auto-resume.")

    if config.MODEL.RESUME:
        logger.info(f"********** Normal Test **********")
        max_accuracy, scaler = load_checkpoint(config, model_without_ddp, optimizer, lr_scheduler, logger, scaler)
        acc1 = validate(config, data_loader_val, model, training_labels)
        logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.3f}%")
        if config.DATA.ADD_META:
            logger.info(f"********** Masked-Meta Test ***********")
            acc1 = validate(config, data_loader_val, model, training_labels, mask_meta=True)
            logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.3f}%")
        if config.EVAL_MODE:
            return

    if config.THROUGHPUT_MODE:
        throughput(data_loader_val, model, logger)
        return

    logger.info("Starting Training")
    start_time = time.time()
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        data_loader_train.sampler.set_epoch(epoch)      
        scaler = train_one_epoch_local_data(config, model, criterion, data_loader_train, optimizer, epoch, mixup_fn, lr_scheduler, scaler)
        if dist.get_rank() == 0 and ((epoch + 1) % config.SAVE_FREQ == 0 or (epoch + 1) == config.TRAIN.EPOCHS):
            save_checkpoint(config, epoch, model_without_ddp, max_accuracy, optimizer, lr_scheduler, logger, scaler)
        
        logger.info(f"********** Normal Test **********")
        acc1 = validate(config, data_loader_val, model, training_labels)
        logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.3f}%")
        max_accuracy = max(max_accuracy, acc1)
        logger.info(f'Max accuracy: {max_accuracy:.3f}%')
        if config.DATA.ADD_META:
            logger.info(f"********** Masked-Meta Test ***********")
            acc1 = validate(config, data_loader_val, model, training_labels, mask_meta=True)
            logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.3f}%")
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time: {}'.format(total_time_str))


def train_one_epoch_local_data(config, model, criterion, data_loader, optimizer, epoch, mixup_fn, lr_scheduler, scaler, tb_logger=None):
    model.train()
    if hasattr(model.module,'cur_epoch'):
        model.module.cur_epoch = epoch
        model.module.total_epoch = config.TRAIN.EPOCHS
    optimizer.zero_grad()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()

    start = time.time()
    end = time.time()
    for idx, data in enumerate(data_loader):
        if config.DATA.ADD_META:
            samples, targets, meta = data
            meta = [m.float() for m in meta]
            meta = torch.stack(meta,dim=0)
            meta = meta.cuda(non_blocking=True)
        else:
            samples, targets= data
            meta = None

        samples = samples.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        if config.AMP_OPT_LEVEL != "O0":
            with autocast():
                if config.DATA.ADD_META:
                    outputs = model(samples, meta)
                else:
                    outputs = model(samples)
        else:
            if config.DATA.ADD_META:
                outputs = model(samples, meta)
            else:
                outputs = model(samples)
        
        if config.TRAIN.ACCUMULATION_STEPS > 1:
            if config.AMP_OPT_LEVEL != "O0":
                with autocast():
                    loss = criterion(outputs, targets)
                    loss = loss / config.TRAIN.ACCUMULATION_STEPS
                scaler.scale(loss).backward()
                
                grad_norm = None # grad_norm values skipped when not stepping
                
                if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
                    scaler.unscale_(optimizer)
                    
                    if config.TRAIN.CLIP_GRAD:
                        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    lr_scheduler.step_update(epoch * num_steps + (idx + 1))

            else:
                loss = criterion(outputs, targets)
                loss = loss / config.TRAIN.ACCUMULATION_STEPS
                loss.backward()

                grad_norm = None # grad_norm values skipped when not stepping

                if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
                    if config.TRAIN.CLIP_GRAD:
                        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
                    optimizer.step()
                    optimizer.zero_grad()
                    lr_scheduler.step_update(epoch * num_steps + (idx + 1))
                
        else:
            optimizer.zero_grad()

            if config.AMP_OPT_LEVEL != "O0":
                with autocast():
                    loss = criterion(outputs, targets)
                scaler.scale(loss).backward()

                scaler.unscale_(optimizer)
                if config.TRAIN.CLIP_GRAD:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)               
                else:
                    grad_norm = get_grad_norm(model.parameters())
                scaler.step(optimizer)
                scaler.update()
                lr_scheduler.step_update(epoch * num_steps + (idx + 1))
            
            else:
                loss = criterion(outputs, targets)
                loss.backward()

                if config.TRAIN.CLIP_GRAD:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
                else:
                    grad_norm = get_grad_norm(model.parameters())
                optimizer.step()
                lr_scheduler.step_update(epoch * num_steps + (idx + 1))

        torch.cuda.synchronize()
        
        time_per_batch = time.time() - end

        if config.TRAIN.ACCUMULATION_STEPS > 1: # scaling loss and batch time if using step accumulation
            loss *= config.TRAIN.ACCUMULATION_STEPS
            time_per_batch *= config.TRAIN.ACCUMULATION_STEPS

        loss_meter.update(loss.item(), targets.size(0))
        batch_time.update(time_per_batch)
        
        if grad_norm:
            norm_meter.update(grad_norm)

        if (idx + 1) % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - (idx + 1))
            
            if config.TRAIN.ACCUMULATION_STEPS > 1:
                 etas /= config.TRAIN.ACCUMULATION_STEPS
            
            logger.info(
                f'Train: [{(epoch + 1)}/{config.TRAIN.EPOCHS}][{(idx + 1)}/{num_steps}]\t'
                f'ETA: {datetime.timedelta(seconds=int(etas))}\t'
                f'lr {lr:.6f}\t'
                f'Batch Time: {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'Loss: {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'Grad Norm: {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                f'Memory: {memory_used:.0f}MB')
        
        end = time.time()

    epoch_time = time.time() - start
    logger.info(f"EPOCH {(epoch + 1)} training takes: {datetime.timedelta(seconds=int(epoch_time))}")

    return scaler


@torch.no_grad()
def validate(config, data_loader, model, training_labels, mask_meta=False):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()

    # construct all AverageMeter
    batch_time = AverageMeter()
    loss_meter = AverageMeter()

    top_k = (1, 3, 5, 10)
    acc_dict = {}
    for cur_k in top_k:
        acc_dict[f"acc{cur_k}"] = AverageMeter()
        acc_dict[f"manyacc{cur_k}"] = AverageMeter()
        acc_dict[f"medacc{cur_k}"] = AverageMeter()
        acc_dict[f"fewacc{cur_k}"] = AverageMeter()

    end = time.time()
    for idx, data in enumerate(data_loader):
        if config.DATA.ADD_META:
            images,target,meta = data
            meta = [m.float() for m in meta]
            meta = torch.stack(meta,dim=0)
            if mask_meta:
                meta = torch.zeros_like(meta)
            meta = meta.cuda(non_blocking=True)
        else:
            images, target = data
            meta = None
        
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        if config.DATA.ADD_META:
            output = model(images,meta)
        else:
            output = model(images)

        # measure and update loss
        loss = criterion(output, target)
        loss = reduce_tensor(loss)
        loss_meter.update(loss.item(), target.size(0))

        #measure and update top-k's
        top_k_accs = accuracy(output, target, topk=top_k)
        for i in range(len(top_k)):
            cur_top_k_acc = reduce_tensor(top_k_accs[i])
            acc_dict[f"acc{top_k[i]}"].update(cur_top_k_acc.item(), target.size(0))

            # many-med-few shot top-k's
            preds = torch.topk(output, k=top_k[i], dim=1).indices
            cur_many_shot, cur_med_shot, cur_few_shot, cur_shot_len = shot_acc(preds, target, training_labels)
            
            # scale to percentage
            cur_many_shot *= 100
            cur_med_shot *= 100
            cur_few_shot *= 100

            # many-med-few-shot update
            if cur_many_shot >= 0.:
                acc_dict[f"manyacc{top_k[i]}"].update(cur_many_shot, cur_shot_len[0])
            if cur_med_shot >= 0.:
                acc_dict[f"medacc{top_k[i]}"].update(cur_med_shot, cur_shot_len[1])
            if cur_few_shot >= 0.:
                acc_dict[f"fewacc{top_k[i]}"].update(cur_few_shot, cur_shot_len[2])

        # update elapsed time
        batch_time.update(time.time() - end)

        if (idx + 1) % config.PRINT_FREQ == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info(
                f'Test: [{(idx + 1)}/{len(data_loader)}]\t'
                f'Batch Time: {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Loss: {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'Acc@1: {acc_dict["acc1"].val:.3f} ({acc_dict["acc1"].avg:.3f}) [{acc_dict["acc1"].count}]\t'
                f'Acc@3: {acc_dict["acc3"].val:.3f} ({acc_dict["acc3"].avg:.3f}) [{acc_dict["acc3"].count}]\t'
                f'Acc@5: {acc_dict["acc5"].val:.3f} ({acc_dict["acc5"].avg:.3f}) [{acc_dict["acc5"].count}]\t'
                f'Acc@10: {acc_dict["acc10"].val:.3f} ({acc_dict["acc10"].avg:.3f}) [{acc_dict["acc10"].count}]\t'
                f'ManyAcc@1: {acc_dict["manyacc1"].val:.3f} ({acc_dict["manyacc1"].avg:.3f}) [{acc_dict["manyacc1"].count}]\t'
                f'ManyAcc@3: {acc_dict["manyacc3"].val:.3f} ({acc_dict["manyacc3"].avg:.3f}) [{acc_dict["manyacc3"].count}]\t'
                f'ManyAcc@5: {acc_dict["manyacc5"].val:.3f} ({acc_dict["manyacc5"].avg:.3f}) [{acc_dict["manyacc5"].count}]\t'
                f'ManyAcc@10: {acc_dict["manyacc10"].val:.3f} ({acc_dict["manyacc10"].avg:.3f}) [{acc_dict["manyacc10"].count}]\t'
                f'MedAcc@1: {acc_dict["medacc1"].val:.3f} ({acc_dict["medacc1"].avg:.3f}) [{acc_dict["medacc1"].count}]\t'
                f'MedAcc@3: {acc_dict["medacc3"].val:.3f} ({acc_dict["medacc3"].avg:.3f}) [{acc_dict["medacc3"].count}]\t'
                f'MedAcc@5: {acc_dict["medacc5"].val:.3f} ({acc_dict["medacc5"].avg:.3f}) [{acc_dict["medacc5"].count}]\t'
                f'MedAcc@10: {acc_dict["medacc10"].val:.3f} ({acc_dict["medacc10"].avg:.3f}) [{acc_dict["medacc10"].count}]\t'
                f'FewAcc@1: {acc_dict["fewacc1"].val:.3f} ({acc_dict["fewacc1"].avg:.3f}) [{acc_dict["fewacc1"].count}]\t'
                f'FewAcc@3: {acc_dict["fewacc3"].val:.3f} ({acc_dict["fewacc3"].avg:.3f}) [{acc_dict["fewacc3"].count}]\t'
                f'FewAcc@5: {acc_dict["fewacc5"].val:.3f} ({acc_dict["fewacc5"].avg:.3f}) [{acc_dict["fewacc5"].count}]\t'
                f'FewAcc@10: {acc_dict["fewacc10"].val:.3f} ({acc_dict["fewacc10"].avg:.3f}) [{acc_dict["fewacc10"].count}]\t'
                f'Memory: {memory_used:.0f}MB')
            
        end = time.time()

    logger.info(
                f'Loss: ({loss_meter.avg:.4f})\t'
                f'Acc@1: ({acc_dict["acc1"].avg:.3f}) [{acc_dict["acc1"].count}]\t'
                f'Acc@3: ({acc_dict["acc3"].avg:.3f}) [{acc_dict["acc3"].count}]\t'
                f'Acc@5: ({acc_dict["acc5"].avg:.3f}) [{acc_dict["acc5"].count}]\t'
                f'Acc@10: ({acc_dict["acc10"].avg:.3f}) [{acc_dict["acc10"].count}]\t'
                f'ManyAcc@1: ({acc_dict["manyacc1"].avg:.3f}) [{acc_dict["manyacc1"].count}]\t'
                f'ManyAcc@3: ({acc_dict["manyacc3"].avg:.3f}) [{acc_dict["manyacc3"].count}]\t'
                f'ManyAcc@5: ({acc_dict["manyacc5"].avg:.3f}) [{acc_dict["manyacc5"].count}]\t'
                f'ManyAcc@10: ({acc_dict["manyacc10"].avg:.3f}) [{acc_dict["manyacc10"].count}]\t'
                f'MedAcc@1: ({acc_dict["medacc1"].avg:.3f}) [{acc_dict["medacc1"].count}]\t'
                f'MedAcc@3: ({acc_dict["medacc3"].avg:.3f}) [{acc_dict["medacc3"].count}]\t'
                f'MedAcc@5: ({acc_dict["medacc5"].avg:.3f}) [{acc_dict["medacc5"].count}]\t'
                f'MedAcc@10: ({acc_dict["medacc10"].avg:.3f}) [{acc_dict["medacc10"].count}]\t'
                f'FewAcc@1: ({acc_dict["fewacc1"].avg:.3f}) [{acc_dict["fewacc1"].count}]\t'
                f'FewAcc@3: ({acc_dict["fewacc3"].avg:.3f}) [{acc_dict["fewacc3"].count}]\t'
                f'FewAcc@5: ({acc_dict["fewacc5"].avg:.3f}) [{acc_dict["fewacc5"].count}]\t'
                f'FewAcc@10: ({acc_dict["fewacc10"].avg:.3f}) [{acc_dict["fewacc10"].count}]\t')
    
    return acc_dict["acc1"].avg


@torch.no_grad()
def throughput(data_loader, model, logger):
    model.eval()

    for idx, (images, _) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        batch_size = images.shape[0]
        for i in range(50):
            model(images)
        torch.cuda.synchronize()
        logger.info(f"-= Throughput averaged 30 times =-")
        tic1 = time.time()
        for i in range(30):
            model(images)
        torch.cuda.synchronize()
        tic2 = time.time()
        logger.info(f"Batch size: {batch_size} Throughput: {30 * batch_size / (tic2 - tic1)}")
        return


if __name__ == '__main__':
    _, config = parse_option()

    # if config.AMP_OPT_LEVEL != "O0":
    #     assert amp is not None, "amp not installed!"

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in env: {rank} / {world_size}")
    else:
        rank = -1
        world_size = -1

    torch.cuda.set_device(config.LOCAL_RANK)
    torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.distributed.barrier()

    seed = config.SEED + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # linear scale the learning rate according to total batch size, may not be optimal
    linear_scaled_lr = config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_min_lr = config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    # gradient accumulation also need to scale the learning rate
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        linear_scaled_lr = linear_scaled_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_warmup_lr = linear_scaled_warmup_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_min_lr = linear_scaled_min_lr * config.TRAIN.ACCUMULATION_STEPS
    config.defrost()
    config.TRAIN.BASE_LR = linear_scaled_lr
    config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
    config.TRAIN.MIN_LR = linear_scaled_min_lr
    config.freeze()

    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=dist.get_rank(), name=f"{config.MODEL.NAME}",local_rank=config.LOCAL_RANK)

    if dist.get_rank() == 0:
        path = os.path.join(config.OUTPUT, "config.json")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"-= Full config saved to: '{path}' =-")

    # print config
    logger.info(config.dump())

    main(config)
