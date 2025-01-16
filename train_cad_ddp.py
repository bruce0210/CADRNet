import os
import argparse
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataset import CADDataLoader
from utils.utils_model import create_logger
from config import config, update_config
from models.model import CADTransformer
from utils.utils_model import OffsetLoss
from eval import do_eval, get_eval_criteria

import torch.multiprocessing
import logging
from collections import OrderedDict

# 设置 multiprocessing 启动方式
try:
    torch.multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass  # 如果已经设置，则忽略

torch.backends.cudnn.benchmark = True
torch.autograd.set_detect_anomaly(True)


def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')
    parser.add_argument('--cfg',
                        type=str,
                        default="config/hrnet48.yaml",
                        help='experiment configure file name'
                        )
    parser.add_argument('--val_only',
                        action="store_true",
                        help='flag to do evaluation on val set')
    parser.add_argument('--test_only',
                        action="store_true",
                        help='flag to do evaluation on test set')
    parser.add_argument('--data_root', type=str,
                        default="/usr/local/CADRNet/data/floorplancad_v2",
                        help='root directory of the dataset')
    parser.add_argument('--embed_backbone', type=str,
                        default="hrnet48",
                        help='embedding backbone model')
    parser.add_argument('--pretrained_model', type=str,
                        default="/usr/local/CADRNet/pretrained_models/hrnetv2_w48_imagenet_pretrained.pth",
                        help='path to the pretrained model')
    parser.add_argument("--local_rank", type=int, default=0,
                        help='local rank for distributed training')
    parser.add_argument("--log_step", type=int,
                        default=100,
                        help='steps for logging')
    parser.add_argument("--img_size", type=int,
                        default=700,
                        help='image size of rasterized image')
    parser.add_argument("--max_prim", type=int,
                        default=12000,
                        help='maximum primitive number for each batch')
    parser.add_argument("--load_ckpt", type=str,
                        default='',
                        help='load checkpoint for testing or evaluation')
    parser.add_argument("--resume_ckpt", type=str,
                        default='',
                        help='resume training from checkpoint')
    parser.add_argument("--log_dir", type=str,
                        default='logs',
                        help='logging directory')
    parser.add_argument('--seed', type=int, default=304,
                        help='random seed')
    parser.add_argument('--debug', action="store_true",
                        help='enable debug mode')
    parser.add_argument('--opts',  # 改为可选参数
                        help="Modify config options using the command-line",
                        default=[],
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()
    return args


def custom_collate(batch):
    """
    自定义 collate_fn，处理可变长度的张量。
    假设返回的样本格式为：
    (image, xy, target, rgb_info, nns, offset_gt, inst_gt, index, basename)
    """
    images = torch.stack([item[0] for item in batch], dim=0)
    xy_list = [item[1] for item in batch]  # list of tensors with shape [N_i, 2]
    target_list = [item[2] for item in batch]  # list of tensors with shape [N_i, 1]
    rgb_info = torch.stack([item[3] for item in batch], dim=0)
    nns = torch.stack([item[4] for item in batch], dim=0)
    offset_gt = torch.stack([item[5] for item in batch], dim=0)
    inst_gt = torch.stack([item[6] for item in batch], dim=0)
    index = [item[7] for item in batch]  # list of tensors
    basename = [item[8] for item in batch]  # list of strings

    # 找到最大长度
    max_len = max([xy.size(0) for xy in xy_list])

    # 初始化填充张量
    padded_xy = torch.zeros(len(batch), max_len, 2).to(xy_list[0].device)
    padded_target = torch.zeros(len(batch), max_len).long().to(target_list[0].device)

    for i in range(len(batch)):
        n = xy_list[i].size(0)
        padded_xy[i, :n, :] = xy_list[i]
        padded_target[i, :n] = target_list[i].squeeze(1)  # 假设 target 是 [N_i, 1]

    return images, padded_xy, padded_target, rgb_info, nns, offset_gt, inst_gt, index, basename


def adjust_state_dict(state_dict, distributed):
    """
    调整 state_dict 的键名以匹配当前模型的包装方式。
    如果 distributed 为 True，确保键名有 'module.' 前缀。
    如果 distributed 为 False，确保键名没有 'module.' 前缀。
    """
    new_state_dict = OrderedDict()
    if distributed:
        # 如果没有 'module.' 前缀，则添加
        if not all(k.startswith('module.') for k in state_dict.keys()):
            for k, v in state_dict.items():
                new_state_dict['module.' + k] = v
        else:
            new_state_dict = state_dict
    else:
        # 如果有 'module.' 前缀，则移除
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
    return new_state_dict


def main():
    args = parse_args()
    cfg = update_config(config, args)

    # 正规化 log_dir，避免路径重复
    cfg.log_dir = os.path.normpath(cfg.log_dir)

    # 强制设置 rgb_dim 为 0，以避免加载不存在的 npy_rgb 文件
    cfg.rgb_dim = 0

    # 设置日志
    os.makedirs(cfg.log_dir, exist_ok=True)
    if args.val_only:
        logger = create_logger(cfg.log_dir, 'val')
    elif args.test_only:
        logger = create_logger(cfg.log_dir, 'test')
    else:
        logger = create_logger(cfg.log_dir, 'train')

    # Determine if distributed training is being used
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    distributed = num_gpus > 1

    if distributed:
        # Initialize distributed training
        torch.cuda.set_device(args.local_rank)
        device = torch.device(f'cuda:{args.local_rank}')
        logger.info(f'Distributed training on GPU {args.local_rank}')
        torch.distributed.init_process_group(
            backend="nccl",
            init_method="env://",
        )
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f'Single GPU training on {device}')

    # Log available GPUs and mapping
    available_gpus = torch.cuda.device_count()
    logger.info(f'Available GPUs: {available_gpus}')
    logger.info(f'Set device to: {device}')

    if args.local_rank >= available_gpus:
        logger.error(f'Invalid local_rank {args.local_rank} for available GPUs {available_gpus}')
        raise ValueError(f'Invalid local_rank {args.local_rank} for available GPUs {available_gpus}')

    # Create Model
    model = CADTransformer(cfg)
    CE_loss = torch.nn.CrossEntropyLoss().to(device)

    # Create Optimizer
    if cfg.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=cfg.weight_decay
        )
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg.learning_rate, momentum=0.9)

    # Move model to device
    model = model.to(device)

    if distributed:
        # Wrap the model with DistributedDataParallel
        model = torch.nn.parallel.DistributedDataParallel(
            module=model,
            broadcast_buffers=False,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True  # 设置为 True 以允许未使用的参数
        )
        logger.info('Model wrapped with DistributedDataParallel')
    else:
        logger.info('Model is not wrapped with DistributedDataParallel')

    model.train()

    # **初始化所有 Dataloaders 先于模式判断**
    # Set up Dataloaders
    # Using DistributedSampler for training in distributed mode
    train_dataset = CADDataLoader(split='train', do_norm=cfg.do_norm, cfg=cfg)
    logger.info(f'Training dataset size: {len(train_dataset)}')

    if distributed and not (args.val_only or args.test_only):
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
    else:
        train_sampler = None

    train_dataloader = DataLoader(
        dataset=train_dataset,
        sampler=train_sampler,
        batch_size=cfg.batch_size,
        num_workers=cfg.WORKERS,
        drop_last=True,
        collate_fn=custom_collate  # 使用自定义 collate_fn
    )
    logger.info(f'Training DataLoader initialized with batch size {cfg.batch_size}')

    # 初始化验证集
    val_dataset = CADDataLoader(split='val', do_norm=cfg.do_norm, cfg=cfg)
    logger.info(f'Validation dataset size: {len(val_dataset)}')
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=cfg.test_batch_size,
        shuffle=False,
        num_workers=cfg.WORKERS,
        drop_last=False,
        collate_fn=custom_collate  # 使用自定义 collate_fn
    )
    logger.info(f'Validation DataLoader initialized with batch size {cfg.test_batch_size}')

    # 初始化测试集
    test_dataset = CADDataLoader(split='test', do_norm=cfg.do_norm, cfg=cfg)
    logger.info(f'Test dataset size: {len(test_dataset)}')
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=cfg.test_batch_size,
        shuffle=False,
        num_workers=cfg.WORKERS,
        drop_last=False,
        collate_fn=custom_collate  # 使用自定义 collate_fn
    )
    logger.info(f'Test DataLoader initialized with batch size {cfg.test_batch_size}')

    # **加载/恢复检查点**
    start_epoch = 0
    if not (args.val_only or args.test_only):
        # Load checkpoint for training
        if cfg.load_ckpt != '':
            if os.path.exists(cfg.load_ckpt):
                checkpoint = torch.load(cfg.load_ckpt, map_location=device)
                state_dict = checkpoint['model_state_dict']
                adjusted_state_dict = adjust_state_dict(state_dict, distributed)
                model.load_state_dict(adjusted_state_dict)
                logger.info(f"=> Loaded checkpoint '{cfg.load_ckpt}' (epoch {checkpoint['epoch']})")
                # 打印模型部分参数范数
                for name, param in model.named_parameters():
                    logger.info(f'Parameter {name} norm: {param.norm().item()}')
                    break  # 只打印第一个参数
            else:
                logger.error(f"=> Failed: no checkpoint found at '{cfg.load_ckpt}'")
                exit(0)

        if cfg.resume_ckpt != '':
            if os.path.exists(cfg.resume_ckpt):
                checkpoint = torch.load(cfg.resume_ckpt, map_location=device)
                start_epoch = checkpoint['epoch']
                state_dict = checkpoint['model_state_dict']
                adjusted_state_dict = adjust_state_dict(state_dict, distributed)
                model.load_state_dict(adjusted_state_dict)
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                # Move optimizer state to device
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if torch.is_tensor(v):
                            state[k] = v.to(device)
                logger.info(f"=> Resumed checkpoint '{cfg.resume_ckpt}' (epoch {checkpoint['epoch']})")
            else:
                logger.error(f"=> Failed: no checkpoint found at '{cfg.resume_ckpt}'")
                exit(0)

    # **处理测试模式**
    if args.test_only and (args.local_rank == 0 or not distributed):
        logger.info("=> Conducting do_eval on test set")
        # Determine which checkpoint to load
        test_ckpt = args.load_ckpt if args.load_ckpt else os.path.join(cfg.log_dir, 'best_model.pth')
        if not os.path.exists(test_ckpt):
            test_ckpt = os.path.join(cfg.log_dir, 'last_model.pth')
            logger.warning(f"=> 'best_model.pth' not found. Using 'last_model.pth' instead.")

        logger.info(f"Attempting to load checkpoint from: {test_ckpt}")  # 添加调试日志
        if os.path.exists(test_ckpt):
            checkpoint = torch.load(test_ckpt, map_location=device)
            state_dict = checkpoint['model_state_dict']
            adjusted_state_dict = adjust_state_dict(state_dict, distributed)
            model.load_state_dict(adjusted_state_dict)
            logger.info(f"=> Loaded checkpoint '{test_ckpt}' (epoch {checkpoint['epoch']})")
        else:
            logger.error(f"=> Failed: no checkpoint found at '{test_ckpt}'")
            exit(0)

        model.eval()  # 设置模型为评估模式
        with torch.no_grad():
            eval_F1 = do_eval(model, test_dataloader, logger, cfg)
        logger.info(f"Test F1 Score: {eval_F1:.2%}")
        exit(0)

    # **处理验证模式**
    if args.val_only and (args.local_rank == 0 or not distributed):
        logger.info("=> Conducting do_eval on validation set")
        # Determine which checkpoint to load
        val_ckpt = args.load_ckpt if args.load_ckpt else os.path.join(cfg.log_dir, 'best_model.pth')
        if not os.path.exists(val_ckpt):
            val_ckpt = os.path.join(cfg.log_dir, 'last_model.pth')
            logger.warning(f"=> 'best_model.pth' not found. Using 'last_model.pth' instead.")

        logger.info(f"Attempting to load checkpoint from: {val_ckpt}")  # 添加调试日志
        if os.path.exists(val_ckpt):
            checkpoint = torch.load(val_ckpt, map_location=device)
            state_dict = checkpoint['model_state_dict']
            adjusted_state_dict = adjust_state_dict(state_dict, distributed)
            model.load_state_dict(adjusted_state_dict)
            logger.info(f"=> Loaded checkpoint '{val_ckpt}' (epoch {checkpoint['epoch']})")
        else:
            logger.error(f"=> Failed: no checkpoint found at '{val_ckpt}'")
            exit(0)

        model.eval()  # 设置模型为评估模式
        with torch.no_grad():
            eval_F1 = do_eval(model, val_dataloader, logger, cfg)
        logger.info(f"Validation F1 Score: {eval_F1:.2%}")
        exit(0)

    # **继续训练模式**
    def bn_momentum_adjust(m, momentum):
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
            m.momentum = momentum

    best_F1, eval_F1 = 0.0, 0.0
    best_epoch = 0
    global_epoch = start_epoch

    logger.info(f"> Starting training from epoch {start_epoch}")

    for epoch in range(start_epoch, cfg.epoch):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)  # 修改: 仅在 train_sampler 存在时调用

        logger.info(f"\nEpoch {epoch + 1}/{cfg.epoch}")
        lr = max(cfg.learning_rate * (cfg.lr_decay ** (epoch // cfg.step_size)),
                 cfg.LEARNING_RATE_CLIP)
        if epoch <= cfg.epoch_warmup:
            lr = cfg.learning_rate_warmup

        logger.info(f'Learning rate: {lr}')
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        momentum = cfg.MOMENTUM_ORIGINAL * (cfg.MOMENTUM_DECCAY ** (epoch // cfg.step_size))
        if momentum < 0.01:
            momentum = 0.01
        logger.info(f'BN momentum updated to: {momentum}')
        if distributed:
            model.module.apply(lambda m: bn_momentum_adjust(m, momentum))
        else:
            model.apply(lambda m: bn_momentum_adjust(m, momentum))

        model.train()

        # Training loop
        with tqdm(train_dataloader, total=len(train_dataloader), smoothing=0.9) as _tqdm:
            for i, (image, xy, target, rgb_info, nns, offset_gt, inst_gt, index, basename) in enumerate(_tqdm):
                optimizer.zero_grad()

                # 将所有输入数据移动到设备上
                image = image.to(device)
                xy = xy.to(device)
                target = target.to(device)
                rgb_info = rgb_info.to(device)
                nns = nns.to(device)
                offset_gt = offset_gt.to(device)
                inst_gt = inst_gt.to(device)

                # 前向传播
                seg_pred = model(image, xy, rgb_info, nns)
                seg_pred = seg_pred.contiguous().view(-1, cfg.num_class + 1)
                target = target.view(-1)

                loss_seg = CE_loss(seg_pred, target)
                loss = loss_seg
                loss.backward()
                optimizer.step()
                _tqdm.set_postfix(loss=loss.item(), l_seg=loss_seg.item())

                if i % args.log_step == 0 and (args.local_rank == 0 or not distributed):
                    logger.info(f'Train loss: {round(loss.item(), 5)}, loss seg: {round(loss_seg.item(), 5)})')

        # Save last model
        if (args.local_rank == 0 or not distributed):
            logger.info('Saving last model...')
            savepath = os.path.join(cfg.log_dir, 'last_model.pth')
            state = {
                'epoch': epoch,
                'best_F1': best_F1,
                'best_epoch': best_epoch,
                'model_state_dict': model.module.state_dict() if distributed else model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, savepath)

        # Validation
        eval = get_eval_criteria(epoch)
        if eval and (args.local_rank == 0 or not distributed):
            logger.info('Performing validation...')
            model.eval()  # 设置模型为评估模式
            with torch.no_grad():
                eval_F1 = do_eval(model, val_dataloader, logger, cfg)

            # Save best model
            if eval_F1 > best_F1:
                best_F1 = eval_F1
                best_epoch = epoch
                logger.info(f'Saving best model... Best F1: {best_F1:.2%}, Best Epoch: {best_epoch}')
                savepath = os.path.join(cfg.log_dir, 'best_model.pth')
                state = {
                    'epoch': epoch,
                    'best_F1': best_F1,
                    'best_epoch': best_epoch,
                    'model_state_dict': model.module.state_dict() if distributed else model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)

        global_epoch += 1


if __name__ == '__main__':
    main()
