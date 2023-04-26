import datetime
import sys
import math
import time
from os import path as osp

import torch.cuda

from datasets import build_dataloader, build_dataset
from datasets.data_sampler import EnlargedSampler

from models import build_model
from utils import (AvgTimer, MessageLogger, get_env_info, get_root_logger,
                   init_tb_logger)
from utils.options import dict2str, parse_options


def init_tb_loggers(opt):
    tb_logger = init_tb_logger(opt['path']['experiments_root'])
    return tb_logger


def create_train_val_dataloader(opt, logger):
    train_set, val_set = None, None
    # create train and val datasets
    for dataset_name, dataset_opt in opt['datasets'].items():
        if isinstance(dataset_opt, int):  # batch_size, num_worker
            continue
        if dataset_name.startswith('train'):
            dataset_enlarge_ratio = dataset_opt.get('dataset_enlarge_ratio', 1)
            if train_set is None:
                train_set = build_dataset(dataset_opt)
            else:
                train_set += build_dataset(dataset_opt)
        elif dataset_name.startswith('val') or dataset_name.startswith('test'):
            if val_set is None:
                val_set = build_dataset(dataset_opt)
            else:
                val_set += build_dataset(dataset_opt)

    # create train and val dataloaders
    train_sampler = EnlargedSampler(train_set, opt['world_size'], opt['rank'], dataset_enlarge_ratio)
    train_loader = build_dataloader(
        train_set,
        opt['datasets'],
        'train',
        num_gpu=opt['num_gpu'],
        dist=opt['dist'],
        sampler=train_sampler,
        seed=opt['manual_seed'])
    batch_size = opt['datasets']['batch_size']
    num_iter_per_epoch = math.ceil(
        len(train_set) * dataset_enlarge_ratio / batch_size)
    total_epochs = int(opt['train']['total_epochs'])
    total_iters = total_epochs * num_iter_per_epoch
    logger.info('Training statistics:'
                f'\n\tNumber of train images: {len(train_set)}'
                f'\n\tDataset enlarge ratio: {dataset_enlarge_ratio}'
                f'\n\tBatch size: {batch_size}'
                f'\n\tWorld size (gpu number): {opt["world_size"]}'
                f'\n\tRequire iter number per epoch: {num_iter_per_epoch}'
                f'\n\tTotal epochs: {total_epochs}; iters: {total_iters}.')

    val_loader = build_dataloader(
        val_set, opt['datasets'], 'val', num_gpu=opt['num_gpu'], dist=opt['dist'], sampler=None,
        seed=opt['manual_seed'])
    logger.info('Validation statistics:'
                f'\n\tNumber of val images: {len(val_set)}')

    return train_loader, train_sampler, val_loader, total_epochs, total_iters


def train_pipeline(root_path):
    # parse options, set distributed setting, set random seed
    opt = parse_options(root_path, is_train=True)
    opt['root_path'] = root_path

    # WARNING: should not use get_root_logger in the above codes, including the called functions
    # Otherwise the logger will not be properly initialized
    log_file = osp.join(opt['path']['log'], f"train_{opt['name']}.log")
    logger = get_root_logger(log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))
    # initialize tensorboard logger
    tb_logger = init_tb_loggers(opt)

    # create train and validation dataloaders
    result = create_train_val_dataloader(opt, logger)
    train_loader, train_sampler, val_loader, total_epochs, total_iters = result
    opt['train']['total_iter'] = total_iters

    # create model
    model = build_model(opt)

    # create message logger (formatted outputs)
    msg_logger = MessageLogger(opt, model.curr_iter, tb_logger)

    # training
    logger.info(f'Start training from epoch: {model.curr_epoch}, iter: {model.curr_iter}')
    data_timer, iter_timer = AvgTimer(), AvgTimer()
    start_time = time.time()

    try:
        while model.curr_epoch < total_epochs:
            model.curr_epoch += 1
            train_sampler.set_epoch(model.curr_epoch)
            for train_data in train_loader:
                data_timer.record()

                model.curr_iter += 1

                # process data and forward pass
                model.feed_data(train_data)
                # backward pass
                model.optimize_parameters()
                # update model per iteration
                model.update_model_per_iteration()

                iter_timer.record()
                if model.curr_iter == 1:
                    # reset start time in msg_logger for more accurate eta_time
                    # not work in resume mode
                    msg_logger.reset_start_time()
                # log
                if model.curr_iter % opt['logger']['print_freq'] == 0:
                    log_vars = {'epoch': model.curr_epoch, 'iter': model.curr_iter}
                    log_vars.update({'lrs': model.get_current_learning_rate()})
                    log_vars.update({'time': iter_timer.get_avg_time(), 'data_time': data_timer.get_avg_time()})
                    log_vars.update(model.get_loss_metrics())
                    msg_logger(log_vars)

                # save models and training states
                if model.curr_iter % opt['logger']['save_checkpoint_freq'] == 0:
                    logger.info('Saving models and training states.')
                    model.save_model(net_only=False, best=False)

                # validation
                if opt.get('val') is not None and (model.curr_iter % opt['val']['val_freq'] == 0):
                    logger.info('Start validation.')
                    torch.cuda.empty_cache()
                    model.validation(val_loader, tb_logger)

                data_timer.start()
                iter_timer.start()
                # end of iter
            # update model per epoch
            model.update_model_per_epoch()
            # end of epoch
        # end of training
    except KeyboardInterrupt:
        # save the current model
        logger.info('Keyboard interrupt. Save model and exit...')
        model.save_model(net_only=False, best=False)
        model.save_model(net_only=True, best=True)
        sys.exit(0)

    consumed_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    logger.info(f'End of training. Time consumed: {consumed_time}')
    logger.info(f'Last Validation.')
    if opt.get('val') is not None:
        model.validation(val_loader, tb_logger)
    logger.info('Save the best model.')
    model.save_model(net_only=True, best=True)  # save the best model

    if tb_logger:
        tb_logger.close()


if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir))
    train_pipeline(root_path)
