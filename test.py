from os import path as osp

from datasets import build_dataloader, build_dataset
from models import build_model
from utils import get_env_info, get_root_logger, get_time_str
from utils.options import dict2str, parse_options


def test_pipeline(root_path):
    # parse options, set distributed setting, set random seed
    opt = parse_options(root_path, is_train=False)

    # initialize loggers
    log_file = osp.join(opt['path']['log'], f"test_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))

    # create test dataset and dataloader
    test_loaders = []
    for _, dataset_opt in sorted(opt['datasets'].items()):
        test_set = build_dataset(dataset_opt)
        test_loader = build_dataloader(
            test_set, dataset_opt, phase='val', num_gpu=opt['num_gpu'], dist=opt['dist'], sampler=None, seed=opt['manual_seed'])
        logger.info(f"Number of test images in {dataset_opt['name']}: {len(test_set)}")
        test_loaders.append(test_loader)

    # create model
    model = build_model(opt)

    for test_loader in test_loaders:
        test_set_name = test_loader.dataset.__class__.__name__
        logger.info(f'Testing {test_set_name}...')
        model.validation(test_loader, tb_logger=None, update=False)


if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir))
    test_pipeline(root_path)
