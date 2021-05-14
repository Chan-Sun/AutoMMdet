import os.path as osp
import time
import numpy as np
import mmcv
from mmcv import Config
from mmcv.utils import get_git_hash
from mmdet.utils import collect_env, get_root_logger

from mmdet import __version__
from mmdet.apis import (train_detector,single_gpu_test)
from mmdet.models import build_detector
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel
from mmcv.runner import (load_checkpoint,wrap_fp16_model)
from mmdet.datasets import (build_dataloader, build_dataset,get_loading_pipeline)
from mmdet.core.evaluation import eval_map
import pandas as pd  
from tqdm import tqdm
import time 

class MMdet_Pipeline():
    def __init__(self,cfg_path,work_dir,anno_root,gpu_id,valid = False):
        self.valid = valid
        self.gpu_id = gpu_id
        self.cfg_path = cfg_path
        self.cfg = Config.fromfile(cfg_path)
        self.cfg.work_dir = work_dir
        self.cfg.gpu_ids=[self.gpu_id]
        self.cfg.data.train.ann_file = anno_root+"/instances_train.json"
        self.cfg.data.val.ann_file = anno_root+"/instances_val.json"
        self.cfg.data.test.ann_file = anno_root+"/instances_test.json"

        mmcv.mkdir_or_exist(osp.abspath(work_dir))
    
    def MMdet_train(self,config):
        ### only train and cancel validate
        # create work_dir
        # dump config
        config.dump(osp.join(config.work_dir, osp.basename(self.cfg_path)))
        # init the logger before other steps
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        log_file = osp.join(config.work_dir, f'{timestamp}.log')
        logger = get_root_logger(log_file=log_file, log_level=config.log_level)

        # init the meta dict to record some important information such as
        # environment info and seed, which will be logged
        meta = dict()
        # log env info
        env_info_dict = collect_env()
        env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
        dash_line = '-' * 60 + '\n'
        logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                    dash_line)
        meta['env_info'] = env_info
        meta['config'] = config.pretty_text
        # log some basic info
        logger.info(f'Distributed training: False')
        logger.info(f'Config:\n{config.pretty_text}')
        # set random seeds
        seed = None
        config.seed = seed
        meta['seed'] = seed
        meta['exp_name'] = osp.basename(self.cfg_path)

        model = build_detector(
            config.model,
            train_cfg=config.get('train_cfg'),
            test_cfg=config.get('test_cfg'))

        datasets = [build_dataset(config.data.train)]
        if config.checkpoint_config is not None:
            # save mmdet version, config file content and class names in
            # checkpoints as meta data
            config.checkpoint_config.meta = dict(
                mmdet_version=__version__ + get_git_hash()[:7],
                CLASSES=datasets[0].CLASSES)
        # add an attribute for visualization convenience
        model.CLASSES = datasets[0].CLASSES
        train_detector(
            model,
            datasets,
            config,
            distributed=None,
            validate=self.valid,
            timestamp=timestamp,
            meta=meta)
            
    def MMdet_Valid(self,ckp_path,config,result_save_path,show,show_dir):
        config.model.pretrained = None
        if config.model.get('neck'):
            if isinstance(config.model.neck, list):
                for neck_cfg in config.model.neck:
                    if neck_cfg.get('rfp_backbone'):
                        if neck_cfg.rfp_backbone.get('pretrained'):
                            neck_cfg.rfp_backbone.pretrained = None
            elif config.model.neck.get('rfp_backbone'):
                if config.model.neck.rfp_backbone.get('pretrained'):
                    config.model.neck.rfp_backbone.pretrained = None

        # in case the test dataset is concatenated
        samples_per_gpu = 1

        config.data.test.test_mode = True
        samples_per_gpu = config.data.test.pop('samples_per_gpu', 1)

        # build the dataloader
        dataset = build_dataset(config.data.test)
        data_loader = build_dataloader(
            dataset,
            samples_per_gpu=samples_per_gpu,
            workers_per_gpu=config.data.workers_per_gpu,
            dist=False,
            shuffle=False)

        # build the model and load che`ckp`oint
        config.model.train_cfg = None
        model = build_detector(config.model, test_cfg=config.get('test_cfg'))
        fp16_cfg = config.get('fp16', None)
        if fp16_cfg is not None:
            wrap_fp16_model(model)
        checkpoint = load_checkpoint(model, ckp_path, map_location='cpu')
        # slightly increase the inference speed
        model = fuse_conv_bn(model)
        # old versions did not save class info in checkpoints, this walkaround is
        # for backward compatibility
        if 'CLASSES' in checkpoint['meta']:
            model.CLASSES = checkpoint['meta']['CLASSES']
        else:
            model.CLASSES = dataset.CLASSES
        
        model = MMDataParallel(model, device_ids=[self.gpu_id])
        # model = MMDataParallel(model, device_ids=[1])
        
        outputs = single_gpu_test(model, data_loader,show,show_dir)    

        eval_kwargs = config.get('evaluation', {}).copy()
        if result_save_path:
            # print(f'\nwriting results to {args.out}')
            mmcv.dump(outputs, result_save_path+"/result.pkl")
            pkl_path = result_save_path+"/result.pkl"
            # self.picmAPAnalysis(pkl_path,config,result_save_path+"/result.csv")
            #### write a function to store mAP result on every picture
        for key in ['interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                    'rule']:
            eval_kwargs.pop(key, None)
        eval_kwargs.update(dict(metric="bbox"))
        result = dataset.evaluate(outputs,**eval_kwargs)
        return result

    def picmAPAnalysis(self,pkl_path,test_cfg,csv_save_path):
        test_cfg.data.test.pipeline = get_loading_pipeline(test_cfg.data.train.pipeline)
        dataset = build_dataset(test_cfg.data.test)
        results = mmcv.load(pkl_path)
        mAPs = []
        for i, (result, ) in tqdm(enumerate(zip(results))):
        # self.dataset[i] should not call directly
        # because there is a risk of mismatch
            data_info = dataset.prepare_train_img(i)
            image_name = data_info["img_info"]["file_name"]
            mAP = self.bbox_map_eval(result, data_info['ann_info'])
            mAPs.append([image_name,mAP])
        mAPs = pd.DataFrame(mAPs)
        mAPs.to_csv(csv_save_path)

    def bbox_map_eval(self,det_result, annotation):
        # use only bbox det result
        if isinstance(det_result, tuple):
            bbox_det_result = [det_result[0]]
        else:
            bbox_det_result = [det_result]
        # mAP
        iou_thrs = np.linspace(
            .5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
        mean_aps = []
        for thr in iou_thrs:
            mean_ap, _ = eval_map(
                bbox_det_result, [annotation], iou_thr=thr, logger='silent')
            mean_aps.append(mean_ap)
        return sum(mean_aps) / len(mean_aps)