import time
import torch
from HPO.MMdetProcess import MMdet_Pipeline

fewshot_list = [5]

cfg_path = "./Cacade_ResNet_50.py"
Dataset_path = 'None'
Save_path = "None"
checkpoint_path = "None"
for fewshot in fewshot_list:
    for trial in range(10):
        anno_root = Dataset_path + "/coco/"+str(fewshot)+"_shot/split-"+str(trial+1)
        current_time = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
        work_dir = Save_path + "/work_dir/"+str(current_time)+"/default_train/"        
        gpu_id = 0
        valid = False
        MMdetpipe = MMdet_Pipeline(cfg_path,work_dir,anno_root,gpu_id,valid)

        MMdetpipe.cfg.checkpoint_config = dict(interval=2)
        MMdetpipe.cfg.runner = dict(type='EpochBasedRunner', max_epochs=1)
        MMdetpipe.cfg.data.test.ann_file = anno_root+"/instances_val.json"
        MMdetpipe.cfg.load_from = checkpoint_path + '/checkpoints/epoch_22.pth'

        try:
            MMdetpipe.MMdet_train(MMdetpipe.cfg)
            ckp_path = work_dir+"/latest.pth"
            result = MMdetpipe.MMdet_Valid(ckp_path,MMdetpipe.cfg,False,False,None)
        except:
            result = 0
            torch.cuda.empty_cache()