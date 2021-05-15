code_path = "/home/hustwen/sun_chen/Optical_VOC/"
sys.path.append(code_path+"AutoMMdet")

import time
import torch 
import pandas as pd
import sys
from HPO.MMdetProcess import MMdet_Pipeline

fewshot_list = [5,10,15,20]
cfg_path = code_path+"AutoMMdet/PascalVOC_lite/Default_Fewshot/faster_rcnn_r50.py"

record = []
for fewshot in fewshot_list:
    for trial in range(10):
        anno_root = code_path + "/Dataset/PascalVOC_lite/coco_test/coco_test_"+str(fewshot)+"/split-"+str(trial+1)
        current_time = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
        work_dir = code_path + "/Work_dir/"+str(current_time)+"/voc_default_train/"        

        gpu_id = 0
        valid = False
        MMdetpipe = MMdet_Pipeline(cfg_path,work_dir,anno_root,gpu_id,valid)

        MMdetpipe.cfg.checkpoint_config = dict(interval=2)
        MMdetpipe.cfg.runner = dict(type='EpochBasedRunner', max_epochs=10)
        MMdetpipe.cfg.data.test.ann_file = anno_root+"/instances_val.json"
        MMdetpipe.cfg.load_from = code_path+'/checkpoints/pascalvoc_lite/fewshot_checkpoints/r50.pth'

        try:
            MMdetpipe.MMdet_train(MMdetpipe.cfg)
            ckp_path = work_dir+"/latest.pth"
            result = MMdetpipe.MMdet_Valid(ckp_path,MMdetpipe.cfg,False,False,None)
        except:
            result = 0
            torch.cuda.empty_cache()
        record.append([fewshot,trial,result["bbox_mAP_50"]])
        best_record_pd = pd.DataFrame(record)
        save_path = work_dir+"/"+str(fewshot)+"-shot"+"_trial-"+str(trial+1)+"_record.csv"
        best_record_pd.to_csv(save_path)
