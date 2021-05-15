import time
import torch
import pandas as pd
import sys
code_path = "/home/hustwen/sun_chen/Optical_VOC/"
sys.path.append(code_path+"AutoMMdet")
from HPO.MMdetProcess import MMdet_Pipeline

fewshot_list = [5,10,15,20]
record = []
gpu_id = 0
valid = False

cfg_path = code_path+"/AutoMMdet/Optical/Default_Cascade/Cascade_ResNet_50.py"
current_time = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
save_dir = code_path + "/Work_dir/"+str(current_time)+"/optical_default_train/"
data_root = code_path + "/Dataset/Optical/optical-img/test-A-image/"

for fewshot in fewshot_list:
    fewshot_dir = save_dir + str(fewshot)+"_shot/"
    for trial in range(10):
        work_dir = fewshot_dir+"trail_"+str(trial+1)
        anno_root = code_path + "Dataset/Optical/coco/" + str(fewshot) + "_shot/split-"+str(trial+1)

        MMdetpipe = MMdet_Pipeline(cfg_path,work_dir,data_root,anno_root,gpu_id,valid)
        MMdetpipe.cfg.checkpoint_config = dict(interval=2)
        MMdetpipe.cfg.runner = dict(type='EpochBasedRunner', max_epochs=10)
        MMdetpipe.cfg.data.test.ann_file = anno_root+"/instances_test.json"
        MMdetpipe.cfg.load_from = code_path + '/checkpoints/optical/epoch_22.pth'
        try:
            MMdetpipe.MMdet_train(MMdetpipe.cfg)
            ckp_path = work_dir+"/latest.pth"
            result = MMdetpipe.MMdet_Valid(ckp_path,MMdetpipe.cfg,False,False,None)
        except:
            result = {"bbox_mAP_50":0}
            torch.cuda.empty_cache()

        record.append([fewshot,trial,result["bbox_mAP_50"]])
        best_record_pd = pd.DataFrame(record,columns=["shot","trial","mAP_50"])
        save_path = work_dir+"/"+str(fewshot)+"-shot"+"_trial-"+str(trial+1)+"_record.csv"
        best_record_pd.to_csv(save_path)