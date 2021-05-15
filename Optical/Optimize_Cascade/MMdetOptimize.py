code_path = "/home/hustwen/sun_chen/Optical_VOC/"
sys.path.append(code_path+"AutoMMdet")

from hyperopt import tpe,rand,anneal
import pandas as pd
import time
import sys
from HPO.Selecct_HPO import MMdet_HPO

config_path = code_path+ "/AutoMMdet/Optical/Optimize_Cascade/Cascade_Configs/Cascade_ResNet_50.py"

max_eval = 40
fewshot_list = [5,10,15,20]
best_record = []
trial_times = 10
###choose from "tpe","rand","HRA","anneal"
optimize_algo = "tpe"

gpu_id = 0
valid = False

for shot in fewshot_list:
    for trial in range(trial_times):
        anno_root = code_path + "Dataset/Optical/coco/" + str(shot) + "_shot/split-"+str(trial+1)

        current_time = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
        work_dir = code_path + "/Work_dir/"+current_time+"/Optical_"+optimize_algo        
        optical_mmdet = MMdet_HPO(config_path,work_dir,max_eval,anno_root,gpu_id,valid,optimizer=optimize_algo)
        best_config, best_loss = optical_mmdet.HPO()
        best_record.append([best_config, best_loss])
        best_record_pd = pd.DataFrame(best_record)
        save_path = work_dir+"/"+str(shot)+"-shot_"+optimize_algo+"_trial-"+str(trial+1)+"_best_record.csv"
        best_record_pd.to_csv(save_path)