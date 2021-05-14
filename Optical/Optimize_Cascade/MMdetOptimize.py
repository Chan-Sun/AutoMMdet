from hyperopt import tpe,rand,anneal
import pandas as pd
import time
from HPO.Selecct_HPO import MMdet_HPO

code_path = "/home/hustwen/sun_chen/Optical_VOC/"

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
        save_path = work_dir+"/"+str(shot)+"_shot_"+optimize_algo+"_trial_"+str(trial+1)+"best_record.csv"
        best_record_pd.to_csv(save_path)