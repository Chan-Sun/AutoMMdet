import pandas as pd
import time
import sys
code_path = "/home/dlsuncheng/Optical_VOC/"

sys.path.append(code_path+"AutoMMdet")
from HPO.Select_HPO import MMdet_HPO

max_eval = 40
fewshot_list = [5,10,15,20]
best_record = []
trial_times = 10
gpu_id = 2

###choose from "tpe","rand","HRA","anneal"
optimize_algo = "anneal"

config_path = code_path+ "/AutoMMdet/Optical/Optimize_Cascade/Cascade_Configs/Cascade_ResNet_50.py"
current_time = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
save_dir = code_path + "/Work_dir/"+current_time+"/Optical_"+optimize_algo+"/"
data_root = code_path + "/Optical_Dataset/optical-img/test-A-image/"
load_path = code_path + "/checkpoints/optical/epoch_22_old.pth"

for shot in fewshot_list:
    fewshot_dir = save_dir + str(shot)+"_shot/"
    for trial in range(trial_times):
        anno_root = code_path + "Optical_Dataset/coco/" + str(shot) + "_shot/split-"+str(trial+1)
        current_time = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
        work_dir = fewshot_dir+"trial_"+str(trial+1)+"/"
        optical_mmdet = MMdet_HPO(config_path,work_dir,optimizer=optimize_algo)
        
        optical_mmdet.max_eval = max_eval
        optical_mmdet.gpu_id = gpu_id
        optical_mmdet.data_path = data_root
        optical_mmdet.anno_path = anno_root
        optical_mmdet.load_path = load_path
        optical_mmdet.max_epoch = 10

        best_config,best_valid_loss, best_loss = optical_mmdet.HPO()
        best_record.append([best_config,best_valid_loss,best_loss])
        best_record_pd = pd.DataFrame(best_record,columns=["best_config","best_valid_loss","best_loss"])
        save_path = work_dir+"/"+str(shot)+"-shot_"+optimize_algo+"_trial-"+str(trial+1)+"_best_record.csv"
        best_record_pd.to_csv(save_path)