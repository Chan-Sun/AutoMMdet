from hyperopt import tpe,rand,anneal
import pandas as pd
import time
config_path = "/home/hustwen/sun_chen/Rapid_Support/Dataset/Optical/configs/Cascade_ResNet_50.py"
max_eval = 40
fewshot_list = [5,10,15,20]
best_record = []
trial_times = 10

Dataset_path = 'None'
Save_path = "None"

for shot in fewshot_list:
    shot = str(shot)
    gpu_id = 0
    valid = False
    optimize_algo = tpe
    for trial in range(trial_times):
        anno_root = Dataset_path+"/coco_test/coco_test_"+str(shot)+"/split-"+str(trial+1)
        current_time = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
        work_dir = Save_path + "/work_dir/"+current_time+"/single_fewshot/"        
        optical_mmdet = MMdet_HPO(config_path,work_dir,max_eval,anno_root,gpu_id,valid,optimize_algo=tpe)
        best_config, best_loss = optical_mmdet.HPO()
        best_record=[best_config, best_loss]
        best_record = pd.DataFrame(best_record)
        save_path = work_dir+"/best_record.csv"
        best_record.to_csv(save_path)