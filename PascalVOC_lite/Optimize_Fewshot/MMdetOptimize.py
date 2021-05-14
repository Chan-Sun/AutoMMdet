from MMdetProcess import AutoSelect
from hyperopt import tpe,rand,anneal
import time
import pandas as pd

multi_metric = ["mAP_50","mAP","mAP_75","mAP_s","mAP_l","mAP_m"]
single_metric = ["mAP_50"]
max_eval = 40
trial_times = 10
single_fewshot_num = [5,10,15,20]
multi_fewshot_num = [5,10,15,20]
best_record = []

for shot in single_fewshot_num:
    for trial in range(trial_times):
        anno_root = "../../coco_test/coco_test_"+str(shot)+"/split-"+str(trial+1)
        current_time = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
        work_dir = "../../work_dir/"+current_time+"/single_fewshot/"        
        gpu_id = 0
        valid = False
        VOC_select = AutoSelect(single_metric, work_dir, max_eval, anno_root,gpu_id,valid)
        best_test_loss,backbone,color_method,pixel_method = VOC_select.hpo_select()
        best_record=[best_test_loss,backbone,color_method,pixel_method]
        best_record = pd.DataFrame(best_record)
        save_path = work_dir+"/best_record.csv"
        best_record.to_csv(save_path)

for shot in multi_fewshot_num:
    for trial in range(trial_times):
        anno_root = "../../coco_test/coco_test_"+str(shot)+"/split-"+str(trial+1)
        current_time = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
        work_dir = "../../work_dir/"+current_time+"/multi_fewshot/"        
        gpu_id = 0
        valid = False
        VOC_select = AutoSelect(multi_metric, work_dir, max_eval, anno_root,gpu_id,valid)
        best_test_loss,backbone,color_method,pixel_method = VOC_select.hpo_select()
        best_record=[best_test_loss,backbone,color_method,pixel_method]
        best_record = pd.DataFrame(best_record)
        save_path = work_dir+"/best_record.csv"
        best_record.to_csv(save_path)