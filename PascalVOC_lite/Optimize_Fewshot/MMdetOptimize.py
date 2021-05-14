from HPO.MMdetProcess import AutoSelect
from hyperopt import tpe,rand,anneal
import time
import pandas as pd

multi_metric = ["mAP_50","mAP","mAP_75","mAP_s","mAP_l","mAP_m"]
single_metric = ["mAP_50"]
max_eval = 40
trial_times = 10
fewshot_num = [5,10,15,20]
best_record = []

###choose from "tpe","rand","HRA","anneal"
optimize_algo = "tpe"

code_path = "/home/hustwen/sun_chen/Optical_VOC/"
cfg_path = code_path+"/AutoMMdet/PascalVOC_lite/Optimize_Fewshot/FasterRCNN_Config/faster_rcnn_r50.py"

gpu_id = 0
valid = False

for shot in fewshot_num:
    for trial in range(trial_times):
        anno_root = code_path + "/Dataset/PascalVOC_lite/coco_test/coco_test_"+str(shot)+"/split-"+str(trial+1)
        current_time = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
        work_dir = code_path + "/Work_dir/"+current_time+"/single_fewshot_voc_"+optimize_algo                
        VOC_select = AutoSelect(cfg_path,single_metric, work_dir, max_eval, anno_root,gpu_id,valid,optimizer=optimize_algo)
        best_test_loss,backbone,color_method,pixel_method = VOC_select.hpo_select()
        best_record=[best_test_loss,backbone,color_method,pixel_method]
        best_record_pd = pd.DataFrame(best_record)
        save_path = work_dir+"/voc_single-metric_"+str(shot)+"-shot_"+optimize_algo+"_trial-"+str(trial+1)+"_best-record.csv"
        best_record_pd.to_csv(save_path)

for shot in fewshot_num:
    for trial in range(trial_times):
        anno_root = code_path + "/Dataset/PascalVOC_lite/coco_test/coco_test_"+str(shot)+"/split-"+str(trial+1)
        current_time = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
        work_dir = Save_path + "/Work_dir/"+current_time+"/multi_fewshot_voc_"+optimize_algo                
        VOC_select = AutoSelect(cfg_path,multi_metric, work_dir, max_eval, anno_root,gpu_id,valid,optimizer=optimize_algo)
        best_test_loss,backbone,color_method,pixel_method = VOC_select.hpo_select()
        best_record=[best_test_loss,backbone,color_method,pixel_method]
        best_record_pd = pd.DataFrame(best_record)
        save_path = work_dir+"/voc_multi-metric_"+str(shot)+"-shot_"+optimize_algo+"_trial-"+str(trial+1)+"_best-record.csv"
        best_record_pd.to_csv(save_path)