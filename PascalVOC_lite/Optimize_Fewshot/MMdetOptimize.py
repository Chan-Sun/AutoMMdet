import time
import pandas as pd
import sys
code_path = "/home/hustwen/sun_chen/Optical_VOC/"
sys.path.append(code_path+"AutoMMdet")
from HPO.Select_HPO import AutoSelect

multi_metric = ["mAP_50","mAP","mAP_75","mAP_s","mAP_l","mAP_m"]
single_metric = ["mAP_50"]
max_eval = 40
trial_times = 10
fewshot_num = [5,10,15,20]
best_record = []
###choose from "tpe","rand","HRA","anneal"
optimize_algo = "tpe"
cfg_path = code_path+"/AutoMMdet/PascalVOC_lite/Optimize_Fewshot/FasterRCNN_Config/"

gpu_id = 0
data_root = code_path + "Dataset/PascalVOC_lite/Test_JPEGImages/"
load_path = code_path + "/checkpoints/pascalvoc_lite/checkpoints/"

current_time = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
save_dir = code_path + "/Work_dir/"+current_time+"/single_fewshot_voc_"+optimize_algo               
for shot in fewshot_num:
    fewshot_dir = save_dir + str(shot)+"_shot/"
    for trial in range(trial_times):
        anno_root = code_path + "/Dataset/PascalVOC_lite/coco_test/coco_test_"+str(shot)+"/split-"+str(trial+1)
        current_time = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
        work_dir = fewshot_dir+"trial_"+str(trial+1)+"/"
        VOC_select = AutoSelect(cfg_path,single_metric,work_dir,optimizer=optimize_algo)
        
        VOC_select.load_path = load_path
        VOC_select.gpu_id = gpu_id
        VOC_select.max_eval = max_eval
        VOC_select.data_root = data_root
        VOC_select.anno_root = anno_root

        best_test_loss,backbone,color_method,pixel_method = VOC_select.hpo_select()
        best_record=[best_test_loss,backbone,color_method,pixel_method]
        best_record_pd = pd.DataFrame(best_record,columns=["loss","backbone","color_method","pixel_method"])
        save_path = work_dir+"/voc_single-metric_"+str(shot)+"-shot_"+optimize_algo+"_trial-"+str(trial+1)+"_best-record.csv"
        best_record_pd.to_csv(save_path)


# current_time = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
# save_dir = code_path + "/Work_dir/"+current_time+"/multi_fewshot_voc_"+optimize_algo        
# for shot in fewshot_num:
#     fewshot_dir = save_dir + str(shot)+"_shot/"
#     for trial in range(trial_times):
#         anno_root = code_path + "/Dataset/PascalVOC_lite/coco_test/coco_test_"+str(shot)+"/split-"+str(trial+1)
#         current_time = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
#         work_dir = fewshot_dir+"trial_"+str(trial+1)+"/"                
#         VOC_select = AutoSelect(cfg_path,multi_metric,work_dir,optimizer=optimize_algo)
#         VOC_select.load_path = load_path
#         VOC_select.gpu_id = gpu_id
#         VOC_select.max_eval = max_eval
#         VOC_select.data_root = data_root
#         VOC_select.anno_root = anno_root
#         best_test_loss,backbone,color_method,pixel_method = VOC_select.hpo_select()
#         best_record=[best_test_loss,backbone,color_method,pixel_method]
#         best_record_pd = pd.DataFrame(best_record,columns=["loss","backbone","color_method","pixel_method"])
#         save_path = work_dir+"/voc_multi-metric_"+str(shot)+"-shot_"+optimize_algo+"_trial-"+str(trial+1)+"_best-record.csv"
#         best_record_pd.to_csv(save_path)