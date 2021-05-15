import time
import numpy as np
import torch
from HPO.hyperopt_yxc import tpe,Trials,rand,anneal
from HPO.AutoHPO_V7V5 import fmin_hyperp_reduce,fmin_raw
import time 
from HPO.SearchSpace import AutoSelect_SearchSpace,MMdet_SearchSpace
from HPO.MMdetProcess import MMdet_Pipeline

class AutoSelect():
    def __init__(self,cfg_path,metric_list,work_dir,optimizer=None):

        self.searchspace = AutoSelect_SearchSpace
        self.gpu_id = 1
        self.valid = False
        self.work_dir = work_dir
        self.cfg_path = cfg_path
        self.max_eval = 40
        self.metric_list = ["bbox_"+metric for metric in metric_list]
        self.data_root = None
        self.anno_root = None
        self.load_path = None
        self.algo = optimizer
        self.max_epoch = 10
    
    def objective(self,param):
        cfg_path = self.cfg_path+"/faster_rcnn_"+param["backbone"]+".py"

        MMdetpipe = MMdet_Pipeline(cfg_path,self.work_dir,self.data_root,self.anno_root,self.gpu_id,self.valid)
        albu_train_transforms =[param["pixel_method"],param["color_method"]]
        albu_preprocess =  dict(type='Albu',transforms=albu_train_transforms,
                                bbox_params=dict(type='BboxParams',format='pascal_voc',
                                                 label_fields=['gt_labels'],min_visibility=0.0,
                                                 filter_lost_elements=True),
                                keymap={'img': 'image','gt_bboxes': 'bboxes'},
                            update_pad_shape=False,
                            skip_img_without_anno=True)
        
        MMdetpipe.cfg.load_path = self.load_path + "/"+param["backbone"]+".pth"
        MMdetpipe.cfg.train_pipeline.insert(5,albu_preprocess)
        MMdetpipe.cfg.checkpoint_config = dict(interval=2)
        MMdetpipe.cfg.runner = dict(type='EpochBasedRunner', max_epochs=self.max_epoch)
        MMdetpipe.cfg.data.test.ann_file = self.anno_root+"/instances_val.json"

        try:
            MMdetpipe.MMdet_train(MMdetpipe.cfg)
            ckp_path = self.work_dir+"/latest.pth"
            result = MMdetpipe.MMdet_Valid(ckp_path,MMdetpipe.cfg,False,False,None)
        except:
            torch.cuda.empty_cache()
            return 1
        if len(result.keys()) == 0:
            print("Testing process is overfitting")
            return 1
        else:
            multi_result = np.mean([result[metric] for metric in self.metric_list])
            print(multi_result)
            return 1-multi_result

    def hpo_select(self):
        start_time = time.time()
        if self.algo == "HRA":
            best_config,results,_ = fmin_hyperp_reduce(fn=self.objective, space=self.searchspace, algo=tpe.suggest, trials=Trials(), max_evals=self.max_eval)
        else:
            optimizer_dict = {"tpe":tpe,"rand":rand,"anneal":anneal}
            optimize_algo = optimizer_dict[self.algo]
            best_config,results,space_save = fmin_raw(fn=self.objective,space=self.searchspace,trials=Trials(),algo=optimize_algo.suggest,max_evals=self.max_eval)
        end_time = time.time()
        print("##################################################################")
        print("##################################################################")
        print("This Auto Select process consumes %s seconds in total"%(end_time-start_time))
        print("##################################################################")
        print("##################################################################")
        results["time"] = end_time-start_time
        results.to_csv(self.work_dir+"/auto_select.csv")
        print(best_config)
        best_test_loss,backbone,color_method,pixel_method=self.BestConfig(best_config)
        return best_test_loss,backbone,color_method,pixel_method
    
    def BestConfig(self,best):
        work_dir = self.work_dir+"/best_config"
        best_cfg_path = "./faster_rcnn_"+best["backbone"]+".py"
        MMdetBest = MMdet_Pipeline(best_cfg_path,work_dir,self.data_root,self.anno_root,self.gpu_id,True) 

        albu_train_transforms =[best["pixel_method"],best["color_method"]]
        albu_preprocess =  dict(type='Albu',transforms=albu_train_transforms,
                                bbox_params=dict(type='BboxParams',format='pascal_voc',
                                                 label_fields=['gt_labels'],min_visibility=0.0,
                                                 filter_lost_elements=True),
                                keymap={'img': 'image','gt_bboxes': 'bboxes'},
                            update_pad_shape=False,
                            skip_img_without_anno=True)

        MMdetBest.cfg.load_path = self.load_path + "/"+best["backbone"]+".pth"    
        MMdetBest.cfg.train_pipeline.insert(5,albu_preprocess)
        MMdetBest.cfg.checkpoint_config = dict(interval=2)
        MMdetBest.cfg.runner = dict(type='EpochBasedRunner', max_epochs=self.max_epoch)
        MMdetBest.cfg.data.test.ann_file = self.anno_root+"/instances_test.json"

        try:
            MMdetBest.MMdet_train(MMdetBest.cfg)
            ckp_path = work_dir+"/latest.pth"
            result = MMdetBest.MMdet_Valid(ckp_path, MMdetBest.cfg, False, False, None)
        except:
            torch.cuda.empty_cache()
            result = {"bbox_mAP_50":0}

        if len(result.keys()) == 0:
            print("Testing process is overfitting")
            return 1
        else:
            test_result = result["bbox_mAP_50"]
            print(best)
            print(test_result)
            return test_result,best["backbone"],best["color_method"]["type"],best["pixel_method"]["type"]

class MMdet_HPO():
    def __init__(self,cfg_path,work_dir,optimizer=None):
        self.work_dir = work_dir
        self.cfg_path = cfg_path
        self.search_space = MMdet_SearchSpace
        self.max_eval = 40
        self.data_path = None
        self.anno_path = None
        self.gpu_id = 0
        self.valid = False
        self.algo = optimizer
        self.load_path = None
        self.max_epoch = 2
    def objective(self,config):
        
        scales = [int(config['model.rpn_head.anchor_generator.scales'])]
        del config['model.rpn_head.anchor_generator.scales']
        config['model.rpn_head.anchor_generator.scales'] = scales
        work_dir = self.work_dir+"/MMdet_HPO"
        MMdetpipe = MMdet_Pipeline(self.cfg_path,work_dir,self.data_path,self.anno_path,self.gpu_id,self.valid)
        MMdetpipe.cfg.merge_from_dict(config)

        MMdetpipe.cfg.checkpoint_config = dict(interval=2)
        MMdetpipe.cfg.runner = dict(type='EpochBasedRunner', max_epochs=self.max_epoch)
        MMdetpipe.cfg.data.test.ann_file = self.anno_path+"/instances_val.json"
        MMdetpipe.cfg.load_from = self.load_path

        try:
            MMdetpipe.MMdet_train(MMdetpipe.cfg)            
            ckp_pth = work_dir+"/latest.pth"
            result = MMdetpipe.MMdet_Valid(ckp_pth,MMdetpipe.cfg,False,False,None)
        except:
            torch.cuda.empty_cache()
            return 1
        
        if len(result.keys()) == 0:
            print("Testing process is overfitting")
            return 1
        else:
            return 1-result["bbox_mAP_50"]

    def HPO(self):
        start_time = time.time()
        if self.algo == "HRA":
            best_config,results,space_save = fmin_hyperp_reduce(fn=self.objective, space=self.search_space, algo=tpe.suggest, trials=Trials(), max_evals=self.max_eval)
        else:
            optimizer_dict = {"tpe":tpe,"rand":rand,"anneal":anneal}
            optimize_algo = optimizer_dict[self.algo]
            best_config,results,space_save = fmin_raw(fn=self.objective,space=self.search_space,trials=Trials(),algo=optimize_algo.suggest,max_evals=self.max_eval)
        end_time = time.time()
        print("##################################################################")
        print("##################################################################")
        print("This HPO process consumes %s seconds in total"%(end_time-start_time))
        print("##################################################################")
        print("##################################################################")
        results["time"] = end_time-start_time
        results.to_csv(self.work_dir+"/raw_results.csv")
        results_reorganized = self.Results_to_csv(results)
        space_save.to_csv(self.work_dir+"/space.csv")
        test_loss = self.BestConfig(best_config)
        valid_loss = results_reorganized.sort_values(by="loss").iloc[0,0]
        return best_config,valid_loss,test_loss
    
    def BestConfig(self,best):
        scales = [int(best['model.rpn_head.anchor_generator.scales'])]
        del best['model.rpn_head.anchor_generator.scales']
        best['model.rpn_head.anchor_generator.scales'] = scales 
        work_dir = self.work_dir+"/best_config"
        MMdetBest = MMdet_Pipeline(self.cfg_path,work_dir,self.data_path,self.anno_path,self.gpu_id,False)
        MMdetBest.cfg.merge_from_dict(best)

        MMdetBest.cfg.checkpoint_config = dict(interval=2)
        MMdetBest.cfg.runner = dict(type='EpochBasedRunner', max_epochs=self.max_epoch)
        MMdetBest.cfg.data.test.ann_file = self.anno_path+"/instances_test.json"  
        MMdetBest.cfg.load_from = self.load_path     

        try:
            MMdetBest.MMdet_train(MMdetBest.cfg)     
            ckp_path = work_dir+"/latest.pth"
            result = MMdetBest.MMdet_Valid(ckp_path, MMdetBest.cfg, False, False, None)
            return 1-result["bbox_mAP_50"]
        except:
            torch.cuda.empty_cache()
            print("cuda failed")      
            return 1

    def CombineColumn(self,keyword,banword,results):
        results[keyword]=0
        combineList = [column for column in results.columns.values if keyword+"_" in column and banword not in column]
        combineDataFrame = results[combineList]
        nan_check = combineDataFrame.count(axis='columns')
        combineDataFrame = combineDataFrame.replace(np.nan, 0)
        for column in combineList:
            results[keyword] = results[keyword] +combineDataFrame[column]
            del results[column]
        for index in range(len(nan_check)):
            if nan_check[index]==0:
                results[keyword][index] =  results[keyword][index-1]
    
    def Results_to_csv(self,results):
        self.CombineColumn("lr","lr_config",results)
        self.CombineColumn("warmup","warmup_ratio",results)
        self.CombineColumn("warmup_ratio","lr_config",results)
        self.CombineColumn("weight_decay","lr_config",results)
        print(results)
        if "Unnamed: 0" in results.columns.values:
            del results["Unnamed: 0"]
        if "power_1" in results.columns.values:
            del results["power_1"]
        results.to_csv(self.work_dir+'/results.csv')
        return results
