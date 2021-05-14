import time
from HPO.MMdetProcess import MMdet_Pipeline


cfg_path = "./Cacade_ResNet_50.py"
Dataset_path = 'None'
Save_path = "None"
checkpoint_path = "None"
fewshot_list = [5]#[5,10,15,20]
cfg_path = "./faster_rcnn_r50.py"

for fewshot in fewshot_list:
    for trial in range(10):
        anno_root = Dataset_path + "../../coco_test/coco_test_"+str(fewshot)+"/split-"+str(trial+1)
        current_time = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
        work_dir =Save_path + "../../work_dir/"+str(current_time)+"/default_train/"        

        gpu_id = 0

        valid = False
        MMdetpipe = MMdet_Pipeline(cfg_path,work_dir,anno_root,gpu_id,valid)
        MMdetpipe.cfg.checkpoint_config = dict(interval=2)
        MMdetpipe.cfg.runner = dict(type='EpochBasedRunner', max_epochs=10)
        MMdetpipe.cfg.data.test.ann_file = anno_root+"/instances_val.json"
        MMdetpipe.cfg.load_from = checkpoint_path+'../../fewshot_checkpoints/r50.pth'
        MMdetpipe.MMdet_train(MMdetpipe.cfg)
        
        ckp_path = work_dir+"/latest.pth"
        result = MMdetpipe.MMdet_Valid(ckp_path,MMdetpipe.cfg,False,False,None)
