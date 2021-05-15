from HPO.hyperopt_yxc import hp

MMdet_SearchSpace = {
                "model.train_cfg.rpn_proposal.nms.iou_threshold":hp.uniform("model.train_cfg.rpn_proposal.nms.iou_threshold",0.5,0.8),
                "model.rpn_head.anchor_generator.scales":hp.quniform("model.rpn_head.anchor_generator.scales",4,9,1),
                'optimizer':hp.choice('optimizer', [                   
                    {"type": "RMSprop","lr":hp.uniform("lr_1",0.00125,0.004),
                                "momentum":0.8,
                                "weight_decay":hp.uniform("weight_decay_1",0.0001,0.001)
                        },              
                    {"type": "Adagrad","lr":hp.uniform("lr_2",0.00125,0.004),
                                "weight_decay":hp.uniform("weight_decay_2",0.0001,0.001)
                        },              
                    {"type": "Adam","lr":hp.uniform("lr_3",0.00125,0.004),
                                "weight_decay":hp.uniform("weight_decay_3",0.0001,0.001)
                        },            
                    {"type": "Adadelta","lr":hp.uniform("lr_4",0.00125,0.004),
                                "rho":hp.uniform("rho",0.8,0.99),
                                "weight_decay":hp.uniform("weight_decay_4",0.0001,0.001)
                        },  
                    {"type": "SGD","lr":hp.uniform("lr_5",0.00125,0.004),
                                "momentum":0.8,
                                "weight_decay":hp.uniform("weight_decay_5",0.0001,0.001)
                        },                                        
                    ]),
                "lr_config":hp.choice("lr_config",[
                    {"policy":"poly","power":hp.uniform("power_1",0.5,2),
                                "warmup":hp.choice("warmup_1",["constant","linear","exp"]),
                                "warmup_ratio":hp.uniform("warmup_ratio_1",0.001,0.005),
                                "warmup_iters":500,
                        },
                    {"policy":"CosineAnnealing","min_lr_ratio":1e-5,
                                "warmup":hp.choice("warmup_2",["constant","linear","exp"]),
                                "warmup_ratio":hp.uniform("warmup_ratio_2",0.001,0.005),
                                "warmup_iters":500,
                        },                    
                    {"policy":"step","step":1,
                                "warmup":hp.choice("warmup_3",["constant","linear","exp"]),
                                "warmup_ratio":hp.uniform("warmup_ratio_3",0.001,0.005),
                                "warmup_iters":500,
                        }
                    ])
                }

GaussNoise = dict(type="GaussNoise",var_limit=(10.0, 50.0), mean=0, per_channel=True, always_apply=False, p=0.5)
Blur = dict(type = "Blur",blur_limit=7, always_apply=False, p=0.5)
RGBshift = dict(type = "RGBShift",r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, always_apply=False, p=0.5)
JpegCompression = dict(type='JpegCompression', quality_lower=85, quality_upper=95, p=0.2)
HorizontalFlip = dict(type='HorizontalFlip',p=0.5)
VerticalFlip = dict(type='VerticalFlip',p=0.5)
ShiftScaleRotate = dict(type='ShiftScaleRotate',shift_limit=0.0625,scale_limit=0.0,rotate_limit=30,interpolation=2,p=0.5)
RandomBrightnessContrast = dict(type='RandomBrightnessContrast',brightness_limit=[0.1, 0.3],contrast_limit=[0.1, 0.3],p=0.2)

AutoSelect_SearchSpace = {
    "backbone":hp.choice("backbone",["r50","r101","x32","x64","hrnet"]),
    "color_method":hp.choice("color_method",[GaussNoise,Blur,RGBshift,RandomBrightnessContrast]),
    "pixel_method":hp.choice("pixel_method",[JpegCompression,VerticalFlip,HorizontalFlip,ShiftScaleRotate])
}    
    