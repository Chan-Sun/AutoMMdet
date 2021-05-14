# Optical_Undersonar
A project using mmdetection to detect voc dataset and optical undersonar dataset

Some automl technique was introduced in this project to get a better output

**environment**: pytorch-1.7,MMCV_1.3.3,MMDetection_2.12

This project are divided into two automl and object detection tasks:

### PascalVOC_lite:
***dataset***: five animal classes in pascalvoc_2007, including dog, cat, cow, sheep and horse

***preprocess method***: Flip/RGBShift/…………(8 methods in all,divided into pixel_level and rgb_level group)

***model***: faster rcnn with different backbones(resnet-50,resnet-101,hrnet,resnext101_32,resnext101_64)

***automl***: use bayesian optimize algorithm (TPE mainly) to conduct combination optimization(preprocess method+backbone)

### Optical:
***dataset***: five optical undersonar classes

***model***: cascade rcnn with resnet_50

***hyperparameter searchspace***: aimed to optimize anchor_scales, optimizer, learning_rate_config .etc hyperparameter in object detection network

***automl***: use bayesian optimize algorithm (TPE mainly) to conduct hpo. Based on TPE, design some hyperparameter reduction algorithm to get a smaller searchspace after a few round of optimzie
