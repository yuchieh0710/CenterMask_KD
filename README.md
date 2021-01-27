# [CenterMask KD](https://arxiv.org/abs/1911.06667) : Apply Knowledge Distillation on CenterMask (Instance Segmentation)


![architecture](architecture.png)

## Abstract

Deep learning has become one of the emerging research fields. In deep learning, many applications are expected to apply on edge computing devices or mobile devices. However, due to the intrinsic nature of deep neural network, numerous parameters and calculations make it difficult for real-time performance on edge devices. Therefore, we need to compress the neural network, and simultaneously maintain the neural network at a certain performance. Among many model compression methods, we chose to use knowledge distillation as our method. It is applied to the task of instance segmentation to improve the accuracy of lightweight models with relatively fewer parameters than its original one. In the experiment, we can find that our method is very effective for improving the performance of the model and thus reveal the feasibility for real-time applications.

### Environment
- V100 or Titan Xp GPU
- CUDA 9.0 
- cuDNN 7.5.1 
- pytorch 1.1.0 [wheel file](https://download.pytorch.org/whl/cu90/torch-1.1.0-cp35-cp35m-linux_x86_64.whl)
- torchvision 0.3.0 [wheel file](https://download.pytorch.org/whl/cu90/torchvision-0.3.0-cp35-cp35m-manylinux1_x86_64.whl)
- Implemented on [CenterMask](https://github.com/youngwanLEE/CenterMask)  

## Installation
Check [INSTALL.md](INSTALL.md) for installation instructions which is orginate from [CenterMask](https://github.com/youngwanLEE/CenterMask).

## Training
Modify basic training setting in [train_net.py](tools/train_net.py) 'main()'

```bash
python tools/train_net.py  
```

## Evaluation

After training, you can evaluate your model in [test_net.py](tools/test_net.py) and visualize in [centermask_demo.py](demo/centermask_demo.py).
Change trained weight path in config you choose. 

##### For single-gpu evaluation 
```bash
python tools/test_net.py --config-file "configs/centermask/centermask_V_19_eSE_FPN_lite_res600_ms_bs16_4x.yaml" 
The results of test-dev can upload json file on [box result](https://competitions.codalab.org/competitions/20794#participate-get-data) & [segm result](https://competitions.codalab.org/competitions/20796#participate-get-data)
```

##### For single-gpu visualize 
```bash
python demo/centermask_demo.py --config-file "configs/centermask/centermask_V_19_eSE_FPN_lite_res600_ms_bs16_4x.yaml"  --weights "tools/checkpoints/student/model_0360000.pth"  --input "demo/test_image"  --output_dir "demo/results/test_result"
The visualize result will under 'demo/demo/results/'
```
