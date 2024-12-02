# HM 2
by Yufeng Guo

## install

by installation, follow the steps:
```

mkdir build && cd build
cmake ../
make
cp ./gra* ../hw2/
cp ./pytorch_apis.py ./hw2/
```

## model architecture
3 layer fully connected network:

with linear layer was implemented through our own CUDA kernel funtion

## Model runing && Result


# HW3 
## Environment Setup:
NVCC:11.8, Nvidia A100
```
conda create -n dgl python=3.7
conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.2 -c pytorch
pip install dgl-cu102==0.6.1
```

By installation, follow steps in HW2

## Graph and Kernel Implementation
- Graph structures was decleared : [csr.h](https://github.com/misra0514/deep-codegen/blob/main/csr.h#L12) 
- spmm was implemented:[kernel.cu](https://github.com/misra0514/deep-codegen/blob/main/kernel.cu)
- backward part:[pytorch_apis.py](https://github.com/misra0514/deep-codegen/blob/main/pytorch_apis.py#L18)
- Model runnng && Testing:  [hw3/main.py]() && [hw3/model2.py]()

## Results
Model was trained in 
#### CORA
![image](hw3/result/cora.png)
#### citeseer
![image](hw3/result/citeseer.png)
#### pubmed
![image](hw3/result/pubmed.png)
#### reddit
![image](hw3/result/reddit.png)


# HW4 
## setup 
**<font color=#FF0000>NOTE: this work is done at a new branch:HW4_ddp</font>**

code download :
```
git clone -b hw4_ddp https://github.com/misra0514/deep-codegen.git 
```

## Result:


# HW5:

Torch 2.0 is required for graph operations

#### approach:
- add change the old interface gspmmv into ```void gspmmv(graph_t& graph, array2d_t<float>& input1, array2d_t<float>& output, bool reverse, bool norm, uintptr_t stream_handle);```, where ```stream_handle``` is the parameter for streaming function. 

- get stream number in ```gp_apis.py```, through ```th.cuda.current_stream().cuda_stream```. 

- change data type from ```uintptr_t``` into ```cudaStream_t``` in ```kernel.cu```. 

- add cuda graph in ```main.py```

#### result:

The following data was collected with 2000 epoches, cora datase: 

| Usage | Time(s) |
| -------- | -------- |
| CudaGraph | 4.32 |
| Normal | 5.67 |