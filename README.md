---
hardwares:
  - NPU
  - Atlas 800T A2
  - Atlas 800I A2
pipeline_tag: multi-modal
license: apache-2.0
frameworks:
  - PyTorch
---
# HunyuanWorld-Voyager推理指导
## 一、准备运行环境

  **表 1**  版本配套表

  | 配套  | 版本 | 环境准备指导 |
  | ----- | ----- |-----|
  | Python | 3.11.10 | - |
  | torch | 2.1.0 | - |

### 1.1 获取CANN&MindIE安装包&环境准备
- 设备支持
Atlas 800I/800T A2(8*64G)推理设备：支持的卡数最小为1
[Atlas 800I/800T A2(8*64G)](https://www.hiascend.com/developer/download/community/result?module=pt+ie+cann&product=4&model=32)
- [环境准备指导](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC2alpha002/softwareinst/instg/instg_0001.html)

### 1.2 CANN安装
```shell
# 增加软件包可执行权限，{version}表示软件版本号，{arch}表示CPU架构，{soc}表示昇腾AI处理器的版本。
chmod +x ./Ascend-cann-toolkit_{version}_linux-{arch}.run
chmod +x ./Ascend-cann-kernels-{soc}_{version}_linux.run
chmod +x ./Ascend-cann-nnal_{version}_linux-{arch}.run  (若使用稀疏FA)
# 校验软件包安装文件的一致性和完整性
./Ascend-cann-toolkit_{version}_linux-{arch}.run --check
./Ascend-cann-kernels-{soc}_{version}_linux.run --check
./Ascend-cann-nnal{version}_linux-{arch}.run --check  (若使用稀疏FA)
# 安装
./Ascend-cann-toolkit_{version}_linux-{arch}.run --install
./Ascend-cann-kernels-{soc}_{version}_linux.run --install
./Ascend-cann-nnal{version}_linux-{arch}.run --torch_atb --install  (若使用稀疏FA)

# 设置环境变量
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
```

### 1.3 环境依赖安装
```shell
pip3 install -r requirements.txt
```

### 1.4 MindIE安装
```shell
# 增加软件包可执行权限，{version}表示软件版本号，{arch}表示CPU架构。
chmod +x ./Ascend-mindie_${version}_linux-${arch}.run
./Ascend-mindie_${version}_linux-${arch}.run --check

# 方式一：默认路径安装
./Ascend-mindie_${version}_linux-${arch}.run --install
# 设置环境变量
cd /usr/local/Ascend/mindie && source set_env.sh

# 方式二：指定路径安装
./Ascend-mindie_${version}_linux-${arch}.run --install-path=${AieInstallPath}
# 设置环境变量
cd ${AieInstallPath}/mindie && source set_env.sh
```

### 1.5 Torch_npu安装
下载 pytorch_v{pytorchversion}_py{pythonversion}.tar.gz
```shell
tar -xzvf pytorch_v{pytorchversion}_py{pythonversion}.tar.gz
# 解压后，会有whl包
pip install torch_npu-{pytorchversion}.xxxx.{arch}.whl
```

### 1.6 gcc、g++安装
```shell
# 若环境镜像中没有gcc、g++，请用户自行安装
yum install gcc
yum install g++

# 导入头文件路径
export CPLUS_INCLUDE_PATH=/usr/include/c++/12/:/usr/include/c++/12/aarch64-openEuler-linux/:$CPLUS_INCLUDE_PATH
```
注：若使用openeuler镜像，需要配置gcc、g++环境，否则会导致`fatal error: 'stdio.h' file not found`

## 二、下载权重

### 2.1 HunyuanWorld-Voyager权重及配置文件说明

-  Huggingface

| 模型                   | 链接  |
|----------------------| ------------ |
| HunyuanWorld-Voyager |  [huggingface](https://huggingface.co/tencent/HunyuanWorld-Voyager/tree/main) |


- Modelscope

|  模型 | 链接                                                                     |
| ------------ |------------------------------------------------------------------------|
| HunyuanWorld-Voyager  | [Modelscope](https://www.modelscope.cn/models/Tencent-Hunyuan/HunyuanWorld-Voyager) |

## 三、HunyuanWorld-Voyager使用

### 3.1 下载到本地
```shell
git clone https://modelers.cn/MindIE/HunyuanWorld-Voyager.git
```

### 3.2 HunyuanWorld-Voyager
将上一步下载的权重放在ckpts路径下
```shell
ln -s /XX/HunyuanWorld-Voyager/* ckpts/
```
#### 3.2.1 单卡
优化点：LA

执行命令：run.sh
```shell
export PYTORCH_NPU_ALLOC_CONF='expandable_segments:True'
export TASK_QUEUE_ENABLE=2
export CPU_AFFINITY_CONF=1
export TOKENIZERS_PARALLELISM=false
export ALGO=1

python3 sample_image2video.py \
    --model HYVideo-T/2 \
    --input-path "examples/case1" \
    --prompt "An old-fashioned European village with thatched roofs on the houses." \
    --i2v-stability \
    --infer-steps 50 \
    --flow-reverse \
    --flow-shift 7.0 \
    --seed 0 \
    --embedded-cfg-scale 6.0 \
    --save-path ./results
```
参数说明：
- ALGO: 为0表示默认FA算子; 设置为1表示使用高性能FA算子

#### 3.2.2 多卡性能测试
优化点：LA、Attention cache、USP+Ring、VAE并行等

执行命令：run_mul.sh
```shell
export PYTORCH_NPU_ALLOC_CONF='expandable_segments:True'
export TASK_QUEUE_ENABLE=2
export CPU_AFFINITY_CONF=1
export TOKENIZERS_PARALLELISM=false
export ALGO=1

torchrun --master_port=2031 --nproc_per_node=8 \
    sample_image2video.py \
    --model HYVideo-T/2 \
    --input-path "examples/case1" \
    --prompt "An old-fashioned European village with thatched roofs on the houses." \
    --i2v-stability \
    --infer-steps 50 \
    --flow-reverse \
    --flow-shift 7.0 \
    --seed 0 \
    --embedded-cfg-scale 6.0 \
    --save-path ./results \
    --ulysses-degree 8 \
    --ring-degree 1 \
    --use_attentioncache \
    --vae-parallel
```
参数说明：
- ALGO: 为0表示默认FA算子; 设置为1表示使用高性能LA算子
- ulysses-degree：USP并行数
- ring-degree：ring并行数
- use_attentioncache：是否开启Attention cache
- vae-parallel：是否开启VAE并行
- use-encoder-load-cpu：默认不开启，OOM时可选择开启
- use_cache：是否开启DIT single block cahce
- use_cache_double：是否开启DIT double block cahce

#### 3.2.2 多机性能测试
优化点：LA、Attention cache、USP+Ring、VAE并行等

执行命令：run_mul_node1.sh、run_mul_node2.sh

## 四、推理结果参考
###  A2 A+K

|          模型          |   分辨率   | 帧数 | 迭代次数 | 卡数 | E2E耗时  |
|:--------------------:|:-------:|:--:|:----:| :-----: |:------:|
| HunyuanWorld-Voyager | 512×768 | 49 |  50  | 8 | 116.5s |

## 五、声明
- 本代码仓提到的数据集和模型仅作为示例，这些数据集和模型仅供您用于非商业目的，如您使用这些数据集和模型来完成示例，请您特别注意应遵守对应数据集和模型的License，如您因使用数据集或模型而产生侵权纠纷，华为不承担任何责任。
- 如您在使用本代码仓的过程中，发现任何问题（包括但不限于功能问题、合规问题），请在本代码仓提交issue，我们将及时审视并解答。

## 六、常见问题
1. 当前支持attentioncache，但是单卡和2卡时可能会OOM，增加的encoder offload cpu会导致耗时增加，建议在OOM开启use-encoder-load-cpu
2. 若遇到报错: `Directory operation failed. Reason: Directory [/usr/local/Ascend/mindie/latest/mindie-rt/aoe] does not exist`,请设置环境变量`unset TUNE_BANK_PATH`
3. 若使用openeuler镜像, 若没有配置gcc、g++环境，会遇到报错：`fatal error: 'stdio.h' file not found`，请参考`1.6 gcc、g++安装`
4. 若循环跑纯模型推理，可能会因为HCCL端口未及时释放，导致因端口被占用而推理失败，报错：`Failed to bind the IP port. Reason: The IP address and port have been bound already.`
  `HCCL function error :HcclGetRootInfo(&hcclID), error code is 7`:  请配置`export HCCL_HOST_SOCKET_PORT_RANGE="auto"`不指定端口
  `HCCL function error :HcclGetRootInfo(&hcclID), error code is 11`: 请配置`sysctl -w net.ipv4.ip_local_reserved_ports=60000-60015`预留端口
5. 当前版本A3设备上暂不支持ALGO=1，如需在A3上使用ALGO=1请联系模型owner

