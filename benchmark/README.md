# 地震波形拾取模型推理时间对比

## 纯GPU推理时间 (秒)

| 数据集 | 样本数 | EQTransformer | PhaseNet | MoE |
|--------|--------|:-----------:|:--------:|:---:|
| Txed | 1万 | 2.8 | 0.9 | 21.0 |
| DiTing | 1万 | 3.1 | 1.2 | 18.7 |
| Instance | 10万 | 36.6 | 13.4 | 197.5 |
| STEAD | 1万 | 2.5 | 0.7 | 2.9 |

> 模型: EQTransformer/PhaseNet(seisbench) + MoE(PS3 + best_model_CPU.pth)
> GPU: 8GB | 内存: 64GB | CPU: 24核
> Instance含resample+bandpass预处理，STEAD从SSD预加载到RAM
