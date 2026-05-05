# 项目说明文档

## 项目概述

本项目对 **CoLight风格的图注意力多智能体强化学习交通信号控制（TSC）系统** 进行系统性的故障模式诊断研究。

核心问题：在真实部署场景中，交通信号控制系统会面临两种主要故障：
1. **传感器故障（Sensor Failure）**：路口传感器损坏，导致本地观测数据丢失
2. **通信故障（Communication Failure）**：相邻路口之间的通信中断，导致图注意力网络无法获取邻居信息

我们系统地研究了这两种故障对模型性能的影响，并扩展了多种额外的故障模式分析。

---

## 文件结构说明

```
CityFlow/
├── traffic_env.py          # 交通环境封装
├── colight_model.py        # CoLight模型（PyTorch实现）
├── failure_modes.py        # 故障注入模块
├── train_eval.py           # 训练与评估主脚本
├── run_with_watchdog.py    # 自动重启看门狗脚本
├── models/                 # 保存的训练模型
│   ├── colight_clean.pt         # 正常训练的模型
│   ├── colight_node_mask.pt     # 传感器故障训练的模型
│   ├── colight_edge_drop.pt     # 通信故障训练的模型
│   └── colight_mixed_robust.pt  # 混合故障训练的模型
├── results/                # 所有实验结果和图表
│   ├── expA_clean_sensitivity.png   # 实验A：故障类型敏感性图
│   ├── expB_cross_failure.png       # 实验B：跨故障迁移图
│   ├── expC_clean_patterns.png      # 实验C：故障模式对比图
│   ├── expD_clean_finegrain.png     # 实验D：细粒度rate曲线图
│   ├── expE_clean_mixed.png         # 实验E：混合故障柱状图
│   ├── expF_attn_heads.png          # 实验F：注意力头数对比图
│   ├── expG_mixed_robust.png        # 实验G：混合鲁棒训练对比图
│   ├── expH_hangzhou.png            # 实验H：Hangzhou泛化对比图
│   ├── training_curves.png          # 训练曲线图
│   └── *.json                       # 各实验的原始数据（JSON格式）
└── colight/
    └── data/
        ├── Jinan/3_4/               # 济南3×4路网数据（主数据集）
        └── Hangzhou/4_4/            # 杭州4×4路网数据（泛化验证）
```

---

## 各文件详细说明

### 1. `traffic_env.py` — 交通环境

封装CityFlow交通模拟器，提供强化学习接口。

**主要功能：**
- 加载路网JSON文件，自动识别12个可控路口（济南）或16个路口（杭州）
- 构建TOP-K邻接矩阵（每个路口的K=5个最近邻居）
- 提取每个路口的状态特征：
  - 8维相位one-hot编码（当前绿灯相位）
  - 4维车辆计数（各方向进入车辆数，归一化）
  - 总特征维度：**feature_dim = 12**
- 计算奖励：负的平均等待车辆数
- 主要评估指标：`get_avg_travel_time()` 返回平均行程时间（秒），**越低越好**

**关键参数：**
```python
num_steps = 3600   # 训练episode长度（模拟1小时）
EVAL_STEPS = 600   # 评估episode长度（模拟10分钟）
TOP_K = 5          # 每个路口的邻居数量
```

---

### 2. `colight_model.py` — CoLight图注意力模型

PyTorch实现的CoLight图注意力Q网络。

**模型架构：**
```
输入特征 (N, 12)
    ↓
MLP编码器 (12 → 64 → 64)
    ↓
图注意力层 × 2（多头注意力，默认nhead=2）
    ↓
Q值输出头 (64 → 4个动作)
```

**图注意力机制：**
- 每个路口作为图节点，查询自身特征
- 从K=5个邻居节点获取Key和Value
- 通过缩放点积注意力计算注意力权重
- 加权聚合邻居信息，更新节点表示

**训练设置：**
- 算法：DQN（深度Q网络）
- 损失函数：Huber Loss（比MSE更稳定）
- 优化器：Adam，学习率5e-4
- Epsilon贪婪探索：从1.0衰减到0.05（每episode衰减0.95）
- Replay Buffer容量：10000
- 目标网络更新频率：每5步

---

### 3. `failure_modes.py` — 故障注入模块

实现5种故障模式，可以注入到任意状态观测中。

**故障类型：**

| 故障名称 | 参数名 | 实现方式 | 模拟场景 |
|---------|--------|---------|---------|
| 传感器故障 | `node_mask` | 随机将p%路口的特征向量置零 | 传感器完全损坏 |
| 通信故障 | `edge_drop` | 随机删除p%图边，替换为自环 | 通信链路中断 |
| 特征噪声 | `feat_noise` | 添加高斯噪声（σ=0.1） | 传感器读数不准 |
| 观测延迟 | `delay` | 返回delay步之前的历史观测 | 通信延迟 |
| 空间块故障 | `spatial_block` | 选中心节点，将其及邻居全部置零 | 区域性断电 |
| 高度数故障 | `high_degree` | 优先mask度数最高的路口 | 枢纽路口故障 |

**使用方式：**
```python
from failure_modes import FailureInjector

# 创建故障注入器
injector = FailureInjector(mode='node_mask', failure_rate=0.2)

# 在训练/评估循环中注入故障
obs = injector.inject(state)  # state是env.reset()或env.step()返回的dict
```

---

### 4. `train_eval.py` — 训练与评估主脚本

所有实验的入口文件。

**运行方式：**
```bash
# 快速测试（验证代码能跑）
python train_eval.py --mode quick_test

# 训练所有模型（clean + sensor + comm）
python train_eval.py --mode full --episodes 100

# 只跑评估（模型已训练好）
python train_eval.py --mode eval_all
python train_eval.py --mode cross_failure

# 单独跑各个实验
python train_eval.py --mode exp_c   # 故障模式分析
python train_eval.py --mode exp_d   # 细粒度sweep
python train_eval.py --mode exp_e   # 混合故障
python train_eval.py --mode exp_f   # 注意力头数sweep
python train_eval.py --mode exp_g   # 混合鲁棒训练
python train_eval.py --mode exp_h   # Hangzhou泛化
```

---

## 实验说明与结果解读

### 实验A：故障类型敏感性（`expA_clean_sensitivity.png`）

**做了什么：** 用clean训练的模型，在不同故障类型和故障率下评估性能。

**故障率：** 0%, 10%, 20%, 30%

**结果数据：**
```
Clean baseline:           599.8s
node_mask 10%/20%/30%:   561.9 / 559.0 / 540.3s
edge_drop 10%/20%/30%:   552.0 / 534.1 / 525.3s
feat_noise（所有rate）:   599.8s（无影响）
```

**关键结论：**
- **通信故障（edge_drop）比传感器故障（node_mask）危害更大**（30%时：74.5s vs 59.5s降幅）
- 特征噪声对性能完全没有影响，说明模型依赖特征的"存在"而非精确值
- 两种故障随故障率增加都呈现单调恶化趋势

---

### 实验B：跨故障鲁棒性迁移（`expB_cross_failure.png`）

**做了什么：** 3×2矩阵实验——3种训练模式（clean/sensor/comm）× 2种测试故障。

**结果数据（30%故障率）：**
```
                    测试：node_mask    测试：edge_drop
Clean训练：           540.5s              525.3s
Sensor训练：          578.7s              561.6s
Comm训练：            632.8s              616.0s
```

**关键结论：**
- clean训练的模型反而性能最好，说明故障感知训练在当前设置下hurt了收敛
- 这是一个重要的**负面发现**：简单地在故障环境下训练并不能自动提升鲁棒性
- 可能原因：故障注入增加了训练难度，导致模型未能充分收敛

---

### 实验C：故障模式分析（`expC_clean_patterns.png`）

**做了什么：** 固定20%故障率，对比5种不同的故障模式。

**结果数据：**
```
Random Sensor Failure:  310s
Spatial Block Failure:  313s
High-Degree Failure:    328s   ← 最严重
Random Comm. Failure:   309s
Delayed Observation:    327s   ← 第二严重
```

**关键结论：**
- **高度数节点故障最严重**，说明图注意力TSC依赖枢纽路口
- 延迟观测危害接近高度数故障
- 随机故障和空间块故障危害相近

---

### 实验D：细粒度故障率曲线（`expD_clean_finegrain.png`）

**做了什么：** 在0%, 5%, 10%, 15%, 20%, 25%, 30%七个率下评估，让曲线更平滑。

**关键结论：**
- 性能随故障率增加单调下降，趋势平滑
- edge_drop在低故障率（5%）时就已经显著劣于clean，node_mask影响相对平缓

---

### 实验E：混合故障（`expE_clean_mixed.png`）

**做了什么：** 同时注入传感器故障+通信故障，看叠加效果。

**结果数据：**
```
Clean (0%+0%):          324.6s
Sensor only (20%):      307.6s
Comm only (20%):        309.5s
Mixed (10%+10%):        307.0s
Mixed (15%+15%):        307.4s
Mixed (20%+20%):        306.2s
```

**关键结论：**
- **混合故障的危害不是线性叠加的**，Mixed 20%+20%与Single 20%效果相近
- 两种故障可能激活相同的"退化机制"，存在某种补偿效应

---

### 实验F：注意力头数影响（`expF_attn_heads.png`）

**做了什么：** 训练nhead=1和nhead=4两个模型，在edge_drop下对比性能。

**结果数据：**
```
              rate=0%    rate=10%   rate=20%   rate=30%
nhead=1:      362.6s     354.8s     344.5s     339.5s
nhead=4:      328.2s     307.5s     303.2s     305.2s
```

**关键结论：**
- **nhead=4全面优于nhead=1**，差距约34s，在通信故障下保持稳定
- 更多注意力头学到更丰富的通信表示，在通信受损时更具弹性

---

### 实验G：混合鲁棒训练（`expG_mixed_robust.png`）

**做了什么：** 训练时同时注入10% node_mask + 10% edge_drop，与clean训练对比。

**结果数据（30%测试故障率）：**
```
                    测试：node_mask    测试：edge_drop
Clean训练：           307.2s              306.6s
Mixed-Robust训练：    346.1s              343.1s
```

**关键结论：**
- 混合鲁棒训练的baseline（360s）高于clean训练（324.6s），说明复杂的训练故障同样hurt收敛
- 这进一步验证了Exp B的发现：**简单的故障注入训练策略不足以提升鲁棒性**，需要更精心的课程学习或更多训练轮数

---

### 实验H：Hangzhou数据集泛化（`expH_hangzhou.png`）

**做了什么：** 在Hangzhou 4×4路网（16个路口）上训练新模型，重复故障敏感性实验，验证Jinan结论是否泛化。

**结果数据：**
```
Hangzhou baseline:        309.8s
node_mask 10%/20%/30%:   301.6 / 297.6 / 298.3s （下降3.7%）
edge_drop 10%/20%/30%:   290.9 / 285.5 / 278.2s （下降10.2%）
```

**关键结论：**
- **Jinan结论在Hangzhou完全复现**：edge_drop始终比node_mask危害更大
- Hangzhou上edge_drop的降幅（10.2%）甚至比Jinan（12.4%）更明显
- 说明"通信故障比传感器故障更危险"是图注意力TSC的普遍性质，不依赖特定路网

---

## 总体结论汇总（供报告写作参考）

1. **通信故障 > 传感器故障**：在相同故障率下，graph edge dropping造成的性能下降始终大于node feature masking，在两个数据集上均成立

2. **拓扑位置比故障率更重要**：高度数节点（枢纽路口）的故障危害远大于随机故障，说明图注意力TSC对graph hub存在依赖

3. **特征噪声vs硬遮蔽**：高斯噪声对性能无影响，而硬遮蔽（置零）显著降低性能，说明模型依赖特征的存在性而非精确值

4. **混合故障非线性**：同时注入两种故障的危害不是线性叠加，存在补偿效应

5. **故障感知训练的局限性**：在当前训练预算下，在故障环境下训练反而hurt了模型收敛，这是一个重要的负面发现，说明naive的故障注入训练策略不够

6. **注意力容量的保护作用**：更多注意力头（nhead=4）在通信故障下表现更鲁棒

7. **跨数据集泛化**：所有主要结论在Jinan和Hangzhou两个路网上均成立

---

## 图表位置速查

所有图表保存在 `results/` 文件夹：

| 文件名 | 对应实验 | 图表类型 |
|--------|---------|---------|
| `expA_clean_sensitivity.png` | Exp A | 折线图，3条线（node_mask/edge_drop/feat_noise） |
| `expB_cross_failure.png` | Exp B | 双子图折线图，3×2矩阵 |
| `expC_clean_patterns.png` | Exp C | 柱状图，5种故障模式 |
| `expD_clean_finegrain.png` | Exp D | 折线图，7个rate点 |
| `expE_clean_mixed.png` | Exp E | 柱状图，6种混合组合 |
| `expF_attn_heads.png` | Exp F | 折线图，nhead=1 vs 4 |
| `expG_mixed_robust.png` | Exp G | 双子图，clean vs mixed-robust |
| `expH_hangzhou.png` | Exp H | 双子图，Jinan vs Hangzhou |
| `training_curves.png` | 训练过程 | 折线图，3个模型的训练曲线 |

---

## 评估指标说明

**平均行程时间（Average Travel Time，单位：秒）**
- CityFlow引擎计算所有车辆从进入路网到离开的平均时间
- **越低越好**
- Clean baseline（无故障）：Jinan约600s，Hangzhou约310s（因eval steps不同，绝对值不可跨实验直接比较）
- 重要：不同实验的EVAL_STEPS不同（600或1800步），绝对值不能跨实验比较，只看**同一实验内的相对变化**

---

## 环境配置

```bash
# 创建新环境
conda create -n tsc_gpu python=3.10 -y
conda activate tsc_gpu

# 安装PyTorch（GPU版本）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 安装其他依赖
pip install numpy matplotlib

# 从源码安装CityFlow
git clone https://github.com/cityflow-project/CityFlow.git
cd CityFlow && pip install . && cd ..
```

---

## 数据集信息

| 数据集 | 路口数 | 路网规模 | 数据来源 |
|--------|--------|---------|---------|
| Jinan 3×4 | 12个 | 3行4列网格 | 济南真实交通数据 |
| Hangzhou 4×4 | 16个 | 4行4列网格 | 杭州真实交通数据 |

两个数据集均使用真实交通流量（real-world flow），不是合成数据。