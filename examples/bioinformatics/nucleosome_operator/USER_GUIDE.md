# 核小体定位算子（Nucleosome Positioning Operator）使用文档

本文档面向**使用者**：安装、训练、评估，以及用训练好的算子对自己的 DNA 序列做预测。
原理与验收细节见 `README.md`（英文，仓库规范文档）。

## 1. 这个案例做什么

**任务**：给定一段 DNA 序列，预测**核小体占据度剖面** `p(x) ∈ [0,1]` —— 即每个碱基
位置被核小体覆盖的概率。这是序列基因组学的核心读出（MNase-seq / ATAC-seq 测的就是
它），也是"序列 → 连续场"的算子学习问题。

**机制模型（标签来源）**：热力学描述（Kaplan et al., Nature 2009 的思路）

```text
DNA 序列 ──位置特异二核苷酸能量表──▶ 足迹能量 E(i)
        ──Boltzmann 配分函数 Z = Σ_S ∏_{i∈S} exp(−γE(i))──▶ 占据度 p(x)
```

其中 `S` 遍历所有互不重叠的 147 bp 足迹起点集合。标签由**精确转移矩阵（配分函数
动态规划）**在对数空间算出，不含拟合参数、不含外部数据。

**算子**：1D Fourier 神经算子（FNO，`physicsnemo.models.fno.FNO`）+ logistic 输出头。

| 项 | 形状 | 含义 |
|---|---|---|
| 输入 | `(n, 6, L)` | 通道 0-3 = 碱基 one-hot（A/C/G/T）；通道 4 = 标准化 Boltzmann 指数 `−γE`；通道 5 = 标准化逆温 `γ` |
| 输出 | `(n, L)` | 占据度 `p(x)`，由 sigmoid 保证落在 `[0,1]` |
| 训练长度 | `L = 1024 bp` | 算子可外推到更长位点（谱模态数固定） |

## 2. 安装

```bash
cd /path/to/physicsnemo
pip install -e .
pip install -r examples/bioinformatics/nucleosome_operator/requirements.txt
```

需要 PyTorch（GPU 训练；CPU 可跑冒烟与小规模评估）。

## 3. 训练

```bash
cd examples/bioinformatics/nucleosome_operator
python train.py                      # 完整训练（cuda 可用时）
python train.py steps=300 device=cpu # 冒烟
```

启动顺序：冻结能量表 → **参考解自证测试**（失败即中止）→ 生成语料并解出精确标签
→ 训练 → 写 checkpoint 到 Hydra 运行目录 `outputs/<date>/<time>/model.pt`。

### 配置参数（`conf/config.yaml`，全部可用命令行覆盖）

| 参数 | 默认 | 含义 |
|---|---|---|
| `seed` | 95051 | 训练随机种子（Collocation 采样与初始化） |
| `steps` | 16000 | 优化步数 |
| `batch_size` | 64 | 每步序列数 |
| `lr` | 1.0e-3 | Adam 初始学习率 |
| `lr_step_size` | 6000 | StepLR 衰减间隔（0 = 关闭） |
| `lr_gamma` | 0.3 | StepLR 衰减因子 |
| `loss` | rel_l2 | 损失（`rel_l2` 或 `mse`） |
| `device` | cuda | 训练设备 |
| `log_every` | 250 | 训练日志间隔 |
| `val_every` | 1000 | 验证评估间隔 |
| `n_train` | 12000 | 训练序列数 |
| `n_val` | 512 | 验证序列数 |
| `n_test` | 600 | 验收测试序列数 |
| `seq_length` | 1024 | 位点长度（bp） |
| `gc_low`, `gc_high` | 0.30, 0.70 | 每条序列的 GC 含量范围 |
| `planted_fraction` | 0.25 | 植入基序的序列比例 |
| `gamma_low`, `gamma_high` | 0.5, 2.0 | 逆温 γ 的对数均匀范围 |
| `val_seed_offset` | 1 | 验证集种子 = `seed + 该值` |
| `test_seed` | 20260920 | 测试集种子（固定，评估可复现） |
| `table_seed` | 20260920 | 能量表种子（checkpoint 契约的一部分） |
| `energy_scale` | 0.22 | 能量表每项标准差（决定序列依赖强度） |
| `width` | 147 | 核小体足迹（bp） |
| `latent_channels` | 96 | 谱层宽度 |
| `num_fno_layers` | 6 | 谱卷积层数 |
| `num_fno_modes` | 128 | 保留的 Fourier 模态数 |
| `padding` | 16 | 谱卷积域填充（常量填充，保持端点非周期） |

示例：

```bash
python train.py steps=200 device=cpu                 # 冒烟（1 分钟内）
python train.py n_train=2000 steps=4000             # 小规模实验
python train.py num_fno_modes=128 latent_channels=96 # 加容量
```

## 4. 评估

```bash
python evaluate.py                                    # 自动找最新 checkpoint
python evaluate.py outputs/2026-09-28/14-07-33/model.pt
```

评估在**独立种子**生成的 600 条未见序列上进行，输出（写入 checkpoint 同目录）：

- `error_report.json` —— 逐案例相对 L2 误差 + 聚合（median/P95/max）+ 分层诊断 + verdict
- `occupancy_profiles.png` —— 最好/中位/最差案例的参考 vs 预测剖面（含能量景观）
- `gamma_sweep.png` —— 固定序列上 γ 扫描的反差曲线与平均占据度（参考 vs 预测）
- `error_structure.png` —— 误差直方图 + 误差 vs γ（按随机/植入基序分层）

**验收口径**：逐案例相对 L2 误差的 **median AND P95** 同时低于阈值（PASS 时 exit 0）；
本案例记录的训练配置实测 median 1.64% / P95 6.98%（门槛 2% / 8%，见 README）。辅助诊断
（不进门槛）：分层误差、γ 分位误差、γ 扫描趋势一致率（实测 100%）、**长度外推**。

**已知误差结构**：误差随 γ 单调增大（γ 四分位 median 0.79% → 3.73%）——γ 越大
定位越锐利、剖面越接近阶跃，同样的绝对误差折算成相对误差越大。若你的工况集中在
γ > 1.4，预期单案例误差可达 ~10%（P95）。

**长度外推不成立**：算子结构上可接受更长的位点，但在 2048 bp（训练长度的 2 倍）上
未重训时 median 误差约 20.5%。**换长度必须重训**。

## 5. 用训练好的算子做预测

```python
import sys, torch
sys.path.insert(0, "examples/bioinformatics/nucleosome_operator")

from helpers import data, energy
from helpers import model as model_mod

# 1) 载入 checkpoint（含权重、配置、标准化常数、能量表参数）
ckpt = torch.load("outputs/<date>/<time>/model.pt", weights_only=True, map_location="cpu")
cfg = dict(ckpt["cfg"])

table = energy.make_energy_table(
    seed=int(cfg["table_seed"]), width=int(cfg["width"]), scale=float(cfg["energy_scale"])
)
operator = model_mod.NucleosomeOperator(
    latent_channels=int(cfg["latent_channels"]),
    num_fno_layers=int(cfg["num_fno_layers"]),
    num_fno_modes=int(cfg["num_fno_modes"]),
    padding=int(cfg["padding"]),
)
operator.load_state_dict(ckpt["state_dict"])
operator.eval()

# 2) 准备自己的序列：int8 碱基码 A=0, C=1, G=2, T=3
seq = torch.tensor([[0, 1, 2, 3] * 256], dtype=torch.int8)   # (1, 1024)
gamma = torch.tensor([1.0], dtype=torch.float64)             # 逆温（序列依赖强度）

# 3) 序列 -> 能量景观 -> 算子输入
energies = energy.sliding_energies(seq, table)               # (1, L-W+1)
landscape = energy.energy_channel(energies, seq.shape[1])    # (1, L)
x = data.encode_input(seq, gamma, landscape, cfg["normalization"])

# 4) 预测
with torch.no_grad():
    occupancy = operator(x)                                  # (1, L) ∈ [0,1]
print(occupancy.shape, float(occupancy.mean()))
```

### 输出解读

- `occupancy[0, j]` = 位置 `j`（0-based，5'→3'）被核小体覆盖的概率。
- **平均占据度** ≈ 被覆盖的碱基比例；哺乳动物染色质典型值 ~0.8（本案例语料实测
  约 0.83）。
- **剖面反差**（`occupancy.std(-1)`）随 `gamma` 增大而增大：γ 大 = 序列强烈决定
  定位（转录因子/核小体强偏好），γ 小 = 定位接近"几何基线"，序列影响弱。
- **长度**：可以传入长度不同于 1024 的序列（谱模态数固定），例如 2048；外推误差
  见 `README.md`。
- **适用范围**：本算子只在**本案例语料分布**上训练（i.i.d. 碱基 + 两类植入基序，
  `L = 1024`，`γ ∈ [0.5, 2]`，固定能量表）。真实基因组（重复序列、串联重复、
  非 i.i.d. 结构）与其它 `γ` 属外推，未做保证。

## 6. 自定义与扩展

| 想改什么 | 怎么做 |
|---|---|
| 换序列长度 | `python train.py seq_length=512`（也可只改评估：预测时传不同长度） |
| 换能量表 | `python train.py table_seed=<新种子> energy_scale=<尺度>`（自证门会重新校验） |
| 换 γ 范围 | `python train.py gamma_low=0.3 gamma_high=3.0` |
| 换足迹宽度 | `python train.py width=100`（DP 与能量表都按宽度自适应） |
| 加容量 | `python train.py num_fno_modes=128 latent_channels=96 num_fno_layers=6` |
| 换损失 | `python train.py loss=mse` |
| 证明非局部性 | `python ablate_receptive_field.py`（同预算下对比局部 CNN） |

## 7. 测试

```bash
cd examples/bioinformatics/nucleosome_operator
python -m pytest tests/ -q          # 生成器 / 景观 / 语料 / 模型 / 自证门
```

参考解的正确性由三类**互不共享代码路径**的 oracle 钉住：小位点穷举枚举、`γ=0` 的
精确 big-int 平铺计数、以及解析可解的两态特例。

## 8. 常见问题

- **自证门失败（`SelfTestFailure`）**：说明参考解或语料契约被破坏（改过
  `nucleosome.py` / `energy.py` / `genome.py`）。先跑 `pytest tests/test_nucleosome.py -q`
  定位，不要绕过门槛。
- **训练 loss 不降**：检查 `lr` / `steps` 是否被覆盖成极端值；`rel_l2` 损失在小批量
  下方差较大，看 `val_every` 的验证行而不是单步 loss。
- **评估 FAIL 但训练时 val 很好**：验证集与测试集种子不同（`seed+1` vs
  `test_seed`），分布相同但样本不同；FAIL 说明容量/步数不足，而不是过拟合。
- **想对真实基因组推理**：先用本案例语料上的误差水平评估可信度；真实基因组的
  重复结构超出语料分布，属外推。
- **`CUDA out of memory`**：把 `n_train` / `batch_size` 调小，或 `device=cpu`。
