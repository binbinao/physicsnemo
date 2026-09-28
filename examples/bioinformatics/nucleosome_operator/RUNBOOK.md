# 核小体定位算子 操作手册（Runbook）

手把手从零跑通本案例：安装 → 冒烟 → 训练 → 验收 → 推理 → 消融。
每步给命令、预期输出、故障判断点。原理见 `README.md`，API 参考见 `USER_GUIDE.md`。

## 前置条件

- Linux x86_64，Python 3.11–3.13
- NVIDIA GPU（CUDA 12.x；本案例在 Tesla T4 验证）。无 GPU 可用 CPU 跑冒烟，但完整训练不现实
- 已 clone physicsnemo 仓库

## 步骤 1：安装依赖（2 分钟）

```bash
cd /path/to/physicsnemo
pip install -e .
pip install -r examples/bioinformatics/nucleosome_operator/requirements.txt
```

**验证**：

```bash
python -c "import torch, hydra, matplotlib, physicsnemo; print('ok')"
# 预期输出: ok
```

**故障**：

- `ModuleNotFoundError: physicsnemo` → 先确保在仓库根目录执行了 `pip install -e .`
- `torch.cuda.is_available()` 为 False → 检查 `nvidia-smi`；无 GPU 则所有命令加 `device=cpu` 且只做冒烟

## 步骤 2：进入案例目录

```bash
cd examples/bioinformatics/nucleosome_operator
```

后续命令都在此目录运行（Hydra 运行目录是相对路径 `outputs/`）。

## 步骤 3：冒烟测试（1 分钟）

```bash
python train.py steps=200 n_train=500 n_val=128 device=cpu val_every=100
```

**预期输出**（关键行）：

```text
[...][main][INFO] - training on cpu
[...][main][INFO] - energy table: seed=20260920 width=147 scale=0.22
[...][main][INFO] - running reference-solver self-tests...
[selftest] brute force: max occupancy deviation 1.78e-15 (L=22, W=5)
[selftest] gamma=0 exact: Z=380989 rel err 0.00e+00, occupancy dev 1.89e-15
[selftest] energy landscape: naive window-sum deviation 0.00e+00
[selftest] limits: favoured mean occupancy 0.9702, penalised 2.38e-06
[selftest] gamma contrast: 0.1578 -> 0.2114 -> 0.2791 (gamma 0.5/1/2)
[selftest] corpus: mean occupancy 0.8389, 16/64 planted rows
[...][main][INFO] - self-tests passed
[...][main][INFO] - corpus: 500 train / 128 val sequences of 1024 bp in ...s
[...][main][INFO] - checkpoint written to outputs/<date>/<time>/model.pt
```

**判断点**：

- 看到 6 行 `[selftest]` 且无异常 → 参考解（标签生成器）正确
- 看到 `self-tests passed` → 物理自检门通过
- 看到 `checkpoint written` → 训练链路通
- 若抛 `SelfTestFailure` → **不要绕过**，跑 `python -m pytest tests/test_nucleosome.py -q` 定位

## 步骤 4：完整训练（T4 约 10-30 分钟，取决于配置）

```bash
python train.py
```

提交配置即复现 `README.md` 的验收数字（16000 步，Tesla T4 约 29 分钟；语料生成 ~58 s）。
**预期输出**：

```text
[...][main][INFO] - corpus: 12000 train / 512 val sequences of 1024 bp in ~58s
[...][main][INFO] - operator: 14219873 parameters
[...][main][INFO] - step    250/16000  loss 5.1e-02  lr 1.00e-03
[...][main][INFO] - step   1000/16000  val rel L2  median ...%  P95 ...%  max ...%
...
[...][main][INFO] - final val rel L2: median 1.5530%  P95 6.8766%  max 20.4487%
[...][main][INFO] - checkpoint written to outputs/2026-09-28/<time>/model.pt
```

**判断点**：

- `val rel L2` 的 median 应随训练整体下降；若长时间不动，检查 `lr` 是否被覆盖
- 验证集是**同分布但不同样本**（`seed+1`）的 512 条序列，用来选 checkpoint，不是泛化指标
- 训练结束后 `outputs/<date>/<time>/` 下应有 `model.pt` 与 `summary.json`

## 步骤 5：验收评估（1 分钟）

```bash
python evaluate.py
```

**预期输出**：

```text
checkpoint: .../outputs/2026-09-28/<time>/model.pt
corpus: 600 held-out sequences of 1024 bp (test_seed=20260920)
========================================================================
ACCEPTANCE EVALUATION (spec section 6)
========================================================================
occupancy rel L2 err:  median 1.6421%  P95 6.9781%  max 18.8585%
                       (thresholds median 2%, P95 8%)
  stratum random         median 1.5845%  P95 7.0764%  max 18.8585%
  stratum planted_motif  median 1.7456%  P95 6.6711%  max 10.1256%
  gamma q1 [0.50, 0.70]  median 0.7869%  P95 1.1208%
  gamma q2 [0.70, 1.04]  median 1.2258%  P95 2.6839%
  gamma q3 [1.04, 1.42]  median 2.1799%  P95 5.2687%
  gamma q4 [1.42, 1.99]  median 3.7308%  P95 9.8985%
mean occupancy: reference 0.8291  operator 0.8288
output range: [0.0000, 1.0000]  finite True
gamma-sweep contrast trend agreement: 100% (threshold 80%)
length extrapolation to 2048 bp: median 20.5114%  P95 34.3262%
========================================================================
VERDICT: PASS
========================================================================
```

**判断点**：

- `VERDICT: PASS` + exit code 0 → 算子达标
- `VERDICT: FAIL` → 检查是否用了冒烟 checkpoint（步数太少），或容量不足
- 数字与 README 表格不完全一致属正常（GPU 浮点非确定性），但量级必须一致
- 若 `gamma-sweep contrast trend agreement` 低于阈值 → 算子没有抓住"γ 增大 ⇒ 定位更
  锐利"这条物理规律，即使误差达标也应视为失败

**产出**（写入 checkpoint 同目录）：`error_report.json`、`occupancy_profiles.png`、
`gamma_sweep.png`、`error_structure.png`。

```bash
# 查看图片（远程则先 scp 下来）
python -c "import webbrowser"  # 或直接打开
ls outputs/<date>/<time>/
```

## 步骤 6：推理（用训练好的算子预测自己的序列）

```bash
python - << 'EOF'
import sys, glob, os, torch
sys.path.insert(0, ".")
from helpers import data, energy
from helpers import model as model_mod

ckpt_path = max(glob.glob("outputs/**/model.pt", recursive=True), key=os.path.getmtime)
print("loading", ckpt_path)
ckpt = torch.load(ckpt_path, weights_only=True, map_location="cpu")
cfg = dict(ckpt["cfg"])
table = energy.make_energy_table(
    seed=int(cfg["table_seed"]), width=int(cfg["width"]), scale=float(cfg["energy_scale"])
)
operator = model_mod.NucleosomeOperator(
    latent_channels=int(cfg["latent_channels"]), num_fno_layers=int(cfg["num_fno_layers"]),
    num_fno_modes=int(cfg["num_fno_modes"]), padding=int(cfg["padding"]),
)
operator.load_state_dict(ckpt["state_dict"]); operator.eval()

seq = torch.tensor([[0, 1, 2, 3] * 256], dtype=torch.int8)     # A=0 C=1 G=2 T=3
gamma = torch.tensor([1.0], dtype=torch.float64)
energies = energy.sliding_energies(seq, table)
landscape = energy.energy_channel(energies, seq.shape[1])
x = data.encode_input(seq, gamma, landscape, cfg["normalization"])
with torch.no_grad():
    p = operator(x)
print("occupancy:", tuple(p.shape), "mean", float(p.mean()))
print("min", float(p.min()), "max", float(p.max()))
EOF
```

**预期输出**：

```text
loading outputs/2026-09-28/14-41-43/model.pt
occupancy: (1, 1024) mean 0.8148
min 0.0079 max 0.9936
```

**判断点**：`mean` 应在 0.5–0.95（语料平均占据度约 0.83）；`min/max` 必须落在 `[0,1]`
（sigmoid 保证）。若 mean 异常（接近 0 或 1），说明输入编码与 checkpoint 不匹配
（例如漏了 `cfg["normalization"]`）。

上例是程序化周期序列（`ACGT` 重复），mean 0.81 属正常范围；换真实序列
（例如某段基因组窗口）时，mean 会随 GC 与序列结构变化。

## 步骤 7：感受野消融（证明需要算子而非局部 CNN，约 5-10 分钟）

```bash
python ablate_receptive_field.py --steps 4000   # 等预算对照（默认取 checkpoint 的步数）
```

**预期输出**：

```text
FNO baseline: median 1.6421%  P95 6.9781%  (14219873 params)
conv RF 41 bp: median 19.8825%  P95 33.2363%  (114369 params)
conv RF 129 bp: median 16.6130%  P95 29.3176%  (114369 params)
========================================================================
Receptive-field ablation at equal budget (16000 steps)
========================================================================
surrogate                    field (bp)    params    median       P95
fno (spectral, full locus)         1024  14219873    1.64%     6.98%
conv RF 41 bp                        41    114369   19.88%    33.24%
conv RF 129 bp                      129    114369   16.61%    29.32%
```

**判断点**：固定预算下局部 CNN 的误差应显著大于谱算子；感受野越大误差越小
——这就是"必须用算子"的量化证据。实测局部 CNN 的 17-20% 中位误差与"无序列依赖"的
几何基线（18.97%）同级，说明它们根本没学到序列调制。报告写入
`outputs/receptive_field_ablation.json`。

## 步骤 8：查看运行配置（可选）

```bash
cat outputs/<date>/<time>/.hydra/config.yaml      # 实际生效的完整配置
cat outputs/<date>/<time>/.hydra/overrides.yaml   # 命令行覆盖记录
cat outputs/<date>/<time>/summary.json            # 参数量 / 验证指标 / 自证诊断
```

## 常见故障汇总

| 症状 | 诊断 | 修复 |
|------|------|------|
| `SelfTestFailure` | 参考解/语料契约被破坏 | 跑 `pytest tests/test_nucleosome.py -q` 定位；不要绕过门槛 |
| `ModuleNotFoundError: hydra` | 没装 requirements | `pip install -r requirements.txt` |
| `CUDA out of memory` | `n_train`/`batch_size` 太大 | `python train.py n_train=2000 batch_size=32` 或 `device=cpu` |
| 评估 FAIL | 用了冒烟 checkpoint 或容量不足 | 跑完整 `python train.py`，或显式指定 checkpoint |
| `evaluate.py` 找到的是冒烟 checkpoint | 默认取**最新**的 `model.pt` | `rm -rf outputs/<冒烟时间戳>/`，或显式传路径 |
| 推理结果异常（mean≈0 或 1） | 输入编码漏了标准化常数 | 用 `cfg["normalization"]`（见步骤 6 代码） |
| 训练慢 | GPU 被其它进程占用 | `nvidia-smi` 查看；本案例单步约 25-80 ms（batch 64, L=1024） |

## 下一步

- 换长度/能量表/γ 范围/容量：见 `USER_GUIDE.md` §6 的参数表与示例
- 想做真实基因组推理：先读 `README.md` 的"限制与适用范围"，真实基因组的重复结构
  超出本案例语料分布
- 想扩成别的生信读出（如剪接位点、转录因子结合）：替换 `helpers/energy.py` 的景观
  定义与 `helpers/nucleosome.py` 的参考解即可，算子与训练链路不变
