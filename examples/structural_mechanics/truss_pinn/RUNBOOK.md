# Truss PINN 操作手册（Runbook）

手把手从零跑通本案例：环境 → 训练 → 评估 → 推理。每步给命令、预期输出、故障判断点。
架构原理见 `README.md`，API 参考见 `USER_GUIDE.md`。

## 前置条件

- Linux x86_64，Python 3.11–3.13
- NVIDIA GPU（CUDA 12.x；本案例在 Tesla T4 验证）。无 GPU 可用 CPU 冒烟，但 40000 步训练不现实
- 已 clone physicsnemo 仓库

## 步骤 1：安装依赖（2 分钟）

```bash
cd /path/to/physicsnemo
pip install -e .
pip install -r examples/structural_mechanics/truss_pinn/requirements.txt
```

**验证**：

```bash
python -c "import physicsnemo, torch, hydra, sympy, matplotlib; print('ok')"
# 预期输出: ok
```

**故障**：
- `ModuleNotFoundError: sympy` → `pip install sympy>=1.12`（requirements.txt 已含，但手动单装 physicsnemo 会漏）
- `torch.cuda.is_available()` 返回 False → 检查 `nvidia-smi`；无 GPU 则所有 `device=cuda` 改为 `device=cpu`，且只做冒烟不做完整训练

## 步骤 2：进入案例目录

```bash
cd examples/structural_mechanics/truss_pinn
```

**后续所有命令都在此目录运行**（Hydra 运行目录是相对路径 `outputs/`）。

## 步骤 3：冒烟测试（3 分钟，可选但推荐）

验证环境正确，不练完：

```bash
python train.py epochs=400 device=cpu log_every=100
```

**预期输出**（关键行）：

```text
[2026-09-23 12:05:36,580][main][INFO] - froze stability envelope P_MAX = 11871.275316 N
[2026-09-23 12:05:36,581][main][INFO] - running FEM physics self-tests...
[2026-09-23 12:05:36,977][main][INFO] - FEM self-tests passed
[2026-09-23 12:05:37,376][main][INFO] - step      1  eq=3.9650e+07 sup=1.2957e-02 mode=3.2287e+01 eig_reg=3.3342e+07  [328.3 ms/step]
[2026-09-23 12:05:37,658][main][INFO] - step      5  eq=3.5066e+07 sup=1.2860e-02 mode=3.2111e+01 eig_reg=9.4056e+06  [122.0 ms/step]
...
checkpoint written to outputs/<date>/<time>/model.pt
```

**判断点**：
- 看到 `FEM self-tests passed` → 物理求解器正确
- 看到 `checkpoint written` → 流程通
- 若报稳定性扫描失败 → 几何文件被改动过，恢复 `helpers/geometry.py`

## 步骤 4：完整训练（~23 分钟，Tesla T4）

```bash
python train.py
```

**预期输出**：

```text
[2026-09-23 ...][main][INFO] - froze stability envelope P_MAX = 11871.275316 N
[2026-09-23 ...][main][INFO] - running FEM physics self-tests...
[2026-09-23 ...][main][INFO] - FEM self-tests passed
[2026-09-23 ...][main][INFO] - step   1000  eq=1.2e+02 sup=3.4e-01 mode=2.1e+01 eig_reg=...  [... ms/step]
[2026-09-23 ...][main][INFO] - val disp_rel_l2=... freq_rel_err=...
...
[2026-09-23 ...][main][INFO] - step  40000  ...
checkpoint written to outputs/2026-09-23/<time>/model.pt
```

**判断点**：
- val `disp_rel_l2` 与 `freq_rel_err` 应整体随训练下降（in-sample 监控，非泛化指标）
- 若 loss 震荡不降：检查 `weight_eq/sup/mode` 是否被覆盖为失衡值；默认值经扫参平衡
- 中断续练：不支持断点续训，需从头重跑（23 分钟可接受）

**产出**：`outputs/<date>/<time>/model.pt`（checkpoint，含 state_dict + 配置 + 冻结 P_MAX）。

## 步骤 5：验收评估（~10 秒）

```bash
python evaluate.py
```

**预期输出**：

```text
displacement rel L2 err:  median 0.2771%  P95 3.3639%  max 15.2315%  (threshold 5%)
freq rel err mode 1:    median 0.2427%  P95 0.4897%  max 0.8715%  (threshold 2%)
freq rel err mode 2:    median 0.2608%  P95 0.5197%  max 0.6717%  (threshold 2%)
freq rel err mode 3:    median 0.2967%  P95 0.4931%  max 0.5711%  (threshold 2%)
mode 1 shape cos sim: median 1.0000  P95 1.0000  (informative)
...
omega_1(P) trend agreement with FEM: 90%
omega_2(P) trend agreement with FEM: 82%
omega_3(P) trend agreement with FEM: 100%
========================================================================
VERDICT: PASS
========================================================================
```

**判断点**：
- `VERDICT: PASS` + exit code 0 → 模型达标
- 数字与上表不完全一致 → 正常（GPU 浮点非确定性），但数量级与趋势必须一致
- `VERDICT: FAIL` → 检查是否用了完整 40000 步训练（冒烟 checkpoint 不达标）

**产出**（写入 checkpoint 目录）：
- `error_report.json`——200 工况逐案例误差 + 聚合 + verdict
- `deformation_comparison.png`——3 工况 FEM vs PINN 变形叠加图
- `frequency_load_curves.png`——ω_i(P) 曲线（FEM 实线 + PINN 散点）

查看图片：

```bash
# 本地查看
eog outputs/<date>/<time>/deformation_comparison.png
eog outputs/<date>/<time>/frequency_load_curves.png
# 或远程：scp 到本地
```

## 步骤 6：用模型做预测（推理）

```bash
python - << 'EOF'
import sys, torch, glob, os
sys.path.insert(0, ".")
from helpers.model import TrussPINN, encode_params
from helpers import geometry

# 找最新 checkpoint
ckpt_path = max(glob.glob("outputs/**/model.pt", recursive=True), key=os.path.getmtime)
print(f"loading {ckpt_path}")
ckpt = torch.load(ckpt_path, weights_only=True, map_location="cpu")
geometry.P_MAX = float(ckpt["p_max"])
model = TrussPINN(); model.load_state_dict(ckpt["state_dict"]); model.eval()

# 预测：节点 1 向下 5000 N；节点 4 向右 8000 N
n_idx = torch.tensor([1, 4]); theta = torch.tensor([-torch.pi/2, 0.0]); P = torch.tensor([-5000.0, 8000.0])
x = encode_params(n_idx, theta, P)
with torch.no_grad():
    u_hat, log_omega_hat, phi_hat = model(x)

print(f"位移 (11 自由 DOF, 米):\n{u_hat}")
print(f"频率 (3 阶, rad/s):\n{log_omega_hat.exp()}")
print(f"振型 (11 DOF x 3 模态):\n{phi_hat}")
EOF
```

**预期输出**：

```text
loading outputs/2026-09-23/<time>/model.pt
位移 (11 自由 DOF, 米):
tensor([[...11 个值...], [...11 个值...]], dtype=torch.float64)
频率 (3 阶, rad/s):
tensor([[...3 个值...], [...3 个值...]], dtype=torch.float64)
振型 (11 DOF x 3 模态):
tensor([...])
```

**参数含义**：
- `n_idx`：载荷节点号，必须是 `geometry.LOADABLE_NODES` 中的 `(1, 2, 4, 5, 6)` 之一
- `theta`：弧度，`-π/2` = 向下，`0` = 向右，`π/2` = 向上
- `P`：有符号牛顿，负 = 反向；|P| 不得超过 P_MAX（11871 N）

**11 个自由 DOF → 完整 14 维位移的映射**（见 `USER_GUIDE.md` §4.1）。

## 步骤 7：查看运行配置（可选）

```bash
cat outputs/<date>/<time>/.hydra/config.yaml    # 实际生效的完整配置
cat outputs/<date>/<time>/.hydra/overrides.yaml # 命令行覆盖记录
```

## 常见故障汇总

| 症状 | 诊断 | 修复 |
|------|------|------|
| `ModuleNotFoundError: sympy` | physicsnemo 单装漏 sympy | `pip install sympy>=1.12` |
| `CUDA out of memory` | batch_size 256 对 T4 正常；若更小 GPU | `python train.py batch_size=128` |
| 评估 FAIL | 用了冒烟 checkpoint | 跑完整 `python train.py`（40000 步），或显式指定 checkpoint：`python evaluate.py outputs/<完整训练的date/time>/model.pt` |
| `evaluate.py` 找到的是冒烟 checkpoint | 步骤 3 的冒烟也产出了 checkpoint，`evaluate.py` 默认找**最新** | 删除冒烟目录：`rm -rf outputs/<冒烟date/time>/`，或用上条显式指定 |
| `evaluate.py` 找不到 checkpoint | 没练过或目录不对 | 确认在 `truss_pinn/` 目录且 `outputs/` 存在 |
| 推理频率为负/NaN | 混用了不同 P_MAX 的 checkpoint 与代码 | 确保 `geometry.P_MAX` 取自同一 checkpoint |
| 位移误差 >15% | mid-gap theta + 小 \|P\| 的已知限制 | 见 `README.md` 误差结构分析；避免该参数区间 |

## 下一步

- 改拓扑/材料：编辑 `helpers/geometry.py`（节点坐标、杆件连接、材料常数），删 `outputs/` 重跑
- 改训练超参：`python train.py <key>=<value>`，或编辑 `conf/config.yaml`
- 集成到自己代码：见 `USER_GUIDE.md` §4 的推理 API
