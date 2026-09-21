# Truss PINN 用户使用文档

本文档面向**使用者**：如何安装、训练、评估，以及用训练好的模型对自己的载荷工况做预测。架构与验收细节见 `README.md`。

## 1. 安装

```bash
# 在仓库根目录安装 physicsnemo（editable）
pip install -e /path/to/physicsnemo

# 安装本例依赖（含 sympy：符号残差编译用）
pip install -r examples/structural_mechanics/truss_pinn/requirements.txt
```

需要 CUDA GPU（默认 `device: cuda`；无 GPU 时自动回退 CPU，训练时长不适用）。

## 2. 训练

```bash
cd examples/structural_mechanics/truss_pinn
python train.py
```

启动序列（~15 s）：冻结稳定包络 P_MAX → FEM 物理自检门 → 训练 40000 步（Tesla T4 约 23 分钟）。checkpoint 写入 Hydra 运行目录 `outputs/<date>/<time>/model.pt`。

### 配置参数（`conf/config.yaml`，全部可用命令行覆盖）

| 参数 | 默认 | 含义 |
|------|------|------|
| `seed` | 95051 | 训练随机种子（决定 collocation 采样与初始化） |
| `epochs` | 40000 | 训练步数 |
| `batch_size` | 256 | 每步 collocation 点数 |
| `lr` | 3.0e-3 | Adam 初始学习率 |
| `weight_eq` | 100.0 | 平衡残差项权重 |
| `weight_sup` | 1.0e8 | 锚点监督项权重 |
| `weight_mode` | 1.0e6 | 模态监督项权重 |
| `lr_step_size` | 10000 | StepLR 衰减间隔（0 = 关闭） |
| `lr_gamma` | 0.3 | StepLR 衰减因子 |
| `freeze_p_max` | true | 启动时是否重算 P_MAX 稳定性扫描 |
| `device` | cuda | 训练设备 |
| `log_every` | 1000 | 分量损失打印间隔 |
| `val_every` | 2000 | in-sample 监控切片评估间隔 |
| `test_seed` | 20260920 | 测试集种子（固定 200 工况，评估可复现） |

示例：

```bash
python train.py epochs=400 device=cpu     # 冒烟（几分钟）
python train.py lr=1e-3 log_every=100     # 覆盖任意键
python train.py freeze_p_max=false        # 跳过稳定性扫描（用已冻结的 P_MAX）
```

## 3. 评估

```bash
python evaluate.py                    # 自动找最新的 outputs/**/model.pt
python evaluate.py outputs/2026-09-21/10-27-02/model.pt   # 显式指定
```

输出（写入 checkpoint 所在目录）：

- `error_report.json`——200 工况逐案例误差 + 聚合（max/median/P95）+ PASS/FAIL
- `deformation_comparison.png`——3 个工况的 FEM vs PINN 变形叠加图
- `frequency_load_curves.png`——41 点 P 扫描的 ω_i(P) 曲线（FEM 实线 + PINN 散点）

评估准则（spec §2.4）：位移相对 L2 误差的 median AND P95 < 5%，且每阶频率相对误差的 median AND P95 < 2%。`evaluate.py` 打印 VERDICT 并以 exit code 报告（0 = PASS）。

## 4. 用训练好的模型做预测

**前提**：先跑过 `python train.py`（产生 `outputs/<date>/<time>/model.pt`）。以下代码在 `examples/structural_mechanics/truss_pinn/` 目录下运行。

```python
import sys, torch
sys.path.insert(0, "examples/structural_mechanics/truss_pinn")

from helpers.model import TrussPINN, encode_params
from helpers import geometry

# 加载 checkpoint
ckpt = torch.load("outputs/<date>/<time>/model.pt", weights_only=True, map_location="cpu")
geometry.P_MAX = float(ckpt["p_max"])          # 必须与训练时冻结值一致
model = TrussPINN()
model.load_state_dict(ckpt["state_dict"])
model.eval()

# 构造输入：载荷节点 n、方向 theta（弧度）、有符号幅值 P（牛顿）
n_idx = torch.tensor([1, 4])                    # 节点 1（左下内）与 4（左上）
theta = torch.tensor([-torch.pi / 2, 0.0])      # 向下 / 向右
P     = torch.tensor([-5000.0, 8000.0])         # 负=沿 theta 反方向
x = encode_params(n_idx, theta, P)              # (2, 8) float32

with torch.no_grad():
    u_hat, log_omega_hat, phi_hat = model(x)
# u_hat:       (2, 11)   — 11 个自由 DOF 的位移（米）
# log_omega_hat:(2, 3)    — 前 3 阶频率的自然对数
# phi_hat:     (2, 11, 3) — 前 3 阶振型（每列一个模态）

omega_hat = log_omega_hat.exp()                 # (2, 3) 频率 rad/s
```

### 输出张量的物理含义

- **自由 DOF 排序**：`geometry.FREE_DOFS` 给出 11 个自由自由度的全局 DOF 编号（14 个总 DOF 中去掉支座 0、1、7）。展开到完整 14 维位移向量：

  ```python
  u14 = torch.zeros(14, dtype=torch.float64)
  u14[list(geometry.FREE_DOFS)] = u_hat[0].to(torch.float64)
  # u14[0:2] = 节点 0 的 (ux, uy)，u14[2:4] = 节点 1，依此类推
  ```

- **载荷节点编码**：`encode_params` 把节点号映射为在 `geometry.LOADABLE_NODES`（`(1, 2, 4, 5, 6)`）中的**位置**做 one-hot——传入的是节点号，不是位置下标。
- **P 的符号**：正 P 沿 `theta` 方向，负 P 反向。`theta + π` 与 `(theta, -P)` 是同一物理载荷。
- **适用范围**：`|P| ≤ P_MAX`（checkpoint 中记录的冻结包络，默认 11871 N）。超出此范围稳定包络不保证，PINN 未在该区域训练。
- **精度预期**：位移 median 误差 ~0.28%，但 mid-gap theta（远离锚点方向）+ 小 |P| 工况可达 ~15%；频率各阶 median ~0.24–0.30%，最坏 0.87%。详见 `README.md` 的误差结构分析。

## 5. 测试

```bash
cd examples/structural_mechanics/truss_pinn
python -m pytest tests/ -v    # 40 项，覆盖几何/FEM/采样/模型/稳定性/符号残差
```

## 6. 常见问题

- **`ModuleNotFoundError: sympy`**：`pip install sympy>=1.12`（requirements.txt 已含；若跳过 `-r requirements.txt` 直接装会缺）。
- **训练 loss 不降**：检查 `weight_eq/sup/mode` 是否被意外覆盖为量级失衡的值；默认三权重经扫参平衡（见 README）。
- **`evaluate.py` 报 NA/不稳定**：checkpoint 的 `p_max` 与当前 `geometry.P_MAX` 不一致；`load_model` 会自动以 checkpoint 为准，勿混用不同冻结值的 checkpoint 与代码版本。
- **想换桁架拓扑/材料**：改 `helpers/geometry.py`（`NODES_XY`、`ELEMENTS`、`SUPPORT_DOFS`、材料常数），删除旧 `outputs/`，重跑 `train.py`（P_MAX 会按新几何重算）。振型监督的锚点网格在 `helpers/sampling.py`，与节点数联动。
