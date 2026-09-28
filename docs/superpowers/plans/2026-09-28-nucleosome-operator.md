# 实施计划：核小体定位算子案例

日期：2026-09-28
规格：`docs/superpowers/specs/2026-09-28-nucleosome-operator-design.md`
位置：`examples/bioinformatics/nucleosome_operator/`

## 交付物

```text
examples/bioinformatics/nucleosome_operator/
├── README.md            英文，仓库 README 规范（含 Getting Started / Additional Information / References）
├── USER_GUIDE.md        中文，面向使用者：安装、训练、评估、推理 API
├── RUNBOOK.md           中文，步骤化操作手册：每步命令 + 预期输出 + 故障判断
├── requirements.txt     运行依赖
├── conf/config.yaml     Hydra 配置（全部可命令行覆盖）
├── train.py             训练入口（自证门 → 语料 → 训练 → checkpoint）
├── evaluate.py          验收评估（门槛 + 分层诊断 + 3 张图 + error_report.json）
├── ablate_receptive_field.py  感受野消融（证明算子的非局部性）
├── helpers/
│   ├── genome.py        序列采样与编码
│   ├── energy.py        二核苷酸能量表 + 滑窗能量景观
│   ├── nucleosome.py    精确配分函数 DP（对数空间）+ 穷举/精确组合学 oracle
│   ├── data.py          语料 split、标准化、输入编码、相对 L2
│   ├── model.py         FNO 算子 + 损失
│   └── selftest.py      启动自证门
└── tests/               pytest 套件（生成器/景观/语料/模型/自证门）
```

## 任务与状态

| # | 任务 | 验收 | 状态 |
|---|------|------|------|
| T1 | 序列采样器（随机 + 植入基序两层） | 同种子复现；碱基码 ∈ [0,4)；植入行比例正确 | 完成 |
| T2 | 二核苷酸能量表 + 滑窗景观 | 列居中；与朴素窗口求和一致 | 完成 |
| T3 | 精确配分函数 DP（对数空间） | 穷举等价 + γ=0 精确组合学 | 完成 |
| T4 | 自证门（6 项） | 全过；失败抛 `SelfTestFailure` | 完成 |
| T5 | 语料 split + 标准化 + 输入编码 | 契约测试（形状/dtype/范围/复现） | 完成 |
| T6 | FNO 算子 + 相对 L2 损失 | 输出 ∈[0,1]；可微；长度可变 | 完成 |
| T7 | `train.py`（Hydra） | 冒烟可跑；checkpoint 含重建语料所需全部常量 | 完成 |
| T8 | `evaluate.py`（门槛 + 诊断 + 图） | held-out 600 例；VERDICT 与 exit code 一致 | 完成 |
| T9 | 感受野消融 | 同预算下局部 CNN 明显劣于算子 | 完成 |
| T10 | 三份文档 + `examples/README.md` 索引行 | markdownlint 通过（根配置 + examples 配置） | 完成 |

## 验证命令

```bash
cd examples/bioinformatics/nucleosome_operator
python -m pytest tests/ -q                 # 单元测试
python train.py                            # 训练（含启动自证门）
python evaluate.py                         # 验收（exit 0 = PASS）
python ablate_receptive_field.py           # 感受野消融
```

## 关键设计决策（记录）

1. **精确参考而非数值积分**：选配分函数转移矩阵（O(L·W) 精确）而不是动力学积分
   （TASEP 平均场之类）—— 参考必须可被穷举/big-int 精确验证，避免"参考本身有
   收敛误差"这类不可证伪的风险。代价：模型是热力学平衡的（见规格 §7 限制）。
2. **对数空间递推**：float64 在长位点（2048 bp，~14 个核小体）会溢出（~1e360），
   所以 `f`/`b` 全程 `logaddexp`；测试里用 4096 bp 位点钉住这一点。
3. **能量表构造**：逐列对 16 类去均值（消除全局堆积偏好）+ `scale=0.22` 标定
   能量尺度，使语料平均占据度 0.83、剖面反差 0.16→0.28 随 γ 单调 —— 让门槛
   指标（相对 L2）数值条件良好，而非饱和到 0/1。
4. **滑窗能量用 gather 累加而非 `conv1d`**：`conv1d` 的 im2col 对 (16, 146) 核
   放大 ~2300 倍，全语料下需要 ~33 GB 被 OOM kill（实测 exit 137）；逐偏移
   gather 只占 O(n·L)。
5. **输入 = one-hot 序列 + 能量景观 + γ**：`E` 是二核苷酸的窗式线性泛函，不可
   逐位置逆推，所以两者都喂给网络；γ 通道让单个算子覆盖弱/强序列依赖区间。
6. **输出加 logistic 头**：占据度是覆盖概率，sigmoid 让"输出永不出界"成为构造
   性质而不是学出来的性质。
7. **损失 = 逐样本相对 L2 平方**：与验收指标同一函数，训练直接优化所测。
