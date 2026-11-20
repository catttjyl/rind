# RIND-JEPA 主代码、93 街区数据读取与统一评测使用手册（2026-11-18）

本代码包汇总了 **RIND-JEPA 最新核心网络与训练入口**、`SceneCell-az24` 无在线光追空间格索引与 GPU 路由算子、内部参照基线源码、93 街区只读划分元数据，以及用于外部模型按同一口径打分的 `baseline_kit/` 评测工具。

## 一、核心开关说明：0-BVH 主算法 vs 在线 BVH 信息参照上限

同一套 `rind/` 主干源码通过 `--num-azimuth-bins` 与输入特征层级切换不同模式：

1. **`0-BVH` 正式主算法（`--num-azimuth-bins 24` + `SceneCell-az24`）**：
   - 将建筑三角面片按 24 个 $15^\circ$ 方位角扇区压缩为 $O(1)$ 场景级 `8×8` 空间格索引（约 `2.77 MiB`/街区），推理阶段零在线光线求交。
   - 配套特征层级：`Tier-G2`（`--model-type rind_full_geom`，完整几何特征）、`Tier-G1`（`--model-type rind_no_edge_margin`，去掉解析边距离）、`Tier-G0`（`--model-type rind_zero_margin`，仅保留三维原始三角顶点、法向面积与太阳向量）。
2. **在线 BVH 信息参照上限（`--num-azimuth-bins 0`）**：
   - 逐帧执行精确太阳方向候选筛查，仅用于标定“精确在线几何筛选下的性能上限”，不作为无在线光追部署主算法汇报。

## 二、目录结构

```text
RIND_G1_Code_and_Baseline_Kit/
├── README.md                              # 本使用手册
├── PORTABILITY.md                         # 环境版本锁定与便携分片协议说明
├── pyproject.toml                         # 包元数据（支持 pip install -e . --no-deps）
├── requirements-cpu-macos-py312.txt       # 依赖版本参考（PyTorch + NumPy）
├── rind/                                  # 核心主干包（models / objectives_jepa / fast_data / fast_train / metrics / split_contract / label_budget）
├── scripts/
│   ├── rind_phase5_cli.py                 # 最新训练/校准/评测统一 CLI 入口（支持 --reg-mode vicreg、--label-budget-fraction 等）
│   ├── rind_scene_cell_az24_gpu.py        # SceneCell-az24 O(1) 空间格建库与 GPU/CPU 在线 16+16 扇区路由实现
│   ├── test_scene_cell_az24_parity.py     # 100.0000% 数学等价与斜坡自遮挡剔除单元测试
│   ├── eval_repr_sufficiency.py           # 合成 32 三角双楼多遮挡与反事实编辑评测脚本
│   └── make_portable_index.py             # 93 街区只读分片哈希索引校验脚本
├── internal_baselines/
│   ├── scene_centric_dual.py              # Dual Static Voxel 4m/2m 稀疏体素参照基线
│   ├── dual_geometry.py                   # Dual 基线几何缓存工具
│   └── reference_pointnet_dsm25d_baselines.py # PointNet+SolarFiLM+GRU / 2.5D DSM+kNN-GRU / Canonical37 MLP 参照实现
├── baseline_kit/
│   ├── baseline_template.py               # 自定义模型接入模板（含 FP32 锁、1%/5%/100% 预算切分与标准导出）
│   └── evaluate_predictions.py            # 统一评测裁判脚本（12 校标定 + 14 验全套指标 + 4000 次配对 Bootstrap）
├── meta/
│   ├── split.json                         # 曼哈顿 4445 街区空间划分契约（只读）
│   ├── manifest.json                      # 93 街区清单元数据
│   ├── PORTABLE_INDEX.json                # 便携分片 SHA-256 校验索引
│   └── ray_query_omp.c                    # 可选 OpenMP C 射线核源码（仅在线参照组使用）
└── vendor/ray_query/                      # 可选原生射线查询适配器源码
```

## 三、开箱三步快速验证（无需下载 4.4 GB 数据集）

解压进入 `RIND_G1_Code_and_Baseline_Kit/` 后（建议执行 `pip install -e . --no-deps` 或在命令前加 `PYTHONPATH=.`），运行以下三条命令确认本地环境可用：

```bash
# 1. 验证主干包可独立导入
PYTHONPATH=. python3 -c "from rind import models, objectives_jepa; print('OK', models.__name__)"

# 2. 验证统一评测裁判脚本（合成多日序列与突变灵敏度自测）
python3 baseline_kit/evaluate_predictions.py --self-test

# 3. 验证外部模型从训练、导出 pred_probs_*.npy 到裁判打分的全链路（<5 秒）
python3 baseline_kit/baseline_template.py --synthetic-smoke
```

## 四、93 街区数据划分与分片字段说明（数据怎么读）

Google Drive `AI-Talent-WorldModel-93tile` 中的 `93` 个 `core_*.tar.gz` 压缩包（四季共 `T = 1,446` 帧、每街区分层采样 `N = 256` 接收点）对应如下固定空间划分：

- **`fit_67`（`67` 街区）**：唯一可用于梯度训练与归一化拟合的训练集。
- **`calibration`（`12` 街区）**：仅用于后训练标量温度缩放与决策阈值网格搜索（`[0.05, 0.95]`，181 格点）。
- **`val_spatial_clean`（`14` 街区，`5,119,233` 有效点帧）**：空间零重叠开发验证集（严禁参与训练或调参）。

单个分片解包或通过 `load_shard_dict` 读取后包含如下主要 `.npy` 数组：

| 字段名 | 形状与类型 | 物理与接口含义 |
| :--- | :--- | :--- |
| `day_ids` / `timestamps_unix_s` | `(T,)` `int32` / `(T,)` `int64` | 所属季节日编号与 `120s` 步长时间戳 |
| `sun_dirs` / `azimuth_bin_idx` | `(T, 3)` `float32` / `(T,)` `int16` | 单位太阳方向向量与对应的 `0..23` 方位角扇区索引 |
| `solar_altitude_deg` / `solar_azimuth_deg` | `(T,)` `float32` | 太阳高度角与方位角（度） |
| `receivers_xyz` / `receiver_rc` | `(N, 3)` `float32` / `(N, 2)` `int32` | 接收点三维坐标与格点行列坐标 |
| `local_features` | `(N, 16)` `float32` | 接收点局部表面与周边栅格高程统计特征 |
| `az_rel_triangles` | `(N, 24, 32, 3, 3)` `float32` | 各方位角扇区预筛 `Top-32` 候选三角形相对顶点坐标 |
| `az_mask` / `az_opaque` / `az_source_ids` | `(N, 24, 32)` `bool` / `bool` / `int64` | 有效候选掩码、不透明标记与原始三角面片 ID |
| `truth` / `valid_mask` | `(T, N)` `int8` / `(T, N)` `bool` | 日照真值（`1` 受照 / `0` 阴影 / `<0` 忽略）与有效掩码 |
| `boundary_distance_m` / `boundary_distance_valid` | `(T, N)` `float32` / `bool` | 接收点到最近阴影边界距离（米）与有效标记 |
| `scene_triangles_xyz` / `scene_opaque` / `scene_source_ids` | `(M_tri, 3, 3)` `float32` 等（仅 `.tar.gz` 内含） | 完整街区三角网格（供 `SceneCell-az24`、`Dual` 体素或点云网格基线使用） |

### 4.1 Python 读取划分与加载分片示例

```python
import sys
from pathlib import Path
sys.path.insert(0, "baseline_kit")

from evaluate_predictions import load_split_tile_ids
from baseline_template import load_shard_dict, subsample_receivers_for_budget

buckets = load_split_tile_ids(Path("meta/split.json"), Path("meta/manifest.json"))
# buckets["fit_67"] (67), buckets["calibration"] (12), buckets["val_spatial_clean"] (14)

shard = load_shard_dict(Path("/path/to/93tile_shards"), buckets["fit_67"][0])
keep_1pct = subsample_receivers_for_budget(buckets["fit_67"][0], shard, fraction=0.01, seed=20261112)
print(shard["truth"].shape, shard["az_rel_triangles"].shape, keep_1pct.sum())
```

## 五、预训练权重加载与主模型训练 / 校准 / 评测命令

### 5.1 加载云盘已标定权重（`calibrated.pt`）

```python
import torch
from rind.models import load_rind_checkpoint
from rind import fast_data

model, provenance = load_rind_checkpoint(
    "/path/to/P3_01_Arm_R0_stjepa_no_sigreg_BEST/calibrated.pt",
    device="cpu",
    require_calibrated=True,
)
model.eval()
print("Model:", provenance["model_type"], "Calibrated:", provenance.get("calibrated"))
```

### 5.2 运行主干训练、校准与评测（`scripts/rind_phase5_cli.py`）

将 Google Drive `AI-Talent-WorldModel-93tile` 的分片解压整理为只读缓存目录 `$CACHE_DIR`（需符合 `meta/PORTABLE_INDEX.json`，SHA-256 为 `f8e2b0334e6eb0787b74607e8a2ca9471d9f6754c76b06750f520d2ab8b71c3a`；若使用 `.tar.gz` 仅需自定义模型或阅读字段则直接用 `baseline_kit` 读取）。正式实验全程使用纯 `IEEE FP32`（`--disable-bf16`，禁止开启 `TF32`）：

- **训练示例 1：纯监督基准 `S0`（`az24 + Tier-G2`）**：
  ```bash
  PYTHONPATH=. python3 -B scripts/rind_phase5_cli.py train \
    --split-file meta/split.json --manifest-file meta/manifest.json \
    --cache-dir "$CACHE_DIR" --portable-index meta/PORTABLE_INDEX.json \
    --portable-index-sha256 f8e2b0334e6eb0787b74607e8a2ca9471d9f6754c76b06750f520d2ab8b71c3a \
    --diagnostic-mode --model-type rind_full_geom --num-azimuth-bins 24 --max-candidates 32 \
    --fit-subset fit_67 --pure-supervised --disable-bf16 --seed 20261114 \
    --checkpoint-out outputs/S0/raw.pt --output outputs/S0/train.json
  ```
- **训练示例 2：ST-JEPA `R0`（`az24 + Tier-G2, w_reg=0`）**：
  ```bash
  PYTHONPATH=. RIND_REG_WEIGHT=0.0 python3 -B scripts/rind_phase5_cli.py train \
    --split-file meta/split.json --manifest-file meta/manifest.json \
    --cache-dir "$CACHE_DIR" --portable-index meta/PORTABLE_INDEX.json \
    --portable-index-sha256 f8e2b0334e6eb0787b74607e8a2ca9471d9f6754c76b06750f520d2ab8b71c3a \
    --diagnostic-mode --model-type rind_full_geom --num-azimuth-bins 24 --max-candidates 32 \
    --fit-subset fit_67 --jepa-mode ema_geom --mask-mode spatial_boundary \
    --jepa-context-mode with_x --reg-mode sigreg --dominant-occluder-dropout-prob 0.25 \
    --counterfactual-weight 0.08 --disable-bf16 --seed 20261114 \
    --checkpoint-out outputs/R0/raw.pt --output outputs/R0/train.json
  ```
- **训练示例 3：JEPA-VICReg（`--reg-mode vicreg`）与纯原始顶点 `Tier-G0` `1%` 冷启动**：
  ```bash
  # 仅在 Tier-G2 上将 --reg-mode 换为 vicreg 即可运行 VICReg；
  # 若跑纯原始三维顶点 Tier-G0 + 1% 冷启动，改用 --model-type rind_zero_margin：
  PYTHONPATH=. RIND_BOUNDARY_METRIC_WEIGHT=0.0 RIND_CAL_LABEL_BUDGET_FRACTION=0.01 \
  python3 -B scripts/rind_phase5_cli.py train \
    --split-file meta/split.json --manifest-file meta/manifest.json \
    --cache-dir "$CACHE_DIR" --portable-index meta/PORTABLE_INDEX.json \
    --portable-index-sha256 f8e2b0334e6eb0787b74607e8a2ca9471d9f6754c76b06750f520d2ab8b71c3a \
    --diagnostic-mode --model-type rind_zero_margin --num-azimuth-bins 24 --max-candidates 32 \
    --fit-subset fit_67 --jepa-mode ema_geom --jepa-context-mode with_x \
    --reg-mode vicreg --label-budget-fraction 0.01 --label-budget-seed 20261112 \
    --disable-bf16 --seed 20261114 \
    --checkpoint-out outputs/G0_1pct/raw.pt --output outputs/G0_1pct/train.json
  ```
- **校准（`calibrate`）与验证（`eval`）**：
  ```bash
  PYTHONPATH=. python3 -B scripts/rind_phase5_cli.py calibrate \
    --split-file meta/split.json --manifest-file meta/manifest.json \
    --cache-dir "$CACHE_DIR" --portable-index meta/PORTABLE_INDEX.json \
    --portable-index-sha256 f8e2b0334e6eb0787b74607e8a2ca9471d9f6754c76b06750f520d2ab8b71c3a \
    --checkpoint outputs/R0/raw.pt --checkpoint-out outputs/R0/calibrated.pt \
    --output outputs/R0/calibrate.json

  PYTHONPATH=. python3 -B scripts/rind_phase5_cli.py eval \
    --split-file meta/split.json --manifest-file meta/manifest.json \
    --cache-dir "$CACHE_DIR" --portable-index meta/PORTABLE_INDEX.json \
    --portable-index-sha256 f8e2b0334e6eb0787b74607e8a2ca9471d9f6754c76b06750f520d2ab8b71c3a \
    --checkpoint outputs/R0/calibrated.pt --bucket val_spatial_clean \
    --no-blocker-distance-buckets --no-shadow-contours --output outputs/R0/eval.json
  ```

## 六、自定义模型 / 对比算法统一接入与打分流程（`baseline_kit/`）

无需修改 `rind/` 核心源码，按下面三步即可用完全一致的口径跑出与主模型可比的结果：

1. **参照 `baseline_kit/baseline_template.py`**：
   - 脚本开头保留纯 `IEEE FP32` 锁（`allow_tf32 = False`）。
   - 用 `load_split_tile_ids` + `load_shard_dict` 只读加载 `fit_67`（训练）和 `calibration`（选模），严禁用 `val_spatial_clean` 调参。
2. **导出统一格式结果**：
   - 为 `calibration`（12 块）与 `val_spatial_clean`（14 块）逐街区导出 `pred_probs_{tile_id}.npy`（`float32`，形状 `[T, N]`）及耗时统计 `perf_receipt.json`。
3. **运行统一裁判脚本出表**：
   ```bash
   python3 baseline_kit/evaluate_predictions.py \
     --data-dir /path/to/93tile_shards \
     --cal-preds-dir outputs/my_baseline \
     --val-preds-dir outputs/my_baseline \
     --output-json outputs/my_baseline/eval_receipt.json
   ```
   若需对比参照模型的 4,000 次配对 Bootstrap 95% 置信区间，追加 `--ref-probs-dir /path/to/ref_probs` 即可。
