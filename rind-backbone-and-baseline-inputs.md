# RIND 学生 backbone 与 baseline 输入公平性设计

本文先说明当前 RIND 主模型的结构、每一步接收什么数据、怎样计算 loss（第 1–4 节，依据下文链接的 [rind/](rind/) 源码整理），再说明对比实验中各 baseline 怎样获得与 RIND 对等的信息（第 5–10 节，属于实验设计，标“暂定”的参数尚未确定）。第 11 节汇总各模型共用与各自的部分，以及散在正文中的待实现、待定和待核实事项。本文不报告任何训练结果。

---

## 1. 范围与源码身份

| 项目             | 内容                                                                                                                                                                                                                                                                          |
| ---------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 对象             | [rind/](rind/)，package version `2.2.0rc5`，benchmark contract `MANHATTAN_4445_ANNUAL_18808_V1`，feature schema `canonical37_v2`                                                                                                                                              |
| 主方法           | `--model-type rind_full_geom`（Tier-G2）+ `--num-azimuth-bins 24`（0-BVH，az24 候选库）+ `--max-candidates 32`。**对比实验中的 RIND 一律指这个配置**                                                                                                                          |
| objective        | 主方法使用 ST-JEPA 目标（README 中的 R0：监督 loss + JEPA 辅助项，见 4.2）。objective 层的对照组目前只做 **direct supervised**（只有监督 loss，见第 7 节）；Geometry-MAE、Geometry-JEPA 暂不做                                                                                |
| 宽度             | `hidden_dim = 96`，4 个注意力头                                                                                                                                                                                                                                               |
| 参数量           | 内部基线文档记为约 263.8K（[reference_pointnet_dsm25d_baselines.py](internal_baselines/reference_pointnet_dsm25d_baselines.py)）                                                                                                                                              |
| 数据划分         | 93 个 tile：`fit_67` 训练；`calibration`（12 个）只用于拟合标量温度和决策阈值；`val_spatial_clean`（14 个）开发验证，不得参与训练或调参（[split.json](meta/split.json)）                                                                                                      |
| 主方法以外的选项 | ① 在线 BVH（`--num-azimuth-bins 0`）：逐帧按真实太阳方向做精确筛选，只作“精确在线筛选”的上限参照；② RIND 自身的特征消融 `rind_no_edge_margin`（G1）、`rind_zero_margin`（G0）、`rind_coarse_aabb`；③ 内部参照基线 `local_convgru_grid`、`pointwise_mlp`、`canonical37_mlp` 等 |

坐标约定：单位为米和秒，**X = 东、Y = 上、Z = 北**；太阳方位角按 `atan2(s_x, s_z)` 从北向东计。

---

## 2. 输入数据

### 2.1 每个 tile 的数据

| 字段                                                           | 形状与类型                               | 内容                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| -------------------------------------------------------------- | ---------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `receivers_xyz` / `receiver_rc`                                | `(N,3)` float32 / `(N,2)`                | 接收点世界坐标，以及在 512×512 网格上的行列号。**N = 256**：从静态可用点中按 tile 哈希偏移、等步长确定性抽取，训练和评测使用同一批点                                                                                                                                                                                                                                                                                                                                   |
| `receiver_obs_mask`                                            | `(N,)` bool                              | 由数据加载器生成，等于 `geometry_known_hw` 在接收点处的值；不读取标签                                                                                                                                                                                                                                                                                                                                                                                                  |
| `static_fields_8hw` + `geometry_known_hw`                      | `(8,512,512)` float32 + `(512,512)` bool | 来自源数据 `MODEL_GEOMETRY.npz` 的 `fields` 和 `geometry_known`；tile 约 100 m，格距 0.1953 m。<br>0 `terrain_world_y_m` = (地形 − ref)/100<br>1 `building_top_world_y_m` = (建筑顶面 − ref)/100<br>2 `building_height_m` = 建筑高度/100<br>3 `terrain_valid`<br>4 `building_top_valid`<br>5 `building_height_valid`（= terrain_valid & top_valid & h ≥ 0）<br>6 `unknown_geometry_mask`<br>7 `geometry_coverage_mask`<br>（ref = 有效地形高度的中位数，无效处填 0。） |
| `local_features`                                               | `(N,16)` float32                         | 由 `static_fields_8hw` 派生：8 个通道在接收点像素处的值，加上以该点为中心的 9×9 窗口（约 1.76 m）均值。**RIND 主模型只通过它读取 raster**                                                                                                                                                                                                                                                                                                                              |
| `sun_dirs` / `azimuth_bin_idx`                                 | `(T,3)` float32 / `(T,)` int16           | 单位太阳向量；方位扇区号 = ⌊方位角 / 15°⌋ mod 24                                                                                                                                                                                                                                                                                                                                                                                                                       |
| `solar_altitude_deg` / `solar_azimuth_deg`                     | `(T,)` float32                           | 太阳高度角与方位角                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| `day_ids` / `timestamps_unix_s`                                | `(T,)`                                   | 帧间隔 120 s，每天从日出到日落。四季包取第 0、13、26、39 周各一天，T = 1,446；全年协议为 52 周、18,808 帧                                                                                                                                                                                                                                                                                                                                                              |
| `scene_triangles_xyz` / `scene_opaque` / `scene_source_ids`    | `(M,3,3)` float32 等                     | 完整三角网格（**mesh**），范围为接收点包围盒外扩 `bbox_margin_m = 4000 m`                                                                                                                                                                                                                                                                                                                                                                                              |
| `receiver_source_triangle_ids`                                 | `(N,)`                                   | 接收点所在表面的三角形，构建候选时排除                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| `az_rel_triangles` / `az_opaque` / `az_mask` / `az_source_ids` | `(N,24,32,3,3)` float32 等               | az24 候选库，见 2.2                                                                                                                                                                                                                                                                                                                                                                                                                                                    |

### 2.2 az24 候选库（0-BVH）

由 `select_azimuth_sector_candidates`（[fullday_data.py:521](rind/fullday_data.py#L521)）对每个接收点离线构建，**不看标签，也不看太阳高度和帧**：

1. **初筛**：三角形不透明；顶部高于接收点（y_max > q_y + 1 mm）；水平距离减去三角形半径 ≤ 4000 m；不是接收点自身所在的面；三角形的方位弧（3 个顶点方位的最小覆盖弧）与该扇区重叠。包含接收点的三角形对所有扇区都合格。
2. **每个扇区 32 个槽**：
   - 16 个槽：按仰角上界 (y_max − q_y) / max(d − r, 0.25 m) 取前 16；
   - 16 个槽：按距离环 [0, 100)、[100, 400)、[400, 800)、[800, 1600)、≥ 1600 m 分组，打分为仰角上界 × |法向 · 方向|，从最远的环开始轮流取；
   - 不足 32 个时，空槽的 mask 为 False。
3. 存储三角形相对接收点的顶点坐标（float32）。

每帧用 `azimuth_bin_idx` 取出对应扇区的 32 个三角形，再结合该帧的真实太阳向量计算 37 维 canonical token（`build_canonical37_torch`，[fast_data.py:267](rind/fast_data.py#L267)）：

| 槽位  | 内容                                                                      |
| ----- | ------------------------------------------------------------------------- |
| 0–2   | 三个顶点沿太阳方向的距离（有符号，log 缩放）                              |
| 3–5   | 三个顶点到太阳射线的径向距离（log）                                       |
| 6–8   | 三条边长（log）                                                           |
| 9     | 入射角（投影面积 / 面积）                                                 |
| 10    | 不透明                                                                    |
| 11    | log 面积                                                                  |
| 12    | sun_y                                                                     |
| 13–21 | 相对顶点（三个顶点按 xyz 字典序排序，除以 1000 m）                        |
| 22–24 | 太阳投影边距：沿太阳方向投影后，射线到三条边的有符号距离，排序后 log 缩放 |
| 25–27 | sun                                                                       |
| 28–33 | bbox 的 min 和 max（除以 1000 m）                                         |
| 34    | 面积（/10⁶）                                                              |
| 35–36 | `projection_degenerate`、`degenerate`                                     |

22–24 槽的三个边距同号时，射线就穿过该三角形的投影，所以这三个槽几乎直接给出了解析相交的答案。

**az24 的局限**：

| 局限         | 说明                                                                                                                                                                          |
| ------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 只覆盖采样点 | 候选库只为每个 tile 的 256 个接收点预先算好；要在其他点或整张 512² 图上预测，需要现算（例如用 SceneCell 索引）                                                                |
| 容量与召回   | 每个 15° 扇区只保留 32 个，排序不看太阳高度。真正挡光的面可能被挤出前 32。召回率可以用 `compute_blocker_distance_stratified_metrics` 的 `candidate_first_blocker_recall` 测量 |
| 方位量化     | 候选来自整个扇区，最多偏离真实射线 ±7.5°，模型需要自己排除不在射线上的面；太阳跨过扇区边界时，候选集会整体跳变                                                                |
| 手工先验     | 排序规则（仰角上界、距离环、朝向）是人为设计的，本身就是很强的归纳偏置，也是 RIND 方法的一部分                                                                                |
| 范围截断     | 4 km 以外没有任何几何信息                                                                                                                                                     |
| 静态         | 几何改变后必须重算                                                                                                                                                            |

### 2.3 只用于训练的标签

| 标签                                              | 形状         | 内容                                                                                                                                                                                                                                                   |
| ------------------------------------------------- | ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `truth`                                           | `(T,N)` int8 | 1 = 亮，0 = 阴影，−1 = 未知                                                                                                                                                                                                                            |
| `valid_mask`                                      | `(T,N)` bool | 该点该帧是否有已知答案                                                                                                                                                                                                                                 |
| `boundary_distance_m` / `boundary_distance_valid` | `(T,N)`      | 在源 512² 状态图上，以接收点为中心、半径 2.25 m 内搜索最近的相反状态像素，取平面距离；找不到时记为 2.25 m。只有当最近的未知像素比 min(该距离, 2.0 m) 更远时才有效。距离无符号，符号由 `truth` 给出（[fullday_data.py:866](rind/fullday_data.py#L866)） |
| `response_5x5_bitpack`                            |              | 接收点 5×5 邻域的状态位图，只供 JEPA teacher 使用，direct supervised 不用                                                                                                                                                                              |

**监督只发生在这 N = 256 个点上。** 源数据里有整张 512² 状态图，但它不进入训练标签。

---

## 3. 主模型结构

**最新代码删除了 CNN 结构，far context box 和 region graph 也已删除。** 8 通道 raster 只以 `local_features`（16 维）的形式进入；mesh 只以 az24 候选的形式进入，远处的几何信息全部经由 az24 候选库进入，范围到 4 km。代码中不再使用 g / a / u 的命名。

### 3.1 结构图

```text
════ 离线（不可训练，不看太阳、不看标签）════
scene_triangles（4 km）──► az24 候选库 (N,24,32,3,3)
receivers_xyz ──► tile 内 256 个点两两 3D 距离 ──► 每点最近 8 个邻居（缓存）

════ 太阳编码 ════
sun_t ──► [s 3, 切平面坐标 2, 切向基 e1 e2 6, 实球谐 l≤4 24, Fourier 24]（59 维）
      ──► MLP ──► sun_emb_t (96)

════ 静态几何记忆 StaticGeometryMemory（≈ 旧 g）════
local16 ──► RunningNormalizer ──► node_stem MLP(16→96→96) ──► h0
邻居消息：edge_msg([h0_i, h0_j, Δxyz/20, log1p(d)/4]) × sigmoid 门，只用 d < 50 m 且已观测的邻居
Z_G = LayerNorm(h0 + 加权均值)                                 (N,96)，每个 tile 计算一次
Solar-FiLM：g_t = (1 + tanh γ(sun_emb_t)) · Z_G + β(sun_emb_t)       (T,N,96)

════ 候选分支（≈ 旧 a）════
每帧取扇区 bin(sun_t) 的 32 个三角形 ──► 37 维 token ──► RunningNormalizer
 ──► FourierSphereTangentEncoder102：37 维 + 65 维派生量
     （太阳切平面坐标系下的 9 个顶点坐标、质心与离轴距离 4、弦协方差 6、
      屋顶/立面线索 3、边长与紧致度 5、相位 6、Fourier 32）
     102 维 ──► MLP ──► 每个候选 96 维
 ──► 10 维射线线索 ──► 每个注意力头一个 logit 偏置（有界 ±4）
 ──► 4 头 cross-attention：query = Linear([g_t, sun_emb_t])，key/value = 32 个候选 + 1 个空槽
 ──► a_t (96)

════ 状态（≈ 旧 u）════
h_t = step_fusion([g_t, sun_emb_t, a_t])                      (288 → 96)
z_direct_t = tanh(init_proj(LN h_t))                          每帧独立
z_roll_t   = (1 − k)·z_{t−1} + k·tanh(proposal(LN h_t))，
             k = σ(gate([LN h_t, z_{t−1}, 太阳运动学 2 维]))
             跨天、相邻帧未同时观测、或到达 reanchor 步时，改用 z_direct_t

════ Readout ════
logit_bce      = bce_head(z) + 候选 LSE 打分（默认 cand_readout = attn_only，此项为 0）
logit_boundary = boundary_head(z)                             只用于 metric boundary loss
logit_fused    = logit_bce + 门 × 残差                        FocalEventTransitionGate，输入全部 detach：
                 z、Δz、不确定度、boundary logit、|Δz|、太阳角速度、局部起伏
logit          = logit_fused + kNN(4) 局部读出残差            邻居 < 15 m；输入 z、local16 邻域对比、起伏
logit_direct   = 同样的读出作用在 z_direct 上                 用于 loss
p = sigmoid(logit / T_cal)；决策阈值 th_cal 在 calibration 上搜索
```

推理模式：`static_direct`（每帧只用 z_direct）、`causal_rollout`（因果递推，可隔帧锚定）、`offline_interp`（非因果双侧锚定插值）。

源码位置：[models.py](rind/models.py) 中的 `RindCandidateWorldModel`、`StaticGeometryMemory`、`FourierSphereTangentEncoder102`、`RayAnisotropicAttentionBias`、`BoundedSphereTangentCell`、`FocalEventTransitionGate`、`LocalValidNeighborhoodReadout`；[fast_data.py](rind/fast_data.py) 中的 `build_canonical37_torch`、`materialize_gpu_tokens_from_azimuth_bank`；[fullday_data.py](rind/fullday_data.py) 中的 `select_azimuth_sector_candidates`、`extract_receiver_local_features_and_crops`。

### 3.2 信息进入的位置

| 信息          | 进入位置                                                            | 静态几何记忆 | 候选分支                          | 状态 z          | 读出头直接使用                  |
| ------------- | ------------------------------------------------------------------- | ------------ | --------------------------------- | --------------- | ------------------------------- |
| 8 通道 raster | 只经 local16（接收点像素 + 9×9 均值）                               | ✔            | 间接（经 query 决定关注哪个候选） | ✔               | ✔（kNN 读出、事件门的局部起伏） |
| receiver xyz  | kNN 图的边特征；候选顶点都相对于它                                  | ✔（被学习）  | 隐含在相对顶点中                  | —               | ✔（kNN 距离）                   |
| mesh          | 只经 az24 候选                                                      | ✘            | ✔                                 | ✔（经候选分支） | ✘                               |
| sun           | 编码后进入 FiLM、query、step_fusion；token 中的太阳相关槽位；递推门 | ✔（FiLM）    | ✔                                 | ✔               | ✔（事件门）                     |
| 时间          | `same_day_adjacent_mask`、太阳运动学                                | —            | —                                 | ✔（递推与重置） | ✔（事件门）                     |

整张 512² raster 只有内部基线 `local_convgru_grid` 使用：它以每个接收点为中心切 21×21 或 64×64 的 crop，通道为 8 个 raster 通道加 1 个 `geometry_known` 掩码，共 9 通道。

### 3.3 与旧版（candidate.8）的主要差别

- 多尺度 CNN 和 region graph 都没有了，静态几何改为 local16 加接收点之间的 kNN 消息传递；
- far context box 已删除；
- 在线 BVH 逐帧查询改为离线、与太阳无关的 az24 候选库，每个扇区 32 个候选（旧版最多 1024 个）；
- 候选池化从 max + attention 改为由射线偏置的多头 cross-attention，query 来自静态几何记忆和太阳；
- 不再有拒绝预测（旧版 reason 1–5）；
- 不再有 teacher 三阶段训练。

---

## 4. Loss

RIND 主方法的总 loss = **监督部分**（4.1）+ **JEPA 辅助部分**（4.2）

### 4.1 监督部分（所有配置共有）

`compute_fullday_loss`（[fast_train.py:80](rind/fast_train.py#L80)），权重取自 `ObjectiveConfig`：

```text
有效点 = valid_mask & truth ≥ 0，y = truth
L_state = [ (BCE(logit, y) + 0.25·BCE(logit_bce, y)) / 1.25  +  BCE(logit_direct, y) ] / 2
                             └───────── aux BCE ─────────┘

L = L_state
  + 0.35 · L_switch     相邻同日帧的切换概率 p_t(1−p_{t−1}) + (1−p_t)p_{t−1} 对真实切换的 focal BCE（γ = 2），
                        亮→暗、暗→亮两个方向平衡，另加困难稳定负样本
                        （具体组合见 compute_focal_event_hard_negative_loss）
  + 0.25 · L_boundary   作用在 logit_boundary 上，预测距离 = 0.75 · logit（米）：
                        exp(−d/0.5) 加权的符号 BCE
                        + 0.6 · SmoothL1（d ≤ 0.5 m）
                        + 0.25 · 远端软目标（d > 0.5 m）
                        + 0.25 · 最近邻空间差分（邻居 ≤ 2.5 m）
                        + 0.25 · 相邻帧时间差分
  + 0.20 · L_hard_sun   “亮且有候选”的点 BCE 加权 ×1.5；
                        并要求它们的平均 logit 比“暗且有候选”的点高出 margin 1（权重 0.15）
```

- 在 N = 256、tile 边长 100 m 时，相邻接收点平均相距约 6 m，`L_boundary` 中的空间差分项（要求邻居 ≤ 2.5 m）多数时候不起作用。这是推断，没有统计过。
- 模型没有事件门时（`has_event_gate = False`），aux BCE 和 `L_boundary` 都会被跳过；`L_hard_sun` 依赖 `candidate_mask`，没有候选时也为 0。这影响 baseline 能拿到的 loss，处理方式见 5.5。

### 4.2 JEPA 辅助部分（主方法 R0）

README 中 R0 的配置为 `--jepa-mode ema_geom --mask-mode spatial_boundary --jepa-context-mode with_x --reg-mode sigreg --dominant-occluder-dropout-prob 0.25 --counterfactual-weight 0.08`，并设环境变量 `RIND_REG_WEIGHT=0.0`。其余权重取 `ObjectiveConfig` 默认值（可用 `RIND_*` 环境变量覆盖）。辅助项由 `RindJepaAuxiliaryModule.forward_objectives`（[objectives_jepa.py:2419](rind/objectives_jepa.py#L2419)）和反事实项（[fast_train.py:1044](rind/fast_train.py#L1044)）组成：

```text
L_total = L_sup（4.1）
  + 0.25 · L_pred       多步潜变量预测：
                        teacher 目标 z*_t = tanh(EMA_proj([z_direct_t, sun_emb_t]))，
                        EMA_proj 是 online_proj 的 EMA 拷贝（衰减 0.996；ema_target_scope = head，
                        即 teacher 的输入就是学生自己的 z_direct）
                        目标集合 Ω_T：spatial_boundary 遮挡，比例 0.25，按 boundary logit 选点（不读标签）
                        上下文 Ω_C：Ω_T 以外、相距 ≥ 1.5 m 的接收点
                        预测 = tanh(MLP([z_direct_t, ctx_t, sun_emb_{t+k}, lead 编码]))，
                        ctx_t = 跨接收点 attention，从 Ω_C 上的 z_direct 聚合
                        在 lead k ∈ {1, 2, 5, 10, 15}、整段同日且有效的帧上，对 z*_{t+k} 求 L1，只在 Ω_T 上计算
  + 0.15 · L_align      L1(online_proj 输出, z_roll.detach())
  + 0.20 · L_trans      相邻帧的 Δz_roll 对 Δz*，按 1 + 4·|Δp| 加权（p 为模型自己的概率，不读标签）
  + 0.05 · L_dual       SmoothL1(z_roll, z_direct.detach()) + SmoothL1(logit_bce, logit_direct.detach())
  + 0 · L_reg           SIGReg，R0 中权重被设为 0
  + 0.08 · L_cf         每 2 步一次，在候选三角形上做局部干预：
                        ① 把一个或全部候选抬高或降低 6 m，用精确射线–三角形求交（Möller–Trumbore）
                           判断该点是否翻转；翻转的点回归概率变化，没翻转的点要求不变
                        ② dominant-occluder dropout（p = 0.25）：删掉最前面的遮挡面后，亮的概率不能下降；
                           如果剩下的候选仍然精确挡住射线，必须预测为阴影（BCE，权重 0.5）
```

几点说明：

- R0 的所有辅助项都**不读标签**，但 `L_cf` 用候选三角形的**解析射线求交**来生成目标，相当于额外的物理监督。对比时需要写明，baseline 没有这项。
- **系数不等于实际占比**：
  - 代码中总 loss 是 `loss = sup_loss + aux_loss + cf_term`（[fast_train.py:1103](rind/fast_train.py#L1103)），其中 `L_sup` 是四项直接相加、没有归一化（[fast_train.py:176-181](rind/fast_train.py#L176-L181)）。
  - 因此按项计，监督四项系数和为 1 + 0.35 + 0.25 + 0.20 = **1.80**，辅助五项系数和为 0.25 + 0.15 + 0.20 + 0.05 + 0.08 = **0.73**。
  - 即使按项计，系数之比也不是实际占比。各项 loss 量级不同，`L_cf` 每 2 步才算一次。实际占比应以训练日志中各项的数值（系数 × 实测 loss）为准，最好统计各项对 backbone 的梯度范数。日志字段：`compute_fullday_loss` 返回 `state_bce_loss`、`transition_focal_loss` 等，辅助项见 `aux_stats`。
- **`L_align` 不更新学生 backbone**：它是 `L1(online_proj 输出, z_roll.detach())`（[objectives_jepa.py:2647](rind/objectives_jepa.py#L2647)），学生一侧做了 detach，只训练 teacher 的投影头。所以这 0.15 对学生表征没有直接作用。
- 在 `spatial_boundary` 模式下，目标点自己的 z_direct_t 仍作为预测器的输入（代码注释：“keeps the deployed geometry query available”），遮挡只限制来自其他接收点的上下文。
- 以上是按代码整理的；各项的数值细节以源码为准，例如 `L_pred` 的 history 衔接、`L_cf` 的子组采样。

### 4.3 校准

训练后，在 calibration 的 12 个 tile 上拟合标量温度（最小化 NLL），再在 [0.05, 0.95] 的 181 个阈值中选 (accuracy + F1) / 2 最大的一个（统一裁判脚本 [evaluate_predictions.py](baseline_kit/evaluate_predictions.py) 的做法）。

---

## 5. 统一的信息形式

对比实验分两套设置（详见第 6 节）：

- **主实验**：几何以各模型能接收的原生形式给出，覆盖范围统一对齐到 4 km；RIND 照常使用 az24 候选，把它当作 RIND 方法的一部分。
- **补充对照**：所有模型都额外拿到 az24 候选，与 RIND G2 使用的 37 维 token 完全相同。

### 5.1 每个 tile 的统一数据包

| 组件               | 内容                                                                                    | 来源                                                                                                                             | 主实验                   | 补充对照 |
| ------------------ | --------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- | ------------------------ | -------- |
| **Q** 查询         | `receivers_xyz`、`receiver_rc`、`receiver_obs_mask`                                     | shard                                                                                                                            | 全部模型                 | 全部模型 |
| **S** 太阳与时间   | `sun_dirs`、`azimuth_bin_idx`、`day_ids`、`timestamps_unix_s`、`same_day_adjacent_mask` | shard                                                                                                                            | 全部模型                 | 全部模型 |
| **L** 局部特征     | `local_features (N,16)`：8 通道在接收点像素处的值 + 9×9 窗口均值                        | shard（由 H0 派生）                                                                                                              | 全部模型（见下）         | 同左     |
| **H** 高度图金字塔 | 见 5.2                                                                                  | H0 即 `static_fields_8hw`；H1、H2 由 [build_geometry_inputs.py](baseline_kit/build_geometry_inputs.py) 从 `scene_triangles` 生成 | 不能直接接收 mesh 的模型 | 同左     |
| **P** LOD 点云     | 见 5.3                                                                                  | 由 [build_geometry_inputs.py](baseline_kit/build_geometry_inputs.py) 从 `scene_triangles` 生成                                   | 能接收点或 mesh 的模型   | 同左     |
| **C** az24 候选    | `(N,24,32,3,3)` + mask + opaque，每帧转为 37 维 token                                   | `select_azimuth_sector_candidates`                                                                                               | **只有 RIND**            | 全部模型 |
| **Y** 标签         | `truth`、`valid_mask`、`boundary_distance_m`                                            | shard                                                                                                                            | 只在 N 个点上            | 同左     |

### 5.2 高度图金字塔 H

| 层  | 覆盖范围               | 格距     | 尺寸 | 通道                                                                           |
| --- | ---------------------- | -------- | ---- | ------------------------------------------------------------------------------ |
| H0  | 100 m（tile 本身）     | 0.195 m  | 512² | 现有 8 通道 + `geometry_known`，共 9 通道                                      |
| H1  | 800 m                  | 1.5625 m | 512² | 由 `scene_triangles` 栅格化：每格最高表面高度、最高建筑高度、覆盖率 / 有效标志 |
| H2  | 8 km（覆盖 4 km 半径） | 15.625 m | 512² | 同 H1                                                                          |

- 三层都以 tile 中心对齐，高度使用同一基准（需要先确认 H0 的高度基准），归一化全局固定，不按 tile 单独归一化。
- 粗层每格取**最大值**而不是均值，避免把遮挡物削平。
- **会丢失的信息**：悬挑、骑楼等多层结构（2.5D 的限制），以及远处的平面细节。丢失多少用“高度图 oracle”量化（第 6 节）。
- **层与层之间的分辨率跳变**：现在每层放大 8 倍。如果想让中距离更细，可以改成每层约 4 倍，比如 100 m、400 m、1.6 km、8 km 四层。 （根据 oracle 量化结果调整）
- 模型不知道每层的尺度，会把每层都当成一张普通图片，所以要给各层特征加上层编码

#### 伪 RGB 映射（供使用 RGB 预训练权重的模型使用，即 DINOv3）

金字塔的每一层各自映射成一张 3 通道图，encoder 对每层单独做一次前向。

**8 个通道的独立信息其实很少。** 按 2.1，8 个通道是 3 个连续场（地形、建筑顶面、建筑高度）加 5 个 0/1 掩码。其中：

- 建筑高度 = 顶面 − 地形；
- `building_height_valid` = `terrain_valid & top_valid & h ≥ 0`。

去掉可推导的部分，独立内容约为 2 个连续场加几个有效性掩码，3 个通道基本装得下：

| 层     | R                                                                 | G                          | B                                                                |
| ------ | ----------------------------------------------------------------- | -------------------------- | ---------------------------------------------------------------- |
| H0     | 表面高度 s：`building_top_valid` 处取顶面，否则取地形（相对 ref） | 建筑高度 h（非建筑处为 0） | 有效性：`geometry_known & geometry_coverage_mask` 为 1，否则为 0 |
| H1、H2 | 每格最高表面高度                                                  | 每格最高建筑高度           | 覆盖率 / 有效标志                                                |

- **H0 丢掉的**：建筑下的地形可由 s − h 恢复；三个 valid 掩码合成一个后，“地形无效”与“顶面无效”的区别只在数据缺口处起作用。几何内容几乎不丢。
- **H1、H2 本来就是 3 通道**，直接对应 RGB。
- **归一化**：全局固定、所有 encoder 共用同一组常数。高度按固定区间（暂定 −20 m 到 200 m）线性映射到 [0, 1]，再按各 encoder 自己的预处理均值和方差标准化。不按 tile 单独归一化。
- **不使用按真实太阳方向渲染的 hillshade**，否则相当于在输入端替模型做了一部分光照计算，太阳信息也会以这种形式进入输入。B 换成固定光源的 hillshade（看起来更像自然图像）可以作为敏感性实验。
- **量化映射本身的损失**：用同一个从零训练的小模型（例如第 9 节的 Conditional U-Net），在相同预算下分别输入 9 通道 H0 和 3 通道伪 RGB，两者的差距就是映射丢掉的信息。只需多跑一次训练。

**真正的损失不在通道数，而在下面三处：**

1. 2.5D 的限制（见上，用高度图 oracle 量化）；
2. encoder 的分辨率：patch 16 时，H0 上每个 token 约 3.1 m；
3. 域差异：高度图不是自然图像。

第 2、3 点是使用 RGB 预训练模型时固有的限制，需如实报告。

### 5.3 LOD 点云 P

- 只在不透明面（屋顶和立面）上按面积采样，地面不采；
- 点距 s(d) = max(s₀, d · Δθ)，d 为到 tile 中心的水平距离。这样从 tile 看出去，远近物体的角分辨率相同，点数随半径只按对数增长。暂定 s₀ = 0.5 m、Δθ = 0.5°，粗估为几十万点量级，需要实测；
- 点特征：相对 tile 中心的 xyz、法向、是否屋顶、不透明；
- 所有点云类模型使用同一份采样结果（固定种子）。

### 5.4 各类模型的接入方式

| 类别                                                                 | 几何（主实验）                                                                   | 太阳                        | 补充对照中的候选 C                                                                                                                                    | 读出                          |
| -------------------------------------------------------------------- | -------------------------------------------------------------------------------- | --------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------- |
| 预训练 2D（DINOv3）                                                  | H 转伪 RGB（5.2），用预训练权重初始化后整体微调                                  | 太阳条件 token              | 有 attention 的架构把候选当 token，由查询点 cross-attend；没有集合机制的架构在查询头接一个通用 set encoder（DeepSets，max + mean 池化），并在文中注明 | 统一查询头（第 9 节）         |
| 2D dense end-to-end（Conditional U-Net、SegMAN、vanilla V-JEPA 2.1） | H0 作主输入；H1、H2 按各架构的多尺度方式融合（如各经小编码器在 bottleneck 拼接） | FiLM 或条件 token           | 同上                                                                                                                                                  | 同上                          |
| 点 / 算子（GINO；Sonata 用预训练权重初始化）                         | P + 查询点上的 local16（或把 H0 转为点）                                         | 拼接到 latent 通道或做 FiLM | 作为查询点邻域的附加输入点                                                                                                                            | 在 `receivers_xyz` 处原生查询 |
| RIND                                                                 | local16 + C                                                                      | 第 3 节                     | 本来就有                                                                                                                                              | 第 3 节                       |

### 5.5 统一的输出、loss 与预算

- **输出**：所有模型在 N 个接收点、每一帧输出一个 logit，导出 `pred_probs_{tile_id}.npy`（float32，`[T, N]`）。温度和阈值由统一裁判脚本在 calibration 上拟合。

- **loss**：各模型使用自己原生的训练目标，不要求 loss 公式相同。loss 只用于模型内部的参数更新，评测由统一裁判脚本完成。需要统一的是**可用的标签信息**：
  - 所有模型可用的标签字段相同：`truth`、`valid_mask`、`boundary_distance_m`，只在同样的 N 个接收点上，按同样的标签预算；
  - 每个模型的 loss 实际用了哪些标签字段、有哪些不读标签的辅助项（例如 latent 预测、重建），都要列出；
- **预算**：同样的 `fit_67`、同样的 N = 256 个接收点、同样的 day batch、相同的更新步数、相同的 seed，统一使用 FP32。
- **允许不同、但必须报告的**：各模型的 loss 及其使用的标签字段、backbone 和读出头结构、模型规模、各类模型的几何格式（H 或 P）、RIND 特有的事件门与 kNN 读出。

---

## 6. 实验设计：主实验与补充对照

| 设置         | 几何输入                                                                   | 回答的问题                                               |
| ------------ | -------------------------------------------------------------------------- | -------------------------------------------------------- |
| **主实验**   | 原生形式，覆盖范围对齐到 4 km：图像类用 H，点云类用 P；RIND 用 local16 + C | 不给手工检索时，谁能自己从几何里找到并用上远处的遮挡信息 |
| **补充对照** | 主实验的输入 + C（与 RIND G2 相同的 37 维 token）                          | 候选给定以后，谁判断遮挡更准                             |

解读时需要说明：

- **主实验中双方的信息并不等价**：RIND 拿到的是 4 km 内完整 3D mesh 经 az24 筛选的结果，图像类模型拿到的是 2.5D、远处变粗的高度图。差距中有一部分来自输入格式，用下面的高度图 oracle 量化后，在与 RIND 比较时单独说明。
- **补充对照中，远距离检索由 az24 完成**，所以不能用它说明 baseline 能否自己学会长程遮挡。37 维 token 的 22–24 槽几乎就是解析相交结果，这一点也要写明。
- **同一模型两套设置的差值**（补充对照 − 主实验）衡量它从“给定候选”中得到多少收益，也就是它自己做检索的缺口。

配套诊断（不需要训练）：

1. **高度图 oracle**：在 H 上做射线步进得到阴影，与 `truth` 比较，作为高度图输入所含信息的上限。
2. **az24 召回**：用精确射线求交（`ExactSceneBVH`）找出每个阴影点帧的第一个遮挡面，统计它落在当帧 32 个候选中的比例，作为候选输入所含信息的上限。现有的 `compute_blocker_distance_stratified_metrics` 已经输出 `candidate_first_blocker_recall`，按遮挡距离分为 <50 m、50–300 m、300–4000 m 三桶。
3. **遮挡物距离分桶评分**：沿用上面的分桶，用于 nonlocality 的分层评分。注意 portable 模式会拒绝这项诊断，需要原始场景读取路径。

---

## 7. objective 层对照组

### 7.1 对比objective

**在同一个 RIND backbone 上，response-targeted latent alignment 是否比直接监督、或只用几何的表征学习更有效。** RIND 关于“学到的表征”的主张在这一层检验。

四个 arm 的结构完全相同，只差 objective，所以这里不需要单独拿出一个 encoder 来比较：

| 比较                                               | 回答什么                                    |
| -------------------------------------------------- | ------------------------------------------- |
| `response_jepa` 对 `direct`                        | response-targeted 对齐是否优于只用监督 loss |
| `response_jepa` 对 `geometry_mae`、`geometry_jepa` | 以阴影响应为目标，是否优于只以几何为目标    |
| `geometry_mae`、`geometry_jepa` 对 `direct`        | 只用几何的辅助目标本身有没有帮助            |

需要同时说明的几点：

- **四个 arm 的标签预算相同。** 但 `response_jepa` 的 teacher 会读取 5×5 阴影响应（`response_5x5_bitpack`），这属于额外的标签读取，需要按对比设计第 4 节的“标签账本”分别记录。另外三个 arm 的辅助项不读标签。
- **这四个 arm 都不是基于 R0。** R0 是 README 中的完整主方法（`ema_geom`，另含 transition、dual-path、反事实和 dropout）；matched arm 把这些附加项都关掉，只保留 objective 本身的差别。

### 7.2 arm 的定义与代码出入

使用 `--matched-objective`的`response_jepa`、 `direct`、`geometry_mae`、`geometry_jepa`四个arm（[cli.py](rind/cli.py)）：与主方法共用 `rind_full_geom` backbone，但关掉了 transition、dual-path、反事实、dropout 和正则，彼此之间只差 objective

但当前代码与对比设计中的文字定义有以下出入（**待定**，见 11.3）：

|                          | 对比设计中的定义                                      | 代码中的 arm                                                                                                                                                                                                    |
| ------------------------ | ----------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 训练方式                 | Direct supervised 无 pretraining / learning objective | 新代码不再有三阶段训练，也因此没有单独的预训练阶段，现在四个 arm 都与监督 loss 同时训练                                                                                                                         |
| Geometry-MAE（出入不大） | 遮住几何，重建几何标签                                | 每帧随机遮掉 25% 的候选三角形（从 mask 中删除并清零），学生在其余候选上运行，再用 Linear(96→37) 重建被遮三角形 token 的均值（asinh 变换后）。目标 token 中含有与太阳相关的槽位，不是纯几何                      |
| Geometry-JEPA            | 遮住几何，预测被遮几何的 teacher latent               | 不遮几何输入：目标点自己的 z_direct 仍然可见，遮罩只限制来自其他接收点的上下文。预测目标是未来第 k 帧（k = 1, 2, 5, 10, 15）EMA backbone 的潜变量，输入包含未来那一帧的太阳。更接近以太阳为条件的潜变量前向预测 |

---

## 8. Frozen encoder probing（当前结构下不再单独做）

### 8.1 对比 objective

Which **learned representations** is most effective for RIND shadow prediction?
做法是冻结 DINOv3、SAM 2 Hiera、V-JEPA 2.1、Sonata 等预训练 encoder，与 RIND 的表征放在同一个公共 probe 下比较，看哪种表征最适合阴影预测。旧版 RIND 有一个“看栅格”的几何编码器，可以拿来做这种比较。

在当前的训练结构下，这一层已经失去意义：

- **它要比较的是 RIND 的表征和基础模型的表征**，但当前 RIND 没有可以单独冻结来比较的 encoder（见 8.2）。
- **去掉 RIND 之后**，冻结层只能回答关于基础模型自身的问题，例如冻结特征保留了多少远处几何、预训练知识有没有用。这些都不检验 RIND 的任何主张。

因此不再单独做冻结比较：

- **DINOv3（2D）和 Sonata（3D 点云）** 用公开的预训练权重初始化，再整体微调，作为端到端模型并入第 9 节，与 RIND 在同一协议下比较；
- **SAM 2 Hiera 和 V-JEPA 2.1 的冻结版本不做**。V-JEPA 2.1 仍以从零训练的 vanilla 版本保留在第 9 节；
- **RIND 自身的主张**：关于表征的部分在第 7 节检验，关于架构和输入的部分在第 9 节检验。

### 8.2 为什么 RIND 没有可以冻结比较的 encoder

当前 RIND 没有一个“与太阳无关、不看标签”的独立表征模块：`StaticGeometryMemory` 只读 local16，远处信息来自逐帧随太阳变化的 az24 候选。技术上可以在 `z_direct` 处切开（模型已输出 `latent_direct`，[models.py:1802-1811](rind/models.py#L1802-L1811)），但仍有三个问题：

- **现有权重已在评测目标上训练过。** 它们是在同一个评测目标上、用阴影标签端到端训练出来的，而基础模型的权重从未被这些标签更新过，两者不处在同一比较条件下。
- **无标签预训练也测不到学到的表征。** RIND 的远处遮挡能力主要来自没有参数的 az24 检索，以及近似射线求交的 canonical37 token。即使改成不用标签预训练（`fast_train` 支持 `supervise_labels=False`，[fast_train.py:705](rind/fast_train.py#L705)，但 CLI 没有开放），冻结后测到的主要也是这套人工计算的价值。
- **公共 probe 逐帧、无状态。** 它会替换掉 RIND 的循环单元，测不到 RIND 时序状态的作用。

---

## 9. End-to-end 模型：怎样接收全部输入

### 9.1 共同设置

- **初始化**：DINOv3 和 Sonata 用公开的预训练权重初始化后整体微调，其余模型从零训练。
- **预算**：同样的 `fit_67`、N = 256 个接收点、day batch、更新步数、seed 集合（17、29、43），统一使用 FP32。
- **loss**：各模型使用原生的训练目标，可用的标签信息相同。
- **统一查询头**：F(q) 是模型（包括它自己的 decoder）输出的特征图或逐点特征，在接收点 q 处取出的特征向量，各模型的取法见 9.2。F(q) 与 L(q)、太阳 embedding 拼接后，经同一个查询 MLP 输出每个接收点、每一帧的 logit。L(q) 是 RIND 也拿到的局部输入（5.1），所有非 RIND 模型同样拿到。
- **几何输入**：2D 模型用 H0、H1、H2（5.2，9 + 3 + 3 通道）；DINOv3 用伪 RGB（5.2），以沿用预训练的 3 通道 patch embedding；GINO 和 Sonata 用 P（5.3）。所有模型共用同一份 H、P 和同一组归一化常数。

### 9.2 各模型的接法

| 模型                   | 几何                                                                                                                                                       | 太阳                                                       | 补充对照中的 C                          | 取特征 F(q)                                                                   |
| ---------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------- | --------------------------------------- | ----------------------------------------------------------------------------- |
| **Conditional U-Net**  | H0（9 通道）作主输入；H1、H2 各经一个小卷积编码器，在中心对齐后重采样到 bottleneck 的网格并拼接                                                            | 每个 block 做 FiLM                                         | 查询头中用 DeepSets 汇总后拼接          | 在全分辨率的最终特征图上，于 `receiver_rc` 处 bilinear 采样                   |
| **SegMAN**             | H0 用 9 通道 patch embedding 替换 RGB stem；H1、H2 嵌入后作为额外 token 进入全局分支（或与 U-Net 一样在 bottleneck 拼接）                                  | FiLM，或加一个太阳 token                                   | 查询头中对候选 token 做 cross-attention | 取解码器最高分辨率的特征（约 1/4 分辨率）bilinear 采样                        |
| **GINO**               | P 作为输入点（带 5.3 的点特征），经 GNO 分别编码到两张 latent grid：tile 范围内的细网格（近处点）和 8 km 范围内的粗网格（远处点），各经 FNO 处理（见 9.3） | 拼接到 FNO 的 latent 通道，或做 FiLM                       | 作为接收点邻域的附加输入点              | GNO decoder 在 `receivers_xyz` 处分别从两张网格查询，两份特征拼接             |
| **vanilla V-JEPA 2.1** | H0 按 patch 16 切成 token（9 通道）；H1、H2 的 token 加层编码后一并输入                                                                                    | 每帧一个太阳条件 token（类似 V-JEPA 2-AC 的 action token） | 候选 token 作为额外输入 token           | attentive 查询头：用接收点位置编码做 query，cross-attend 到 patch token       |
| **DINOv3**             | H0、H1、H2 各映射成一张 512² 伪 RGB（5.2），沿用预训练的 patch embedding（patch 16），每层单独前向，token 加层编码                                         | 每帧一个太阳条件 token 输入 ViT                            | 候选 token 作为额外输入 token           | attentive 查询头：用接收点位置编码做 query，cross-attend 到三层的 patch token |
| **Sonata**             | P（5.3）：坐标为相对 tile 中心的 xyz，法向取自三角形，颜色用伪 RGB 映射在点处的取值                                                                        | 拼接到逐点特征，或做 FiLM                                  | 作为接收点邻域的附加输入点              | 在 `receivers_xyz` 处对逐点特征做 kNN 插值                                    |

### 9.3 需要明确报告的部分

- **GINO 使用两级 latent grid，这是对原版的改动。** GINO 中间的 FNO 层要求规则网格并做 FFT，显存随网格格数增长。8 km 的范围如果按 0.2 m 划分，每边约 4 万格，无法计算；若改用算得动的格数，格距又会粗到几十米，近处细节全部丢失。所以改为两张网格：tile 范围内一张细网格，8 km 范围内一张粗网格。GNO 把近处点和远处点分别编码到两张网格，decoder 在接收点处从两张网格取特征后拼接。细网格不必达到 0.2 m，接收点处的细节由 L(q) 提供。
- **内部分辨率**：SegMAN 约为 1/4 分辨率，V-JEPA 2.1 和 DINOv3 每个 token 约 3.1 m，边界精度会因此吃亏。这视为这些模型自身的限制，不做额外补偿。
- **太阳的接入方式**：各模型按最自然的方式接入（见 9.2），逐一列出。
- **模型规模**：RIND 很小（内部文档记为约 264K 参数，未核实）。端到端模型优先选较小的版本，参数量全部列出。DINOv3 和 Sonata 选公开版本中较小的一档。
- **预训练模型的域差异**：DINOv3 在自然图像上预训练，高度图不是自然图像；Sonata 预训练数据的点距远小于 P 中远处 LOD 点的点距。微调可以部分弥补，但需写明。

---

## 10. 统一的下游评价

### 10.1 统一裁判

所有模型（包括 RIND）都通过同一个裁判脚本 [evaluate_predictions.py](baseline_kit/evaluate_predictions.py) 评分，不再使用 RIND 自己的 `eval` 输出做跨模型比较。

- **提交格式**：每个 tile 一个 `pred_probs_{tile_id}.npy`（float32，`[T, N]`），也可以提交 logit（加 `--logits`）。需要提交 calibration（12 个 tile）和 val_spatial_clean（14 个 tile）两组。
- **校准**：只在 calibration 上进行，见 [evaluate_predictions.py:60](baseline_kit/evaluate_predictions.py#L60)。
  - 提交 logit 时，按 NLL 拟合一个标量温度（粗细两级网格，范围 0.25–4.0）。
  - 阈值在 [0.05, 0.95] 间取 181 个点，选使 (accuracy + F1) / 2 最大的那个。
- **评测集**：val_spatial_clean。
- **共同支撑集**：只在 `valid_mask & truth ≥ 0` 的点帧上计分。这个集合只由数据决定，与模型无关，所以所有模型天然共用同一个支撑集。
- **参考模型**：同样提交 calibration 和 val_spatial_clean 两组预测（`--ref-cal-preds-dir`、`--ref-probs-dir`，logit 加 `--ref-logits`），裁判按与被评模型完全相同的流程，在 calibration 上校准参考模型自己的温度和阈值，输出在 `reference_calibration` 中。
- **统计不确定度**：
  - 相对参考模型做配对 tile bootstrap（4,000 次重采样），对 state accuracy、Brier 和严格切换 F1 分别报告差值的 95% 区间和逐 tile 胜出数；
  - 多个 seed 报告均值 ± 标准差。

使用前需要先完成：**RIND 导出 `pred_probs`**。当前 CLI 只有指标汇总，没有导出逐 tile 预测文件的功能。`eval` 在打开 `--shadow-contours` 时已会收集 logits，在此基础上加导出即可。

### 10.2 指标清单

| 指标                                | 定义                                                                                                                                                                                                                                                           | 在哪里计算                                                                                                                                                                                                                                 |
| ----------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| state accuracy / error              | 二值化后的逐点帧准确率                                                                                                                                                                                                                                         | 统一裁判                                                                                                                                                                                                                                   |
| F1、Sun IoU                         | 以“亮”为正类                                                                                                                                                                                                                                                   | 统一裁判                                                                                                                                                                                                                                   |
| Brier                               | (p − y)² 的平均                                                                                                                                                                                                                                                | 统一裁判                                                                                                                                                                                                                                   |
| insolation bias                     | \|FP − FN\| / 点帧数，衡量整体偏亮或偏暗                                                                                                                                                                                                                       | 统一裁判                                                                                                                                                                                                                                   |
| 边界 <0.5 m 准确率                  | `boundary_distance_valid` 且距离 < 0.5 m 的点帧上的准确率                                                                                                                                                                                                      | 统一裁判                                                                                                                                                                                                                                   |
| 严格switch F1（`trans_f1_0s`）      | 同一天内相邻 120 s 帧对，预测的切换必须与真实切换在同一帧                                                                                                                                                                                                      | 统一裁判                                                                                                                                                                                                                                   |
| 事件召回（±2 min）、中位延迟        | 每个真实切换事件，在 ±1 帧内有同方向的预测切换即算命中                                                                                                                                                                                                         | 统一裁判                                                                                                                                                                                                                                   |
| balanced accuracy                   | (TPR + TNR) / 2                                                                                                                                                                                                                                                | 统一裁判                                                                                                                                                                                                                                   |
| ECE                                 | 按预测概率等宽分为 15 箱，各箱 \|平均概率 − 实际亮的比例\| 按点帧数加权求和；多 tile 时先合并各箱计数再算                                                                                                                                                      | 统一裁判                                                                                                                                                                                                                                   |
| 变化点 / 稳态点 Brier               | 同一天内相邻两帧 (t−1, t) 都有效时，按第 t 帧计分：真值发生切换的算变化点，没变的算稳态点，分别计算 Brier；当天第一帧和前一帧无效的帧不计入                                                                                                                    | 统一裁判                                                                                                                                                                                                                                   |
| 低太阳角分组                        | 太阳高度角 0–10° 的帧单独重算全套指标（读分片中的 `solar_altitude_deg`，缺失时由 `sun_dirs` 算出）；其中切换 F1 只用前后两帧都在低太阳角的帧对                                                                                                                 | 统一裁判                                                                                                                                                                                                                                   |
| precision / recall、Wilson 95% 区间 | 同 [metrics.py](rind/metrics.py) 的 `ConfusionAccumulator`                                                                                                                                                                                                     | `rind.metrics`                                                                                                                                                                                                                             |
| Shadow IoU                          | 以“暗”为正类的 IoU                                                                                                                                                                                                                                             | `rind.metrics`                                                                                                                                                                                                                             |
| BCE NLL                             | 评测集上的交叉熵                                                                                                                                                                                                                                               | `rind.metrics`                                                                                                                                                                                                                             |
| 5 个太阳分区                        | 日出低角、上午、正午、下午偏西、日落低角（低角为高度角 ≤ 15°，正午为方位角 165–195°）                                                                                                                                                                          | `rind.metrics`                                                                                                                                                                                                                             |
| 边界距离分带                        | <0.5 / 0.5–2 / >2 m 三带，每带完整混淆矩阵和 Brier                                                                                                                                                                                                             | `rind.metrics`                                                                                                                                                                                                                             |
| 分方向切换 F1                       | 亮→暗、暗→亮分别计算，即对比设计中的“方向错误数”                                                                                                                                                                                                               | `rind.metrics`                                                                                                                                                                                                                             |
| 空间组聚类 bootstrap                | 按 spatial group 重采样的 95% 区间                                                                                                                                                                                                                             | `rind.metrics`                                                                                                                                                                                                                             |
| 遮挡物距离分桶                      | 用精确光线求交（`ExactSceneBVH`）找出每个点帧朝太阳方向的第一个遮挡物，按距离分为 <50 / 50–300 / 300–4000 m，每桶报告 shadow recall、state accuracy 和 Brier；太阳在地平线以上且 4 km 内无遮挡的点帧没有遮挡物，单独报告真值为亮却预测为暗的比例（受光点误报） | 统一裁判（`--blocker-buckets`）：从 `.tar.gz` 分片读场景网格做光线求交，分桶边界与 `rind.metrics.compute_blocker_distance_stratified_metrics` 一致。遮挡距离只由几何决定，每个 tile 追一次后存入 `--blocker-cache-dir`，之后各模型直接复用 |
| 阴影轮廓 F1                         | 栅格四邻接边上的轮廓匹配，容差 1 像素 / 0.195 / 0.5 / 1.0 m                                                                                                                                                                                                    | `rind.metrics`（`compute_shadow_contour_f1`），**N = 256 时不算**，轮廓需要稠密接收点，只在 `all_valid` 下计算                                                                                                                             |
| 0.5 m 带内的距离 MAE 和 p95         | —                                                                                                                                                                                                                                                              | **无法计算**：模型只提交概率，没有预测距离                                                                                                                                                                                                 |
| 计算成本                            | 推理延迟（每个有效点帧的 μs）、参数量、训练 GPU-hours                                                                                                                                                                                                          | **裁判之外单独测量**：只有 RIND 有 `benchmark-latency`，其他模型需按同一口径测                                                                                                                                                             |

标为 “`rind.metrics`” 的指标，在裁判中调用 `rind.metrics.evaluate_trajectory_predictions` 即可得到，不需要重写；这一步还没接上。裁判本来就会导入 `rind.split_contract`。

### 10.3 各层的主指标

裁判一次就能算出 10.2 中的全部常规指标，代价很小；只有遮挡物距离分桶需要光线求交。

**所有层共用的三个主指标：**

1. **Brier**；
2. **边界精度**：评测用 N = 256 时，用边界 <0.5 m 准确率；评测用 `all_valid` 时，用 0.195 m 轮廓 F1；
3. **严格 switch F1**（`trans_f1_0s`）。

| 层                   | 正文                                                           | 附表（次要）                                                                                                    |
| -------------------- | -------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| objective（第 7 节） | 三个主指标在 1%、5%、100% 三档标签预算下的曲线                 | accuracy、Sun / Shadow IoU、ECE、分方向切换 F1、边界距离分带、变化点 / 稳态点 Brier、5 个太阳分区、低太阳角分组 |
| 端到端（第 9 节）    | 三个主指标，遮挡物距离分桶的 shadow recall，并单独报受光点误报 | 同上，再加计算成本（推理延迟、参数量、训练 GPU-hours）                                                          |

说明：

- **端到端层要跑遮挡物距离分桶**，因为第 6 节主实验问的正是“谁能用上远处的遮挡信息”。
- **objective 层的重点是标签预算曲线**。几种 objective 在 100% 标签下的差别通常很小，主要差别预期出现在低预算下。
- **objective 层各 arm 之间的差值预期较小**，必须用配对 bootstrap 和多 seed，单个 seed 的差值不作结论。
- **降低预算时，`response_jepa` 的 teacher 能读到的信息也随之减少**：`apply_label_budget_view` 会去掉预打包的 5×5 响应，只从有标签的点重建（[objectives_jepa.py:253](rind/objectives_jepa.py#L253)）。这一点要记入标签账本。

---

## 11. 共用部分、各自部分与待实现汇总

### 11.1 共用与各自

| 共用（所有模型完全相同）                                                                            | 各自（允许不同，但必须报告）                                                         |
| --------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| 数据划分：`fit_67` 训练、`calibration` 校准、`val_spatial_clean` 评测（1）                          | backbone 结构与模型规模（参数量全部列出，9.3）                                       |
| 同样的 N = 256 个接收点、Q / S / L 三个组件（5.1）                                                  | 几何格式：图像类用 H，点云类用 P，RIND 用 local16 + C（5.4、6）                      |
| 同一份 H、P（固定种子）和同一组全局归一化常数（5.2、5.3）                                           | H1、H2 的多尺度融合方式，太阳的接入方式（9.2）                                       |
| 可用的标签字段 `truth`、`valid_mask`、`boundary_distance_m`，同样的标签预算（5.5）                  | 原生 loss，及其实际读取的标签字段和不读标签的辅助项（5.5）                           |
| `fit_67`、day batch、更新步数、seed 集合（17、29、43）、FP32（5.5、9.1）                            | 初始化：DINOv3、Sonata 用预训练权重，其余从零训练（9.1）                             |
| 非 RIND 模型的统一查询头 MLP([F(q), L(q), sun_emb])（9.1）                                          | F(q) 的取法（9.2）                                                                   |
| 补充对照中的候选 C：与 RIND G2 相同的 az24 候选和 37 维 token（5.1、6）                             | RIND 特有的部分：事件门、kNN 读出、递推状态，以及 R0 的 JEPA 辅助项和 `L_cf`（3、4） |
| 输出格式 `pred_probs_{tile_id}.npy`（`[T, N]`）；统一裁判的校准、共同支撑集、配对 bootstrap（10.1） |                                                                                      |

### 11.2 待实现

**几何输入与基线模板（第 5、9 节）**

2. **加载端的统一预处理**：脚本不做归一化，需要一份所有模型共用的加载代码，负责：全局归一化常数、H0 → 伪 RGB 映射（5.2）、给各层特征加层编码（5.2）。
3. **扩展基线模板**：[baseline_template.py](baseline_kit/baseline_template.py) 中的 `StarterBaselineModel` 目前只用 `receivers_xyz` 和 `sun_dirs`。需要接入 L、H、P（补充对照中再接 C），并提供共用的统一查询头和通用 set encoder（DeepSets，max + mean 池化，5.4）。
4. **为外部模型生成并分发 az24 候选库**（补充对照，第 6 节），以及在 baseline 侧逐帧算出 37 维 token 的代码（复用 `build_canonical37_torch`）。

**配套诊断（第 6 节，不需要训练）**

5. **高度图 oracle 脚本**：在 H 上做射线步进，与 `truth` 比较。它的结果还决定 11.3 中层间倍率的取舍。
6. **在 val 上跑 az24 召回**（`candidate_first_blocker_recall`）。portable 模式会拒绝这项诊断，需要原始场景读取路径。

**训练与 loss（第 4、5、7 节）**

8. **标签账本**：逐个模型列出 loss 实际读取的标签字段和不读标签的辅助项。下面几项要特别写明：
   - `response_jepa` 的 teacher 读取 `response_5x5_bitpack`（7.1）；
   - 降低预算时，`apply_label_budget_view` 只从有标签的点重建 5×5 响应（10.3）；
   - R0 的 `L_cf` 用候选三角形的解析射线求交生成目标，相当于额外的物理监督，baseline 没有这项（4.2）。
9. **objective 层三档标签预算的训练**：1%、5%、100%，每档跑完 3 个 seed（10.3）。CLI 和基线模板都已支持 `label_budget_fraction`（seed 20261112）。
10. **伪 RGB 映射损失的对照**：同一个 Conditional U-Net，在相同预算下分别输入 9 通道 H0 和 3 通道伪 RGB，多跑一次训练（5.2）。另有可选的敏感性实验：B 通道换成固定光源 hillshade。

**评测（第 10 节）**

13. **RIND 导出 `pred_probs`**：当前 CLI 只输出指标汇总。`eval` 打开 `--shadow-contours` 时已会收集 logits，在此基础上加导出即可（10.1）。
14. **裁判接入 `rind.metrics.evaluate_trajectory_predictions`**：10.2 中标为 `rind.metrics` 的指标靠它得到，这一步还没接上。
15. **非 RIND 模型的计算成本测量**：推理延迟按 RIND `benchmark-latency` 的口径测，再加参数量和训练 GPU-hours（10.2）。
16. **all_valid / 全图评测的路径**（只在 11.3 第 1 项选择 all_valid 时需要）：
    - az24 候选库只为 256 个采样点预先算好（2.2）。其他点需要现算，可以用 [rind_scene_cell_az24_gpu.py](scripts/rind_scene_cell_az24_gpu.py)，它声称与 `select_azimuth_sector_candidates` 结果完全一致，并配有 [test_scene_cell_az24_parity.py](scripts/test_scene_cell_az24_parity.py)；
    - 需要估计现算候选和遮挡物距离分桶光线求交的计算代价；
    - 阴影轮廓 F1 依赖这条路径（10.2、10.3）。

### 11.3 待定的设计选择

2. **matched arm 与对比设计文字定义的三处出入**（7.2）：没有单独的预训练阶段；Geometry-MAE 的重建目标含有与太阳相关的槽位；Geometry-JEPA 不遮几何输入，实际是以太阳为条件的潜变量前向预测。需要决定是改代码还是改定义。
3. **高度图金字塔的层间倍率**：目前每层 8 倍（三层），或改为每层约 4 倍（100 m、400 m、1.6 km、8 km 四层）。根据高度图 oracle 的结果决定（5.2）。
4. **暂定参数**：
   - 伪 RGB 的高度区间：−20 m 到 200 m（5.2）；
   - P 的点距参数：s₀ = 0.5 m、Δθ = 0.5°（5.3）；
   - GINO 两级 latent grid 的格数和格距（9.3）。

### 11.4 尚不确定、需要核实

- **分片是否带 `scene_ground_mask`**：如果不带，H1、H2 的建筑高度通道以及 P 的地面剔除，都改用 DSM → DTM 开运算估计地形，这是近似（[build_geometry_inputs.py](baseline_kit/build_geometry_inputs.py) 文件头）。
