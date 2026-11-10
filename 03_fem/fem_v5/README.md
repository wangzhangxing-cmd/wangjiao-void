# 第3章有限元验证 v5：建模与结果提取

本目录有两个脚本：

| 脚本 | 运行环境 | 作用 |
|---|---|---|
| `build_fem_model.py` | Abaqus 自带 Python（含 CAE 的 Run Script）或普通 Python，2.7/3 均可；只需要 numpy，不依赖仓库里的其他文件 | 生成 Abaqus `.inp` 和对应的 `_meta.json` |
| `extract_fem_results.py` | Abaqus 自带 Python（命令行或 CAE 的 Run Script，2.7/3 均可） | 从 ODB 提取与半解析解同口径的结果 |

两个脚本可以单独拷到任何装有 Abaqus 的电脑上使用。

## 与旧模型（`03_fem/results_b10_p100`）相比改了什么

| 项目 | 旧模型 V4 | 新模型 V5 | 原因 |
|---|---|---|---|
| 单元 | CPE4R（单积分点） | CPE8R（二次单元） | 应力梯度大，需要能外推到表面 |
| 钢板沿厚度 | 2 层 | 4 层（medium），6 层（fine） | 半解析解报的是内表面峰值 |
| 环向网格 | 均匀，约 0.48° | 向脱空端点加密到 0.04°（medium）、0.015°（fine），端点正好是节点 | 峰值出现在端点附近 |
| 钢板峰值取法 | 积分点最大值 | 外推到节点的表面应力（同时输出积分点最大值作对照） | 和半解析的表面应力同口径 |
| 钢板–衬砌连接 | Tie | 共用节点（完全粘结） | 和半解析假设一致 |
| 开挖 | 删除单元 | 洞壁支护压力从 p0 线性降到 0，分 30 个增量 | 卸荷路径和半解析完全一致 |
| 外边界 | 260 m 正方形 | 半径 40R=260 m 的圆，外圈 3R 以外用粗网格并用 Tie 连接 | 远场更接近无限域 |
| 界面压力 | 近壁单元中心应力 | 岩侧和衬砌侧节点应力，以及节点力合力三种算法 | 和半解析界面牵引同一位置 |
| 塑性应变范数 | PE12 被当成张量剪切 | PE12（工程剪应变）按张量换算：+0.5·γ² | 旧脚本把剪切项多算了约 2 倍 |

RC 五层材料按 `02_code/core/layered_ring_operator.py` 的同一算法内置在脚本中，和半解析模型、旧 V4 模型的工程常数逐位一致。

## 生成模型

三种方式任选一种：

- Abaqus/CAE：File → Run Script → 选 `build_fem_model.py`。不带参数，自动生成下表 6 个模型。
- 命令行：
```bash
abaqus python build_fem_model.py --matrix
```
- 只生成单个模型：
```bash
abaqus python build_fem_model.py --beta 10 --rock ep --mesh medium --pressures 1,2,3,4,5,6,7,8,9,10
```

输出到脚本所在文件夹下的 `models/`；CAE 里若取不到脚本路径，则输出到 CAE 的工作目录下的 `models/`，屏幕上会打印实际路径。fine 模型约 90 MB，生成要几分钟。

`--matrix` 生成的 6 个模型及用途：

| 作业名 | 单元数 | 用途 |
|---|---|---|
| `V5_B10_EL_medium` | 12.3 万 | 弹性围岩、10° 脱空：和弹性半解析解直接对比，验证界面方程和衬砌算子，不受塑性本构差异影响 |
| `V5_B10_EP_medium` | 12.3 万 | 主验证模型：替代旧 V4 的弹塑性模型 |
| `V5_B00_EL_medium` | 11.7 万 | 无脱空、弹性：检验有限元本身（应等于轴对称弹性解） |
| `V5_B00_EP_medium` | 11.7 万 | 无脱空、弹塑性：和轴对称闭式解对比，分辨 Abaqus 光滑 Mohr–Coulomb 与经典 Mohr–Coulomb 的差异，并检查 1–2 MPa 下的"提前塑性"是否是有限元自身造成的 |
| `V5_B10_EP_coarse` | 3.8 万 | 有限元网格收敛性 |
| `V5_B10_EP_fine` | 42.7 万 | 有限元网格收敛性（较慢，可最后跑） |

## 运行 Abaqus

```bash
abaqus job=V5_B10_EP_medium input=V5_B10_EP_medium.inp cpus=8 interactive
```

每个分析步只输出最后一个增量步（`*Output, field, frequency=10000`，Abaqus 总会写出分析步的最后一个增量步）。

弹塑性模型用了非对称求解器（`unsymm=YES`，非关联流动需要）。如果某个压力步不收敛，可以把该步 `*Static` 的初始增量从 0.1 改成 0.02。

## 提取结果

最简单的做法：把 `extract_fem_results.py` 放到 ODB 所在文件夹，`_meta.json` 放在同一文件夹或其下的 `models/` 中，然后在这个文件夹里运行（或在 CAE 里 Run Script）：

```bash
abaqus python extract_fem_results.py
```

脚本会处理该文件夹里所有带 `_meta.json` 的 ODB，结果写到 `results_<作业名>/`。也可以指定单个 ODB：

```bash
abaqus python extract_fem_results.py --odb V5_B10_EP_medium.odb --meta models/V5_B10_EP_medium_meta.json --out results_V5_B10_EP_medium
```

输出：

- `summary.csv` / `summary.json`：每个压力步一行，包括
  - `q_mean_resultant_mpa`：接触段衬砌外表面节点力合力 ÷ 接触弧长（平均接触压力的主指标）
  - `q_mean_rock_stress_mpa` / `q_mean_lining_stress_mpa`：由岩侧、衬砌侧节点应力求的弧段平均，三者应接近
  - `contact_ur_mm` / `void_ur_mm`：衬砌外表面接触段、脱空段平均径向位移（相对衬砌安装时刻）
  - `seal_mises_max_mpa`、`seal_peak_local_angle_deg`：钢板内外表面节点应力的最大 von Mises 应力及位置（局部角，0 为拱顶）
  - `seal_mises_ip_max_mpa`：积分点最大值，仅作对照
  - `concrete_maxp_max_mpa`、`plastic_area_m2`、`plastic_outer_radius_m`
  - `tie_gap_mm`：接触段衬砌与洞壁的位移差，应约为 0
- `Pxxx_seal_surface.csv`：钢板内外表面每个节点的 σrr、σθθ、σzz、τrθ 和 von Mises 应力（画图8(b)）
- `Pxxx_interface.csv`：每个界面节点的衬砌、围岩位移和两侧牵引（画图8(a)、界面压力分布）
- `Pxxx_concrete.csv`：各角度混凝土最大主应力（画图8(c)）
- `Pxxx_rock_plastic_cells.csv`：运行期塑性增量范数 > 1e-6 的单元（画图9）
- `summary.json` 里的 `excavation`：开挖结束时的塑性区面积和外半径，可以和半解析安装状态对比

运行期塑性区按单元平均的 ΔPE 张量范数 > 1e-6 判定，ΔPE 相对衬砌安装时刻计算，与论文中半解析的判定方式相同。`reaches_outer_ring` 应为 `false`；若为 `true`，说明塑性区越过了 3R，需要把 `build_fem_model.py` 里的 `r_tie` 放大。

## 拿到结果后的检查顺序

1. **B00_EL**：`q_mean` 应为 0.87524 × p（MPa），`contact_ur_mm` 应为 0.5974 × p。这两个数是轴对称闭式解，用 `02_code/core/axisymmetric_closed_form_reference.py` 算得。偏差应小于 0.5%，否则先查有限元模型。
2. **B00_EP**：闭式解给出的运行期首次屈服内压为 4.41 MPa；10 MPa 时 q=8.712 MPa、洞壁位移 6.484 mm。如果有限元在 4.41 MPa 以下就出现运行期塑性，说明论文图9 低压段的差异来自有限元或 Abaqus 本构，不是半解析模型的问题。
3. **B10_EL**：和弹性半解析解对比。用 `FreeSectorInterface(10, 面板数, 1024, 8192).solve(p)` 就能算出来，几秒钟一个工况，不依赖核函数文件。
4. **B10_EP coarse/medium/fine**：确认有限元钢板峰值收敛后，再和半解析的高分段数结果（或外推值）比较。

## 已知局限

- Abaqus 的 Mohr–Coulomb 流动势是光滑化的（子午面偏心率 0.1，偏平面 Menetrey–Willam 形式），半解析用的是经典非光滑 Mohr–Coulomb。两者的差异由 B00_EP 和闭式解的对比量化。
- 脱空区按"持续存在的非接触界面"处理（不建接触），和半解析假设一致；若需检查脱空是否闭合，可看 `Pxxx_interface.csv` 中脱空段衬砌和岩壁的径向位移差。
- 两个脚本已按 Python 2.7 语法检查，并在 Python 3 下实测；提取脚本用仿造 ODB 测过全部代码路径，但都没有在真实 Abaqus 中运行过。衬砌侧牵引（`q_lining_side`）依赖 `directions=YES` 输出局部柱坐标分量；若它与岩侧牵引明显不同，以岩侧和合力结果为准。

## 2026-11-10 修改：围岩端点网格与求解控制

`V5_B10_EP_medium`（旧版网格）在内压约 3.1 MPa 时停滞：残差最大的节点全部位于脱空端点处的围岩（±4.8°～5.0°，距洞壁不到 0.13 m）。原因是粘结–自由交界点有应力奇异性，围岩在端点处加密到 0.04° 后，塑性应变集中在极小单元内，Mohr–Coulomb 返回映射不收敛。修改如下：

- 围岩与衬砌的环向网格分开。衬砌仍在端点加密（medium 0.04°），围岩在端点只加密到 `rock_theta_min`（coarse 0.5°、medium 0.25°、fine 0.125°），与半解析围岩单元（约 0.7°）同量级。两者用节点–面粘结约束连接，衬砌为从面；两边网格都包含脱空端点。
- 弹塑性分析步采用旧 V4 模型已验证能收敛的控制参数：不连续分析、位移修正容许比 10%、线搜索、更多的尝试和回退次数；每个增量步最大 0.1 MPa。
- 提取脚本相应修改：围岩侧量先在围岩节点上计算，再按角度插值到衬砌节点。`Pxxx_interface.csv` 中去掉了 `rock_node` 列。

已算完的两个弹性模型不需要重算：弹性问题与围岩端点网格无关，它们的结果仍然有效。

## 无人值守批量计算（推荐用于弹塑性模型）

通过远程桌面在 CAE 里提交的作业，可能在远程会话断开或 CAE 崩溃时被终止（`.msg` 中显示 SIGTERM）。`run_models.bat` 依次计算 `JOBS` 中列出的模型，算完后自动提取全部结果，进度写入 `models\run_log.txt`。用"任务计划程序"启动它，计算就不再依赖远程会话和 CAE：

1. 把 `run_models.bat` 放在 `build_fem_model.py` 旁边，`.inp` 和 `_meta.json` 放在其下的 `models` 文件夹；按需修改脚本里的 `JOBS` 和 `CPUS`。
2. 打开"任务计划程序"，点"创建任务"（不是"创建基本任务"）：
   - 常规：选"不管用户是否登录都要运行"，勾选"使用最高权限运行"；
   - 触发器：新建，选"一次"，时间设为两分钟后；
   - 操作：新建，程序或脚本选 `run_models.bat`，"起始于"填 `run_models.bat` 所在文件夹；
   - 条件：取消"只有在计算机使用交流电源时才启动此任务"；
   - 设置：取消"如果任务运行时间超过以下时间，停止任务"。
3. 确定后输入 Windows 账户密码。到时间后任务自动开始，可以断开远程桌面。
4. 查看进度：`models\run_log.txt`，以及各作业的 `.sta` 文件。

