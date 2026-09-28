# ECAL event 16474：无效几何输入的来源与修复

## 已验证的来源

数据集：`mum_regression_20to120MeV_10cmPlateGun_0To10deg`。
原始文件：`oa_g4_mum_00000001-0016_vmw3il6uds5o_SG4_000`。
truthmaker/detresp entry：16474；旧 validation PyG graph index：11102。

ROOT 中 digit 0 的 channel 为 0，坐标为 `(1e10, 1e10, 1e10)`。
其真实 contributor 的 primaryId 为 **1**，位置约为
`(7488.23, 40.924, 7700.79) mm`；不是人工噪声的 primaryId=-1。

在原始 detresp 事件加载的真实几何上重放，得到：

```text
old_geom_ok=0 geom_id=0 old_chan_ok=1 old_channel=0
new_geom_ok=1 new_chan_ok=1 new_channel=2282324480
```

因果链：

1. `ICrystal::DirectConversion` 忽略通用 `GetGeometryId` 的失败返回值，
   且 geometry/channel ID 在 hit 循环外声明，有复用上一 hit 状态的风险。
2. 这个事件的后续 `GetChannelId` 返回 true，却留下 channel=0。
   因此仅检查该函数的布尔返回值不足以判断 ECAL channel 有效。
3. truthmaker 将 channel=0 转为 sequence=-1，调用
   `IECALGeom::GetCrystalPosition(-1)`，获得有限的 `1e10` 占位坐标。
4. truthmaker 及转换器没有拦截，坏节点进入 seed window 和事件 GNN。

ECAL 专用几何查询已有晶体边界/包裹体附近的投影和查找处理；此次实测能找回
有效 channel。修复仅在通用映射失败或 channel 无效时调用该回退，保留成功映射。

## 修改

- `ICEDUST_packages/SimDetectorResponse/src/ICrystal.cxx`：逐 hit 初始化 ID，
  检查几何查询与 channel，ECAL 映射失败时尝试专用查询，仍无效则跳过。
- `ICEDUST_packages/SimDetectorResponse/src/IECALDetResp.cxx`：MC hit 生成路径采用相同防护。
- `ICEDUST_packages/ReconECAL/src/IECALTruthMaker.cxx`：无效 digit 不进入 truth
  cluster 或 waveform fitter；保留原 digit index，重新构建 cluster；记录被剔除对象。
- `AnaApp/AddStrEcalRandomNoise.cxx`：零占用率/零 rate 直接返回，检查输入参数，
  检查噪声候选的晶体几何，打印实际配置和剔除计数。
- `sim_generator/sim_pipeline.sh`：调用当前源码生成的 `AddStrEcalRandomNoise`，
  替换原来调用的旧 `AddEcalRandomNoise`；显式设置 straw floor rate 为零。
- PyG converter：在 seeding 和训练特征统计前检查几何。
  legacy 模式过滤坏 pulse，并同步过滤所有 pulse 字段，保留原 pulse/digit ID。
  regression 模式跳过受污染的整个事件，避免旧 ROOT 中已污染的 cluster 标签继续使用。
- PyG checker：增加非有限值、占位坐标、无效 channel/sequence/block/crystal 检查。

注意：几何坐标的 `1e9 mm` 检查仅用于拦截库的 sentinel，不是探测器的 fiducial cut；
主要几何有效性依据仍然是 ECAL channel 和几何 ID。

## 验证

- ReconECAL、SimDetectorResponse 库编译及链接通过。
- AddStrEcalRandomNoise 可执行文件编译及链接通过。
- 5 项 Python 回归测试通过，覆盖无效 ID/坐标、字段对齐、正常事件保持、
  truth digit 无效但 reco pulse 表面有效、两个 regression step 的事件隔离。
- 在原始 event 16474 上运行新 `ICrystal`：12 个原始 hit 生成 12 个 carrier，
  全部归入 4 个有效 ECAL channel，异常 hit 被正确映射而非丢失。
- 对旧 detresp 的该事件运行新 truthmaker：剔除 digit 0，保留 digit 1–4，
  输出 1 个重建后的 truth cluster，所有保留的 digit 几何有效。
- 从真实 ROOT 提取 entry 16473–16475：converter 输入 pulse 数 `[6,4,0]`，
  regression 清理后 `[6,0,0]`；只隔离中间的坏事件，其他 entry 和字段保持对齐。
- 用用户参数在原始 G4 event 16474 运行噪声程序：ECAL 和 Straw 新增 hit 均为 0。
- 新 checker 能识别旧 PyG graph 11102 的坏坐标和所有无效几何 ID。

测试命令：

```bash
/home/comet/davidz/PyTorch/miniforge3/envs/gnn-pyg/bin/python \
  -m unittest discover -s pyg_converter -p test_ecal_geometry.py -v
```

## 生效范围

本次修改源码、构建 ICEDUST build 目录中的两个库，并更新 ReconDev/build/bin 中的
噪声程序。没有覆盖 ICEDUST_install 中的库，也没有重写已有 ROOT/PT 数据。
流水线默认使用 ICEDUST_install/setup.sh；正式重新生产前需安装更新后的两个库。
安装时应保留现有生产参数文件。

对于已有数据，可以重新运行 PyG converter 隔离坏事件；完整恢复这个物理 hit 的
能量贡献，需要从原始 G4 重新运行修复后的 detresp → truthmaker → converter。
仅重跑 truthmaker 能安全剔除旧的无效 digit，但无法恢复错误 channel 中的信号。
