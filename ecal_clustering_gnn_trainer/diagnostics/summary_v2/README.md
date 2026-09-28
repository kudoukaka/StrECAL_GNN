# ECAL summary：完整残差统计

`reconstruct_ecal_clusters_from_eval_root.cxx` 现在使用独立的未分箱统计量，
显示范围外的有限残差也参与计算。`resolution` 明确定义为总体标准差，
同时输出相对零点的 RMSE。匹配和能量筛选仍沿用 evaluation 已保存的结果，
matched 统计不代表全部 truth objects。

- bias = mean(residual)
- std = sqrt(mean((residual - bias)^2))
- RMSE = sqrt(mean(residual^2)) = hypot(bias, std)
- YZ combined resolution = hypot(std(dy), std(dz))
- YZ radial mean = mean(hypot(dy, dz))
- YZ radial RMSE = sqrt(mean(dy^2 + dz^2))

原 `matched_yz_radial_error_mm` 二维图保留，新增一维
`matched_yz_radial_distance_mm` 表示真正的径向距离。
所有有限残差的计数、均值、标准差、RMSE、显示范围、underflow/overflow
以及无效值数量写入 `residual_statistics` TTree。
非有限值、零分母等未定义相对误差不参与统计，并计入 invalid。
空样本的统计量为 NaN，不显示为零分辨率。

这里的两个 ROOT 文件与文本是基于原始 epoch 005/050 evaluation ROOT
重新计算的结果。原数据目录只读，因此新结果写在此处。

| matched 指标 | epoch 005 | epoch 050 |
|---|---:|---:|
| t0 bias (ns) | -2.9361 | 1.3372 |
| t0 std (ns) | 7.6267 | 8.5680 |
| t0 RMSE (ns) | 8.1724 | 8.6717 |
| YZ radial mean (mm) | 7.7475 | 7.8106 |
| t0 超出显示范围 [-50,50) 的数量 | 537 | 753 |

验证包含：非零 bias、显示范围内外混合样本、范围边界、非有限值、
空样本、径向距离、ROOT 写入后读取、Step-2 零能量分母。
另独立遍历两个真实 evaluation tree，核对 t0 和径向矩、尾部计数及
NO/WITH delayed 分组重新合并后的统计量，均通过。

从 ReconDev 目录运行单元测试：

```bash
c++ $(root-config --cflags) -std=c++14 \
  ecal_clustering_gnn_trainer/tests/test_reconstruction_summary.cxx \
  $(root-config --libs) -o /tmp/test_ecal_summary
/tmp/test_ecal_summary
```
