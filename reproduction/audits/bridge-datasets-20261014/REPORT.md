# 桥梁缺陷数据核验记录

核验日期：2026-10-14（Asia/Shanghai）。本阶段只检查，不改动原始数据、不启动训练。

## 官方一致性结论

| 数据 | 结论 | 证据 |
|---|---|---|
| dacl10k_v3_512 | 与官方 v3 的 512 发布包一致 | 两份 ZIP 的 MD5/大小一致；ZIP CRC 通过；全部解压文件与 ZIP 内文件 SHA256 一致 |
| cif | 已下载的官方 full 部分逐字节一致 | 498 个 Parquet 的大小与 SHA256 均与官方 LFS 一致，无缺失或多余分片 |
| Bridge2893_split_seed42 | 自定义组合数据集，内部核验通过 | 元数据校验、图像/掩码可读性、尺寸、同源组与相同图像跨划分检查通过 |

### DACL10k-v3

官方来源：[UniBw 数据仓库，DOI 10.60776/XL7ZRG](https://doi.org/10.60776/XL7ZRG)。本地目录：
`/home/skye/data/Skye/databases/dacl10k_v3_512`。

| 官方文件 | MD5 | 大小（字节） |
|---|---|---:|
| dacl10k_v3_images_512.zip | fe4a578197dbdb1ec3143358d2194ed7 | 1090777963 |
| dacl10k_v3_devphase_annotations_512.zip | ca1bab7849c9f89631e57e5fb013be21 | 7185826 |

实际压缩包、官方校验值及大小全部匹配。9,920 个图像文件、7,910 个标注文件均与压缩包内容一致；全部图像可以解码，分辨率为 512×512。

| 划分 | 图像 | 公开标注 |
|---|---:|---:|
| train | 6935 | 6935 |
| valid / validation | 975 | 975 |
| testdev | 1012 | 当前发布包不提供 |
| test | 998 | 当前发布包不提供 |

实际标注覆盖 19 类。按照 13 个损伤类与 6 个对象类的定义，train 中 534 张、validation 中 58 张没有标注损伤类；其中完全没有 shape 的分别为 73、6 张。
这些只能称为“没有标注目标损伤”的候选图像，不能仅凭标注缺失认定完全无缺陷。
若只定义 Crack/Spalling/Rust/Efflorescence 四类为目标，未标注这四类的数量为 train 1330、validation 197，但可能含其他损伤。

原始 JSON 的 `imagePath` 使用 `../../img/...`，与解压目录不同；按 `imageName` 和 split 配对全部通过。迁移加载器需按实际目录解析，不能直接拼接这个原始字段。

官方 v3 元数据说明修正了部分 v2 类别歧义并改动文件名，因此本检查没有将 v2 的校验值或文件名套用到 v3。

### CiF（Cracks in the Foundation）

官方来源：[ibm-research/cif-dataset](https://huggingface.co/datasets/ibm-research/cif-dataset)。
核验固定到仓库 revision `75057c6b0838a24c401c292a7e54a88a4cf28271`。
本地目录：`/home/skye/data/Skye/databases/cif`。

| 划分 | 分片 | 样本 | 无缺陷实例标注的样本 |
|---|---:|---:|---:|
| train_full | 351 | 9096 | 0 |
| val_full | 75 | 1942 | 0 |
| test_full | 72 | 1858 | 0 |
| 合计 | 498 | 12896 | 0 |

这 498 个分片均与官方大小、SHA256 一致，Parquet 标注列可读，类别 ID 属于官方 1–6。
数据包含 Algae、Crack、Net-Crack、Crack with Precipitation、Rust、Spalling。
本地只包含 `full` 三个划分；没有下载官方 `tiled` 三个划分。因此结论是“full 部分完整且一致”，不是已下载官方所有数据。

**发现官方 full 划分存在 53 组跨划分相同图像。**
先检查全部文件名，再读取涉及的 44 个分片内图像数据，确认全部 53 组的嵌入图像字节 SHA256 相同：

| 划分组合 | 相同图像组 |
|---|---:|
| train_full / test_full | 30 |
| train_full / val_full | 16 |
| val_full / test_full | 7 |

这不是本地下载或整理造成的重复，因为对应 Parquet 全部与官方文件一致。后续桥梁异常检测评估需要明确处理这些图像级泄漏候选，并保留官方划分与研究使用划分的映射。
不同 split 还复用了数字 `image_id`，加载时应使用 `(split, image_id)` 或文件名；单独的数字 ID 重复不构成图像重复证据。
`cif_split_overlap.json` 包含全部重复样本的分片、行号及图像 SHA256。其 `annotation_sha256` 计算包括对象 ID，不应据此单独判断几何标注不同。

`cif_audit.json` 中 `official_files_match=true` 表示文件完全一致；`passed=false` 来源于 53 项跨 split 同名/同内容告警，不表示文件损坏。

全分辨率 train 的 9096 张图均有缺陷实例标注，不能直接视作仅正常样本的 UAD 训练集。

### Bridge2893_split_seed42

根据现有协议，此数据来自 CODEBRIM 与 S2DS，按 parent group、seed 42 划分；它不是上述两个公开数据集的官方划分。

| 划分 | 正常图 | 缺陷图 | 合计 | parent groups |
|---|---:|---:|---:|---:|
| train | 1014 | 1018 | 2032 | 1533 |
| val | 144 | 135 | 279 | 219 |
| test | 289 | 293 | 582 | 438 |
| 合计 | 1447 | 1446 | 2893 | 2190 |

- 10 项元数据 SHA256 与 `checksums.sha256` 一致。
- 2893 张图像均可完整解码，图像文件 SHA256 全部不同。
- 无 parent group、sample ID 或相同图像跨 split 重叠。
- 1446 个缺陷掩码可读、非空、与图像尺寸一致，像素值为 0/255。
- 掩码是合并的二值缺陷区域，不是四类分别编码的语义掩码；类别信息另在 manifest 中。
- 仅正常样本训练应读取 train/normal 的 1014 张图，不能把 train 的 2032 张图全用于该训练协议。

## 核验范围与留存

官方元数据快照在 `official/`；完整检查结果在 `dacl_audit.json`、`cif_audit.json`、`cif_split_overlap.json` 和 `bridge2893_audit.json`。
核验脚本位于项目 `setup/`。Parquet 检查依赖 pyarrow 17.0.0 安装在隔离的 `runs/audit-tools` 中，未修改 dinomaly 环境现有依赖。
CiF 图像重复检查针对所有跨 split 同名组；没有声称检测了改名图像、不同编码但同像素图像或近重复场景。
