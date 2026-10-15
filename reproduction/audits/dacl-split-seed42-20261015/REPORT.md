# DACL10k_v3_512_split_seed42 数据检查

目录：`/home/skye/data/Skye/databases/DACL10k_v3_512_split_seed42`。
日期：2026-10-15。检查未改动数据，也未启动训练。

## 数据和划分

| 划分 | 正常候选 | 异常图 | 合计 | 来源 |
|---|---:|---:|---:|---|
| train | 413 | 5135 | 5548 | 官方 train 的 80% |
| val | 103 | 1284 | 1387 | 官方 train 的 20% |
| test | 55 | 920 | 975 | 完整官方 validation |
| 合计 | 571 | 7339 | 7910 | 官方公开标注部分 |

所有图像为 512×512。这是一个自定义 seed-42 划分，而不是完整官方原始划分；没有纳入无公开标注的 testdev 或 hidden test。
官方 train 被按内容组、19 类标注存在性和正常候选标志分层成 train/val，原官方 validation 保留为本数据目录的 test。
这不是简单的全数据 70/10/20 随机划分。

## 独立复核范围

检查脚本：`setup/audit_dacl_split.py`。实际检查结果保存在同目录 `audit.json`。
逐项复核：全文件 SHA256 清单与实际目录清单；manifest、parent_mapping 一致性；图像与 JSON 对应关系；源文件/复制文件 SHA256；解码 RGB 像素 SHA256；图像及标注尺寸；标注版本；polygon 坐标有限性；类别集合和异常规则；各 split 实际计数与类别边际；源 ZIP SHA256；跨 split 内容组、JPEG 和 RGB 重叠。

原始 source 目录为 `/home/skye/data/Skye/databases/dacl10k_v3_512`，之前已核验其 512 发布包与官方校验值一致。
标注 JSON 保留源文件原始字节。它内部的 `split` / `imagePath` 仍是原官方元数据，加载时应采用 CSV 的路径及 `assigned_split`，并通过 JSON `imageName` 对应图像，不能按 JSON stem 直接匹配。

## 异常定义和标注格式

文件为 JPEG 图像与原始 polygon JSON，没有直接生成 PNG 二值掩码。
以下 14 类任意一类出现即 `anomaly=1`：

ACrack、Cavity、Crack、Efflorescence、ExposedRebars、Graffiti、Hollowareas、Restformwork、Rockpocket、Rust、Spalling、WConccor、Weathering、Wetspot。

Bearing、Drainage、EJoint、JTape、PEquipment 五类单独出现不判为异常。
因此该数据集的正常候选定义比之前按 13 损伤类统计更严格：新增 WConccor 为异常。
正常候选仅表示没有上述 14 类损伤标注，并不代表人工认证完全健康。

19 类 polygon 可重叠。对二值异常定位，需要仅合并 14 个异常类的 polygon，不应把五个对象类的区域也当异常。
当前 Bridge 数据入口要求 `category`、`mask_path` 和 PNG 掩码，不能直接套用本目录的 `anomaly`、`annotation_path` 和 JSON 标注；运行前应增加对应加载适配。

## 重复与评估限制

内容组是 JPEG SHA256 相同或解码 RGB SHA256 相同所形成的组，不是真实桥梁/拍摄会话分组。
已知一组内容完全相同的两张图 `00304.jpg`、`01861.jpg` 保留在同一 train split，且两份标注不同；两张原图和各自原始标注均保留。
这组样本不会跨 split 泄漏，但不能据此声称已排除近重复图或完成真实桥梁/会话隔离。

测试集有 920/975=94.36% 的异常图。即使全部预测异常，图像 F1 也可达到 97.10%；因此不能仅凭图像 AP/F1 的高数值判断模型有效，更不能与前两轮 1:1 数据的 F1 直接比较。
后续评估需要同时保留图像 AUROC、正常图误报率/特异度、召回率和像素定位指标，并使用验证集确定固定阈值。

仅正常样本训练时应取 train 中的 413 张正常候选，不能用全部 5548 张图进行该训练协议。该训练规模明显小于 Bridge512 的 2211 张正常图。
