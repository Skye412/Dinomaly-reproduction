# Dinomaly 复现准备记录

日期：2026-10-13（Asia/Shanghai）

## 服务器与代码

- SSH：`ssh skye-cong`，120.26.32.49:12139，用户 skye，已通过密钥登录验证。
- 服务器代码：`/home/skye/data/Skye/Dinomaly`。
- 官方仓库：https://github.com/guojiajeremy/Dinomaly 。
- 固定提交：`1f252be03a918789b19848f0ca37166e7c28dada`。
- GPU：Tesla V100-PCIE-16GB；论文/作者 README 使用 RTX 3090 24GB。
- 仅修改 `dinomaly_mvtec_uni.py`、`dinomaly_visa_uni.py` 的设备 `cuda:1` → `cuda:0`，补丁保存在服务器 `reproduction/server.patch`。

## Conda 环境

- 名称：`dinomaly`；路径：`/home/skye/miniconda3/envs/dinomaly`。
- Python 3.8.12，PyTorch 1.12.0+cu113，torchvision 0.13.0+cu113。
- 按官方 `requirements.txt` 安装，并补装其遗漏的 `colorama==0.4.6`。
- `pip check`、两个训练入口的导入、CUDA 矩阵运算、27 个类别的数据加载抽查均通过。
- 全部依赖版本与 conda 导出保存在服务器 `reproduction/requirements-lock.txt` 与 `reproduction/environment.yml`。

## 数据检查

| 数据集 | 类别 | 正常训练 | 正常测试 | 异常测试 | 异常掩码 |
|---|---:|---:|---:|---:|---:|
| MVTec AD | 15 | 3629 | 467 | 1258 | 1258 |
| VisA | 12 | 8659 | 962 | 1200 | 1200 |

- 原始目录：`/home/skye/data/Skye/databases/MVTEC` 与 `/home/skye/data/Skye/databases/VISA`。
- 官方 VisA split：https://raw.githubusercontent.com/amazon-science/spot-diff/main/split_csv/1cls.csv 。服务器的 split 与 Dinomaly 自带 split 均与官方逐行一致。
- 官方 split SHA256：`a48557e6033318cb90556f706196bc9d247a776a23ea51aecee5a80dd0332995`。
- VisA 官方 split 所列图像与掩码无缺失、无额外文件；MVTec 每类数量及异常掩码配对检查通过。
- 检查 18633 个图像/掩码文件的格式可读性（PIL verify；PNG 包括 CRC 检查），完全解码掩码，检查 2458 对异常图/掩码的尺寸一致性和掩码非空性。
- 原始目录含额外 `meta.json`；官方训练入口不使用该文件，使用文件夹与官方 CSV 划分。
- 已用作者 `prepare_data/prepare_visa.py` 生成 `/home/skye/data/Skye/databases/VISA_pytorch/1cls`，保留原始目录。
- 转换结果也已验证：10821 张图像的 SHA256 与本地原始图像一致，1200 张掩码与原始掩码的非零区域二值化结果逐像素一致，无额外文件。
- 核验结果是结构、数量、划分及文件完整性一致；没有重新下载官方图像压缩包，不能据此声称每个原始图像字节与官方发布包完全一致。

## 后续训练入口

```bash
ssh skye-cong
source /home/skye/miniconda3/etc/profile.d/conda.sh
conda activate dinomaly
cd /home/skye/data/Skye/Dinomaly
python dinomaly_mvtec_uni.py --data_path /home/skye/data/Skye/databases/MVTEC
python dinomaly_visa_uni.py --data_path /home/skye/data/Skye/databases/VISA_pytorch/1cls
```

本阶段没有启动完整训练。官方默认 ViT-B/14 DINOv2-R，resize 448、crop 392、batch 16、10000 iterations；批量 16 在 V100 16GB 上尚未做完整训练显存测试。预训练 backbone 权重尚未下载，首次加载会访问 Meta 官方下载地址。作者代码中的 checkpoint 保存默认被注释；正式复现前需要启用保存并确认评估/记录策略。

参考：论文第 4.1 节；[Dinomaly 官方 README](https://github.com/guojiajeremy/Dinomaly)；[VisA 官方仓库](https://github.com/amazon-science/spot-diff)；[MVTec 官方说明](https://www.mvtec.com/research-teaching/datasets/mvtec-ad)。
