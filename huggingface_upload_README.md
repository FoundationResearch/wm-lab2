# 上传数据集到 Hugging Face（100GB+）

本仓库会把 MineDojo 轨迹数据写到例如 `data/wasd12holdrandview/`。
当数据达到 100GB+ 且包含大量小文件时，推荐**先打包成（压缩的）tar 包再分批上传**，会明显更稳、更快，也更容易断点续传。

下面以你提到的 `alpha` / `beta` 两次 run 为例：你已经把它们分别放在两个目录里：
- alpha: `data/wasd12holdrandview/`
- beta: `data/wasd12holdrandview2/`

我们直接把**整个目录**分别打成两个压缩包再上传。

## 0) 准备工作

- 注册 Hugging Face 账号（或组织）
- 确认你有足够的 dataset 存储配额

安装 Hub 客户端（可选开启更快上传）：

```bash
pip install -U "huggingface_hub[hf_transfer]"
```

设置 Hugging Face Token（推荐用环境变量；不需要 `huggingface-cli login`）：

```bash
export HF_TOKEN="hf_xxx你的tokenxxx"
```

可选：开启 `hf_transfer`（通常更快）：

```bash
export HUGGINGFACE_HUB_ENABLE_HF_TRANSFER=1
```

## 1) 选择你的 dataset repo（你已创建好）

本项目使用的 dataset repo 是：

```bash
alexzms/FastvideoWorldModel-MC
```

所以你可以直接上传文件到这个 repo（不需要再 `repo create`）。

## 2) 安装打包工具（推荐 zstd）

`tar + zstd` 对大量小文件非常友好。

```bash
sudo apt-get update
sudo apt-get install -y zstd
```

## 3) 按目录分别打包（alpha / beta）

以下命令会在 `data/` 下生成两个压缩包（并且会排除 `.minedojo_env_init.lock` 这种锁文件）：
- `data/wasd12holdrandview_alpha.tar.zst`
- `data/wasd12holdrandview_beta.tar.zst`

```bash
# alpha: 打包整个目录 data/wasd12holdrandview
tar -C data --exclude=".minedojo_env_init.lock" -cf - wasd12holdrandview \
  | zstd -T0 -19 -o data/wasd12holdrandview_alpha.tar.zst

# beta: 打包整个目录 data/wasd12holdrandview2
tar -C data --exclude=".minedojo_env_init.lock" -cf - wasd12holdrandview2 \
  | zstd -T0 -19 -o data/wasd12holdrandview_beta.tar.zst
```

可选：快速校验压缩包是否损坏：

```bash
zstd -t data/wasd12holdrandview_alpha.tar.zst
zstd -t data/wasd12holdrandview_beta.tar.zst
```

## 4) 上传压缩包到 Hugging Face

把两个压缩包上传到 dataset repo 的 `archives/` 目录下：

```bash
huggingface-cli upload alexzms/FastvideoWorldModel-MC \
  data/wasd12holdrandview_alpha.tar.zst \
  --repo-type dataset \
  --path-in-repo archives/wasd12holdrandview_alpha.tar.zst \
  --commit-message "Upload alpha archive"

huggingface-cli upload alexzms/FastvideoWorldModel-MC \
  data/wasd12holdrandview_beta.tar.zst \
  --repo-type dataset \
  --path-in-repo archives/wasd12holdrandview_beta.tar.zst \
  --commit-message "Upload beta archive"
```

## 5) 强烈建议：不要只做“一个超大包”，最好做分片（更稳）

如果 `alpha` 或 `beta` 单个压缩包依然很大（比如几十 GB），建议把每次 run 再切成多个分片（例如每片 5–10GB），再逐片上传：
- 上传失败更容易重试
- 断点恢复成本更低

一种简单策略是先把 `episode_*_alpha` 拷贝/移动到临时目录，然后用你熟悉的方式分桶（例如每 N 个 episode 一个 tar）。

## 6) 数据集说明（dataset card）

建议在 Hugging Face dataset repo 根目录放一个 `README.md`，说明：
- 每个压缩包里包含什么结构（`episode_*/{manifest.json,data.npz,video.mp4,...}`）
- 采集参数（fps、分辨率、max_steps、policy 等）
- 许可证/使用限制



