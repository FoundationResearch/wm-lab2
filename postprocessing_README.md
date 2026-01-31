## Post-processing

入口脚本：`python -m wm_lab2.tools.postprocess`

它会扫描 `--root` 下的 `episode_*` 目录（支持 `--recursive`），读取：
- `manifest.json`
- `data.npz`
-（可选）`video.mp4`

并把派生结果写到 episode 目录或 `postprocessed/` 子目录（由 `--out_mode` 控制）。

---

## 通用用法

列出可用 processor（见 `wm_lab2/postprocess/registry.py`）：

- 当前支持：
  - `actions_json`
  - `hyw-4average`
  - `mg12average`

执行：

```bash
conda activate alexwm2
python -m wm_lab2.tools.postprocess --root ./data --recursive --only_ok --processor <name> --out_mode subdir
```

参数：
- `--only_ok`：只处理 data manager 判定为 ok 的 episode
- `--out_mode inplace|subdir`：
  - `inplace`：写到 episode 根目录
  - `subdir`：写到 `episode_*/<processor_name>/`
- `--overwrite`：覆盖已有输出

---

## Processor: `actions_json`

用途：把 `data.npz` 里的 `actions` 导出为 JSON，便于直接读。

输出文件：
- `actions.json`（写在 `postprocessed/` 或 episode 根目录，取决于 `--out_mode`）

格式：
- `episode_id`
- `action_nvec`（来自 manifest）
- `actions`：二维数组，shape `(T, D)`

---

## Processor: `hyw-4average`

用途：把 **4 帧合成 1 个 latent**（每 4 帧做 majority voting），并输出：
- 每个 latent 的离散动作（move/view）
- 每个 latent 的相机内外参（K + w2c/extrinsic）

输入：
- `data.npz`：需要 `actions/positions/yaws/pitches`
- `manifest.json`：用于取 `image_size_hw`

latent 定义：
- latent 0 对应 frames `[0,1,2,3]`
- latent 1 对应 frames `[4,5,6,7]`
- …

输出文件：
- `hyw-4average_actions.json`
- `hyw-4average_camera.json`

### `hyw-4average_actions.json` 格式

key 是 latent index（字符串），value：

- `move_action`：`""/W/A/S/D/WA/WD/SA/SD`
- `view_action`：`""/LR/LL/LU/LD`

其中 `view_action` 的解析规则：
- 对每个 latent，取 yaw/pitch 的净变化（首帧→末帧）
- 若 yaw 与 pitch 同时变化，选绝对变化更大的轴
- yaw：`dyaw>0 => LR`，`dyaw<0 => LL`
- pitch：`dpitch>0 => LD`（看下），`dpitch<0 => LU`（看上）

### `hyw-4average_camera.json` 格式

key 是 latent index（字符串），value：
- `intrinsic` / `K`：3x3
- `w2c` / `extrinsic`：4x4

注意：
- `K` 目前用 `image_size_hw` + 默认 `fov=70°` 推导
- `w2c` 使用每个 latent 的代表 pose（默认用该组最后一帧的 pos/yaw/pitch）

---

## Processor: `mg12average`

用途：把 **12 帧合成 1 个 latent**（每 12 帧做 majority voting），输出 move/view + 相机参数。

输出目录（推荐 `--out_mode subdir`）：
- `data/<policy>/<episode>/mg12average/`

输出文件：
- `mg12average_actions.json`
- `mg12average_camera.json`

latent 定义：
- latent 0 对应 frames `[0..11]`
- latent 1 对应 frames `[12..23]`
- …


