## Data Collection (MineDojo)

入口脚本：`data_collector.py`

数据结构：每个 episode 一个目录：
`<out_root>/episode_<UTC时间戳>_<index>/`
- `manifest.json`
- `video.mp4`
- `data.npz`（actions/positions/yaws/pitches 与视频帧对齐）

---

## 快速开始（推荐）

### `wasd4hold`（为 hyw 的 4 帧 latent 对齐设计）

特点：**WASD 动作每 4 帧保持不变**，方便后续 `hyw-4average` 做 4 帧 -> 1 latent。

默认行为（当你不显式传 `--out_root` / `--image_size_hw` 时）：
- `out_root` → `./data/wasd4hold`
- `image_size_hw` → `256,256`

运行：

```bash
conda activate alexwm2
python data_collector.py --policy wasd4hold --num_episodes 5 --image_size_hw 256,256 --out_root ./data/wasd4hold --fps 25 --max_steps 125
```

如果你想手动覆盖：

```bash
python data_collector.py --policy wasd4hold --hold_frames 4 --image_size_hw 256,256 --out_root ./data/wasd4hold
```

---

## 所有支持的 policy

- **`safe_random`**：默认策略；从 `no_op()` 出发，仅随机移动/视角维度，避免 inventory/equip 报错
- **`wasd`**：只动前后/左右/（可选）跳跃
- **`wasd4hold`**：`wasd` 版本，但动作会 hold N 帧（默认 4 帧）
- **`wasd12hold`**：`wasd` 版本，但动作会 hold 12 帧（为 12-frame latent 的 postprocess 设计）
- **`custom`**：通过 `--policy_entrypoint pkg.module:factory` 注入自定义策略

自定义策略 factory 签名：

```python
def make_policy(nvec, noop, rng):
    ...
```

返回对象需要实现：
- `reset(obs)`
- `act(obs) -> np.ndarray`

---

## 常用参数

- `--task_id`：默认 `open-ended`
- `--image_size_hw`：`H,W`（MineDojo 用 H,W）
- `--out_root`：输出根目录
- `--seed` / `--fps` / `--max_steps`（默认 `fps=25`，`max_steps=125`）
- `--p_jump`：用于 `wasd/wasd4hold`
- `--hold_frames`：用于 `wasd4hold`


