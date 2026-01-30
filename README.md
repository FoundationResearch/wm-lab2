## wm-lab2 (MineDojo 数据采集)

这个 repo 用于从 **MineDojo** 采集轨迹数据，代码按三层组织：
- **Input layer**: `wm_lab2/inputs/`（动作生成策略 / policy，可替换）
- **Sim layer**: `wm_lab2/sim/`（MineDojo 环境创建 + rollout）
- **Output layer**: `wm_lab2/output/`（写 `video.mp4` + `data.npz` + `manifest.json`）

你主要跑 `data_collector.py`（CLI 入口），通过参数选择不同的 policy。

---

## 安装/环境

安装细节见：`installation_README.md`（Java8 / gym 版本等）。

你本机已经有 conda env：`alexwm2` 的话，直接：

```bash
conda activate alexwm2
python --version
python -c "import minedojo, numpy; print('env_ok')"
```

---

## 快速运行（采 1 个 episode）

### 默认（安全随机，避免 equip air）

```bash
python data_collector.py --num_episodes 1 --max_steps 100 --out_root ./data --policy safe_random
```

### 只允许 WASD (+ 少量 jump)

```bash
python data_collector.py --num_episodes 1 --max_steps 200 --out_root ./data --policy wasd --p_jump 0.05
```

---

## 输出格式

每个 episode 一个文件夹：

`data/episode_<UTC时间戳>_<index>/`
- `manifest.json`: 环境参数 + policy + action space 信息
- `video.mp4`: 观测 RGB 视频（H.264）
- `data.npz`: 对齐的 step 序列
  - `actions`: (T, D) int64
  - `positions`: (T, 3) float32
  - `yaws`: (T,) float32
  - `pitches`: (T,) float32

---

## Policy 怎么自定义（接 agent）

CLI 支持 `--policy custom`，并用 `--policy_entrypoint` 指定一个 **factory**：

```bash
python data_collector.py \
  --num_episodes 1 --max_steps 200 --out_root ./data \
  --policy custom \
  --policy_entrypoint examples.custom_policy:make_policy
```

其中 `examples/custom_policy.py` 里提供了模板：factory 签名必须是：

```python
def make_policy(nvec, noop, rng) -> Policy:
    ...
```

返回对象需要实现：
- `reset(obs) -> None`
- `act(obs) -> np.ndarray`  （动作向量 shape (D,)）

---

## 常用参数速查

- `--task_id`: MineDojo task（默认 `open-ended`）
- `--image_size_hw`: `H,W`（默认 `160,256`）
- `--policy`: `safe_random | wasd | custom`
- `--active_dims`: safe_random 只随机哪些维度（默认 `0,1,2,3,4`）
- `--out_root`: 输出根目录

---

## Data Manager（交互式管理 episode：删除/分类）

进入交互式命令行（推荐在 `alexwm2` 环境里运行）：

```bash
python -m wm_lab2.tools.data_manager --root ./data --recursive
```

常用命令：
- `list` / `list ok` / `list empty` / `list failed`
- `show <episode_id_or_prefix>`
- `rm <episode_id_or_prefix>`
- `rm_empty`（一键删除空 episode）
- `rm_failed`（一键删除失败 episode）
- `mv <episode_id_or_prefix> <category>`（把 episode 移动到 `data/<category>/<episode_id>/` 方便分类）

---

## Post-processing（把 npz 转成其他格式，方便扩展）

把所有 ok episode 的 `actions` 导出成 JSON（写到 `postprocessed/` 子目录）：

```bash
python -m wm_lab2.tools.postprocess --root ./data --only_ok --processor actions_json --out_mode subdir
```

说明：
- `--processor actions_json` 是一个内置 processor；后续想支持别的格式，只要在 `wm_lab2/postprocess/` 新增 processor 并注册即可。


