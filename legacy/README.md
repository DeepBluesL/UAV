# 历史实现与参考代码

当前开发和训练使用根目录的 `project/`。本目录保留旧版本作对照，源码内容未改动；不被 `project` 导入。

| 文件 | 用途 |
| --- | --- |
| `main.py`、`uav_isac_basic.py` | 单 UAV 演示；`main.py` 也可调用旧 C-HAPPO |
| `c_happo_hybrid_beamforming.py` | 旧 C-HAPPO 训练与混合波束实现 |
| `dual_uav_rl.py`、`uav_rl_visualization.py` | 旧双 UAV 实现及绘图工具 |
| `test.py` | 原始物理公式与演示，已拆分迁入 `project`；不是单元测试 |
| `test_*.py` | 上述旧实现的回归测试 |
| `2uav/` | Spinning Up 风格的 `core.py`、`ppo.py` 和工具参考代码 |

## 运行旧版本

从仓库根目录执行以下 **CMD 单行命令**。旧脚本使用同目录导入，请直接按文件路径运行。

```bat
python legacy/main.py --mode basic
python legacy/c_happo_hybrid_beamforming.py --help
python legacy/dual_uav_rl.py --help
python -B -m unittest discover -s legacy -p "test_*.py" -v
```

旧实现的默认产物写入仓库根目录 `output/`，与新实现的 `project/output/` 分开。旧命令 `python main.py` 对应现在的 `python legacy/main.py`；不提供 `python -m legacy.main` 入口。

`2uav/` 只作为源码参考，保留了原有导入问题和 Gym、MPI 等额外依赖，不属于当前可直接运行的主项目。参考代码的 Spinning Up 许可见 [LICENSE-spinningup.txt](../project/LICENSE-spinningup.txt)。
