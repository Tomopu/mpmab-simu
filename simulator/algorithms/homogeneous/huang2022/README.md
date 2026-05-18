# huang2022/

Huang et al. (2022) "Towards Optimal Algorithms for Multi-Player Bandits without Collision Sensing Information" の homogeneous 設定実装。

## ファイル

| ファイル | 用途 |
|---------|------|
| `algorithm.py` | `HomogeneousHuang2022` の初期化と `run()` |
| `phases.py` | `find_good_arm`, `virtual_musical_chairs`, `virtual_number_players` |
| `exploration.py` | `distributed_exploration` |
| `communication.py` | 簡略通信、accept/reject、旧 fallback helper |
| `results.py` | `PlayerState` と `phase_durations` への変換 |

forced-collision bit 通信は完全再現ではなく、推定値の直接集約と通信時間コストの消費で近似している。

