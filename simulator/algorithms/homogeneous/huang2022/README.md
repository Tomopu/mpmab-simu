# huang2022/

Huang et al. (2022) "Towards Optimal Algorithms for Multi-Player Bandits without Collision Sensing Information" の homogeneous 設定実装。

## ファイル

| ファイル | 用途 |
|---------|------|
| `algorithm.py` | `HomogeneousHuang2022` の初期化と `run()` |
| `algorithm1_find_good_arm.py` | Algorithm 1: `find_good_arm` |
| `algorithm2_virtual_musical_chairs.py` | Algorithm 2: `virtual_musical_chairs` |
| `algorithm3_virtual_number_players.py` | Algorithm 3: `virtual_number_players` |
| `algorithm4_distributed_exploration.py` | Algorithm 4: `distributed_exploration` |
| `communication.py` | 簡略通信と accept/reject helper |
| `results.py` | `PlayerState` と `phase_durations` への変換 |

forced-collision bit 通信は完全再現ではなく、推定値の直接集約と通信時間コストの消費で近似している。
