# izumi2026/

Izumi et al. (2026) "Multi-Channel Communication Algorithm for Multi-Player Multi-Armed Bandits without Collision Sensing" の homogeneous multi-channel 設定実装。

## ファイル

| ファイル | 用途 |
|---------|------|
| `algorithm.py` | `HomogeneousMultiChannelIzumi2026` の初期化と `run()` |
| `phases.py` | `find_multiple_good_arms`, `parallel_virtual_musical_chairs`, `parallel_virtual_number_players` |
| `exploration.py` | `hierarchical_distributed_exploration` |
| `communication.py` | 簡略通信、accept/reject、割当 helper、旧 fallback helper |
| `results.py` | `PlayerStateIzumi` と `phase_durations` への変換 |

通信は初回実装では簡略化しており、通信 payload は直接集約し、通信時間コストだけを `Trace` に記録する。

