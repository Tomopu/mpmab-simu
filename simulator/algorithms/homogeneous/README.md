# homogeneous/

Homogeneous 設定の MPMAB アルゴリズム実装を置くディレクトリ。

## 構成

| パス | 用途 |
|------|------|
| `huang2022/` | Huang et al. (2022) の homogeneous no-sensing アルゴリズム |
| `izumi2026/` | Izumi et al. (2026) の homogeneous multi-channel アルゴリズム |
| `states.py` | アルゴリズム結果を表す `PlayerState` dataclass |
| `math_helpers.py` | 共通の数値 helper |

各論文ディレクトリは `algorithm.py` を公開入口にし、初期化フェーズ、探索フェーズ、通信 helper、結果変換を分けている。

