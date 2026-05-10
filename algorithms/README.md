# algorithms/

MPMAB シミュレーターのアルゴリズム実装モジュール。

論文単位でクラスを追加する。各クラスは `BaseAlgorithm` を継承し、`run(runner)` メソッドを実装する。

## 役割

- 論文のアルゴリズムを homogeneous / heterogeneous 設定ごとに実装する。
- no-sensing 制約を守る（collision flag はアルゴリズムの意思決定に使わない）。
- 内部 index は 0-based に統一し、論文の 1-based との対応はコメントで明記する。

## 主要ファイル

| ファイル | 内容 |
|---------|------|
| `__init__.py` | `HomogeneousHuang2022` をエクスポート |
| `base.py` | `BaseAlgorithm` 抽象基底クラス |
| `homogeneous_huang2022.py` | Huang et al. (2022) の homogeneous 設定アルゴリズム |

### `BaseAlgorithm`

```python
class BaseAlgorithm(ABC):
    @abstractmethod
    def run(self, runner: Runner, **kwargs) -> dict:
        ...
```

### `HomogeneousHuang2022`

```python
algo = HomogeneousHuang2022(K=5, M=2, delta=1e-5, seed=42)
result = algo.run(runner)
player_states = result["player_states"]  # List[PlayerState]
```

論文 Huang et al. (2022) "Towards Optimal Algorithms for Multi-Player Bandits without Collision Sensing Information" の homogeneous 設定実装。

- `find_good_arm(runner)` — Algorithm 1: フェーズ倍増で good arm を探索。
- `virtual_musical_chairs(runner, k_tilde, tau)` — Algorithm 2: 外部 rank を割り当て。
- `virtual_number_players(runner, k_tilde, s_list, tau)` — Algorithm 3: プレイヤー数と内部 rank を推定。
- `distributed_exploration(runner, k_tilde, j_list, M_hat_list, tau)` — Algorithm 4: UCB ベースの arm 割当。
- `run(runner)` — Algorithm 5: 各フェーズを順次実行するエントリーポイント。

## 今後追加予定

| クラス名 | 論文 | 設定 |
|---------|------|------|
| `HomogeneousMultiChannelIzumi2026` | Izumi et al. (2026) | homogeneous, n チャンネル拡張 |
| `HeterogeneousShiBeacon2021` | Shi et al. (2021) | heterogeneous, beacon 方式 |
| `HeterogeneousWangOrthogonalization2020` | Wang et al. (2020) | heterogeneous, 直交化方式 |
