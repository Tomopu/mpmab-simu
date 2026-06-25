# 擬似コード・実装レビュー (2026-06-26)

対象ファイル:

- `docs/pseudocode/20260510_izumi2026_homogeneous_pseudocode.md`
- `docs/pseudocode/20260626_izumi2026_heterogeneous_pseudocode.md`
- `simulator/algorithms/homogeneous/izumi2026/`
- `simulator/algorithms/heterogeneous/izumi2026/`

---

## テスト結果

`pytest tests/` で全 **80 件 PASS**。実行時エラーなし。

---

## Homogeneous 擬似コードの評価

問題なし。5 アルゴリズム（FindMultipleGoodArms・ParallelVirtualMusicalChairs・
ParallelVirtualNumberPlayers・HierarchicalDistributedExploration・
ProposedParallelAlgorithm）と補完擬似コード（ComGrandLeader / ComSubLeader /
ComFollower / AcceptReject / AssignAndUpdate）が正確に記述されている。

τ_rank / τ_comm の使い分けと論文誤りに関する注記も適切。

---

## Homogeneous 実装の評価

各アルゴリズムが擬似コードと対応していることを確認した。

### Algorithm 1 (FindMultipleGoodArms)

| 擬似コードの変数 | 実装 | 一致 |
|---|---|---|
| T1 = 6\|K\|·2^p·ceil(ln(2n/δ)) | `6 * Ka * (2**p) * ceil(ln(2n/delta))` | ✓ |
| T2 = \|K\|·2^p·ceil(ln(2n/δ)) | `Ka * (2**p) * ceil(ln(2n/delta))` | ✓ |
| accept 閾値 2^{1-p} | `thr = 2.0 ** (1 - p)` | ✓ |
| R'[ℓ] >= 1 → G に追加 | `Rprime[m][ell_idx] >= 1 → G_list append` | ✓ |

### Algorithm 2 (ParallelVirtualMusicalChairs)

| 擬似コードの条件 | 実装 | 一致 |
|---|---|---|
| 総ステップ ceil(K·τ/n) | `_ceil(K * tau / n)` | ✓ |
| ブロック先頭 t mod K == 1 | `t0 % K == 0` (0-based) | ✓ |
| スプレッド式 F_{i+1} (1-based) | `{(l[j0] + (i0+1-j0)) % K}` (0-based) | ✓ |
| pull 条件 ((t+i-2) mod K)+1 = ℓ_i | `(t0 + i0) % K == l[m][i0]` | ✓ |
| r > 0 → rank 確定 | `rewards[m] > 0 and s_list[m] == -1` | ✓ |

### Algorithm 3 (ParallelVirtualNumberPlayers)

| 擬似コードの条件 | 実装 | 一致 |
|---|---|---|
| h = 1..ceil(2K/n) | `H = _ceil(2*K/n)` | ✓ |
| hopping 式 ((s+(v-2s)-1) mod K)+1 | `ell = (v - s_1based - 1) % K` | ✓ |
| slot 条件 ((ℓ_i-1+i-1) mod K)+1 = k | `(ell + i0) % K == k_0` | ✓ |
| τ 回後 R==0 → collision | `R_sum == 0 → M_hat += 1` | ✓ |
| v <= 2s → j += 1 | `v_collision <= 2 * s_1based` | ✓ |

### Algorithm 4 (HierarchicalDistributedExploration)

擬似コードに沿って実装されており、補完擬似コードの「Implementation Notes」に
記載された簡略化（直接集約・通信コストのみ記録）がコード内コメントで明示されている。
AssignAndUpdate の active_arms 更新が擬似コードと意図的に異なる点もコメントで説明済み。

### Algorithm 5 (ProposedParallelAlgorithm)

τ_rank を ParallelVMC に、τ_comm を ParallelVNP・HDE に渡すという
論文誤りへの対処が正しく実装されている。

---

## Heterogeneous 擬似コードの問題点

### [Bug] Algorithm 2 のスプレッド式に -1 の誤りがある

**場所**: `docs/pseudocode/20260626_izumi2026_heterogeneous_pseudocode.md`
Algorithm 2 (CollisionSensingParallelVirtualMusicalChairs) の forbidden set 更新式。

**現在の擬似コード**:
```text
F_{i+1} <- { ((ell_j + (i - j) - 1) mod K) + 1 : 1 <= j <= i }
```

**正しい式** (homogeneous 擬似コードおよび数学的導出と一致):
```text
F_{i+1} <- { ((ell_j + (i - j)) mod K) + 1 : 1 <= j <= i }
```

**根拠**:

arm i が時刻 t で引かれる条件は `((t + i - 2) mod K) + 1 = ℓ_i`。
arm j と arm i+1 が同一ブロック内で衝突しない条件は、
pull タイムが異なること、すなわち `ℓ_j - j ≢ ℓ_{i+1} - (i+1) (mod K)`。
整理すると `ℓ_{i+1} ≢ (ℓ_j + (i+1-j)) (mod K)`。
1-based スロット値としての禁止値は `((ℓ_j + i - j) mod K) + 1`。
これは homogeneous 擬似コードの式と一致し、`-1` は不要。

**実装への影響**: 両実装（homogeneous / heterogeneous）は正しい式で書かれているため、
動作への影響はない。擬似コードのみ修正が必要。

---

## Heterogeneous 実装の問題点

### [Bug A] `Dict` が import されていない (`algorithm.py` L35)

**場所**: `simulator/algorithms/heterogeneous/izumi2026/algorithm.py`

`from __future__ import annotations` のおかげで実行時エラーにはならないが、
`Dict` が `typing` から import されておらず、型チェッカーが未定義と報告する。

```python
# 現在 (L35)
from typing import List, Optional

# run() の戻り値型 (L102)
def run(self, runner: HeterogeneousRunner) -> Dict[str, object]:
```

**修正**: `Dict` を import するか、Python 3.9+ の `dict[str, object]` に変更する。

### [Bug B] `n_followers_g` がダウンリンクで最後グループの値になっている (`algorithm5_parallel_beacon_epoch.py` L283)

**場所**: `simulator/algorithms/heterogeneous/izumi2026/algorithm5_parallel_beacon_epoch.py`

アップリンクのループ内で `n_followers_g` がグループごとに上書きされ、ループ終了後は
`g = n`（最後のグループ）の follower 数になる。続くダウンリンクコスト計算がその値を参照しており、
グループサイズが不均一な場合（M が n の倍数でない場合）に通信コストが不正確になる。

```python
for g in range(1, n + 1):
    ...
    n_followers_g = max(0, len(group_pids) - 1)  # ← ループのたびに上書き
    ...

# ループ後: n_followers_g は g=n の値のみ
comm_steps_down = (n_subleaders + n_followers_g) * arm_bits_needed  # ← 不正確
```

**修正**: ループ外で `max_followers_per_group` を集計し、それを使う。

### [Bug C] `prev_p` が未使用のデッドコード (`algorithm5_parallel_beacon_epoch.py` L190, L306)

**場所**: `simulator/algorithms/heterogeneous/izumi2026/algorithm5_parallel_beacon_epoch.py`

`prev_p` が初期化・更新されるが、どこでも読まれていない。

```python
prev_p = [[-1] * M for _ in range(K)]  # L190: 初期化（未使用）
...
prev_p = [row[:] for row in curr_p]    # L306: 更新（未使用）
```

**修正**: 両行を削除する。

---

## 修正優先度まとめ

| # | 種別 | 場所 | 優先度 |
|---|------|------|--------|
| 1 | 擬似コードの式誤り | heterogeneous 擬似コード Algorithm 2 | **高** |
| 2 | `Dict` import 漏れ | heterogeneous/izumi2026/algorithm.py L35 | **中** |
| 3 | `n_followers_g` ステール変数 | algorithm5_parallel_beacon_epoch.py L283 | **中** |
| 4 | `prev_p` デッドコード | algorithm5_parallel_beacon_epoch.py L190, L306 | **低** |
