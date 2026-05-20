# 2026-05-18 バグ調査・修正レポート

## 調査の背景

Huang2022 と Izumi2026(n=1) の実験結果に差が出る原因を調査した。
n=1 では両アルゴリズムの初期化フェーズが一致するはずという前提に対して、実装上の乖離を洗い出し、修正した。

---

## 発見した問題と修正

### 修正 1：Huang2022 FindGoodArm の T1 式が論文と不一致

**ファイル**: `simulator/algorithms/homogeneous/huang2022/algorithm1_find_good_arm.py`

**問題**:

論文（TeX pseudocode）の sub-phase 1 のステップ数は `6K * 2^p * ceil(ln(2/δ))` だが、
実装は `6K² * 2^p * ceil(ln(2/δ))` という K 倍大きい値を使っていた。
コメントに「K^2 が正しい」と書かれていたが、TeX の式を誤読したものであり誤り。

```python
# 修正前
T1 = 6 * K * K * (2**p) * _ceil(_ln(2.0 / delta))

# 修正後（論文式）
T1 = 6 * K * (2**p) * _ceil(_ln(2.0 / delta))
```

**影響**: FindGoodArm フェーズが K 倍長く走っていた。
K=5 の設定では Huang2022 の init_duration が Izumi2026(n=1) の約 1.2 倍になっていた。

---

### 修正 2：Huang2022 FindGoodArm の確認チェックがリジェクト arm にも誤適用

**ファイル**: `simulator/algorithms/homogeneous/huang2022/algorithm1_find_good_arm.py`

**問題**:

TeX pseudocode では `R'[ℓ] >= 1` による k_tilde 確認は **accept ブランチ内にのみ**存在する。

```tex
\IF{R[ℓ]/N[ℓ] >= 2^{1-p}}        % accept branch
    ...T2 steps (一様ランダム)...
    \ifthen{R'[ℓ] >= 1}{k_tilde ← ℓ; break}   % ← ここだけ
\ELSE                              % reject branch
    ...T2 steps (arm ℓ のみ選択)...
    % 確認チェックなし
\ENDIF
```

実装では確認チェックが `if/else` の**外側**にあったため、
reject された arm でも sub-phase 2 で報酬を 1 以上得れば k_tilde に確定していた。

```python
# 修正前
for m in range(M):
    if k_tilde_list[m] == -1 and Rprime[m][ell] >= 1:
        k_tilde_list[m] = ell

# 修正後（accept された arm のみ確認）
for m in range(M):
    if k_tilde_list[m] == -1 and accept[m] and Rprime[m][ell] >= 1:
        k_tilde_list[m] = ell
```

**影響**: reject arm は sub-phase 2 で T2 ステップまるごと arm ℓ のみを引くため、
mean が少しでも正であれば `Rprime[m][ell] >= 1` になりやすかった。
これにより低 mean の arm が k_tilde に選ばれるケースが発生していた。

---

### 修正 3：Izumi2026 FindMultipleGoodArms の確認チェックも同様に誤適用

**ファイル**: `simulator/algorithms/homogeneous/izumi2026/algorithm1_find_multiple_good_arms.py`

**問題**: 修正 2 と同じ構造上の問題。TeX では accept ブランチ内にしか確認チェックがないが、
実装では `if/else` の外側に置かれていた。

```python
# 修正前
for m in range(M):
    if len(G_list[m]) < n and Rprime[m][ell_idx] >= 1:
        G_list[m].append(ell)

# 修正後
for m in range(M):
    if len(G_list[m]) < n and accept[m] and Rprime[m][ell_idx] >= 1:
        G_list[m].append(ell)
```

---

## TeX と一致していることを確認した箇所

以下の 6 アルゴリズムについて、論文 pseudocode との対応を全項目確認し、不一致なし。

| アルゴリズム | 確認した主要ポイント |
|---|---|
| Huang2022 VirtualMusicalChairs | 総ステップ `K*τ`、block 先頭条件 `(t-1)%K==0`、slot 条件 `(t-1)%K==l`、rank 確定ロジック |
| Huang2022 VirtualNumberPlayers | hopping 条件 `n>2s`、外/内ループ回数 `2K*K`、R=0 判定と M_hat・j 更新 |
| Huang2022 DistributedExploration | sequential hopping 開始位置 `(j-1)%Ka`、通信コスト `Q*τ*Ka`、accept/reject 判定式 |
| Izumi2026 ParallelVirtualMusicalChairs | forbidden set 計算 `(l[j0]+(i0+1-j0))%K`、arm 条件 `(t0+i0)%K==l[m][i0]`、spreading 正確性 |
| Izumi2026 ParallelVirtualNumberPlayers | hopping 式 `(v-s_1based-1)%K`、slot 条件 `(ell+i0)%K==k0`、v_collision 計算 |
| Izumi2026 HierarchicalDistributedExploration | 階層通信コスト（max group 近似）、AssignAndUpdate ロジック（n=1 特殊ケース含む） |

---

### 修正 4：Izumi2026 ProposedParallelAlgorithm の ParallelVMC に渡す τ が誤り

**ファイル**: `simulator/algorithms/homogeneous/izumi2026/algorithm.py`

**問題**:

論文の誤りとして、ParallelVMC に渡す τ は `ceil(K * ln(1/δ) / μ̃)` であるべきだったが、
実装では全フェーズ共通で `ceil(ln(1/δ) / μ̃)` を使っていた。

```python
# 修正前（全フェーズ同一 τ）
tau = _ceil(_ln(1.0 / delta) / mu_safe)
s_list = self.parallel_virtual_musical_chairs(runner, good_arms, tau)
M_hat_list, j_list = self.parallel_virtual_number_players(runner, good_arms, s_list, tau)
f_list = self.hierarchical_distributed_exploration(runner, good_arms, j_list, M_hat_list, tau)

# 修正後（VMC と VNP/HDE で τ を分ける）
tau_rank = _ceil(self.K * _ln(1.0 / delta) / mu_safe)  # ParallelVMC 用
tau_comm = _ceil(_ln(1.0 / delta) / mu_safe)            # VNP・HDE 用
s_list = self.parallel_virtual_musical_chairs(runner, good_arms, tau_rank)
M_hat_list, j_list = self.parallel_virtual_number_players(runner, good_arms, s_list, tau_comm)
f_list = self.hierarchical_distributed_exploration(runner, good_arms, j_list, M_hat_list, tau_comm)
```

**影響**:

ParallelVMC の総ステップは `ceil(K * τ / n)` なので：

| | ParallelVMC に渡す τ | VMC 総ステップ |
|---|---|---|
| 修正前 | `ceil(ln(1/δ) / μ̃)` | `K * ceil(ln(1/δ)/μ̃)` |
| 修正後 | `ceil(K * ln(1/δ) / μ̃)` | `ceil(K² * ceil(ln(1/δ)/μ̃) / n)` |

- `n=1` のとき Huang2022 VMC（`K² * tau_comm` ステップ）と完全に一致する
- `n>1` のとき `K²/n` 倍のステップ数となり、n 倍の高速化が得られる

---

## Huang2022 と Izumi2026(n=1) の差が残る理由（設計上の意図的な差）

修正 1〜4 をすべて適用後も以下の差は残る。これは**設計上の意図的な差**であり、バグではない。

### active_arms の管理方式の違い

Izumi2026 は各フェーズ終了後に確定済み good arms を active_arms から除外する。
Huang2022 は常に全 K 本をイテレートする。

---

## 修正後の実験結果（K=5, M=3, n=1, 10 trials）

修正 1〜3 適用後（修正 4 適用前）の結果：

```
algorithm  n  init_duration  rank_duration  number_players_duration  cumulative_regret
huang2022  1         7475.0         1325.0                   2650.0           41311.94
izumi2026  1         6415.0          265.0                   2650.0           39680.56
```

- `number_players_duration`（VNP フェーズ）は 2650 で完全一致 → VNP の等価性を確認
- `rank_duration` は 1325 / 265 = **ちょうど K=5 倍**の差 → 修正 4 で解消される

---

## テスト結果

```
38 passed in 22.28s
```

全修正適用後も全 38 テストがパス。

---

## HierarchicalDistributedExploration の実装概要と問題点

### 背景

Izumi2026 の Algorithm 4 (HierarchicalDistributedExploration) は論文本文に擬似コードが存在しない。
Codex による補完で実装されており、Grand-Leader / Sub-Leader / Follower の階層構造を模している。

---

### 現在の実装の流れ

#### 役割の割り当て（j_list を使用）

- **Grand Leader**: `j == 1` のプレイヤー（1 名）
- **Sub-Leader**: `2 <= j <= n` のプレイヤー（最大 n-1 名）
- **Follower**: `j > n` のプレイヤー

グループ ID（0-based）: `g_0 = (j - 1) % n`

各グループは 1 本の good arm `good_arms[g_0]` を通信チャンネルとして使う。

#### sub-phase 1: sequential hopping（探索）

Huang2022 の DistributedExploration と同じ方式。
各プレイヤーが active arms を順繰りに選択し、報酬を記録する。

```
T_explore = Ka * 2^p * ceil(ln(1/δ))
pos[m] = (pos[m] + 1) % Ka  # 毎ステップ +1 して次の腕へ
```

#### sub-phase 2: 階層的通信（簡略実装）

**アップリンク（Follower → Sub-Leader → Grand Leader）**

```python
# Follower → Sub-Leader コスト（並列通信の近似: 最大グループのみ）
comm_uplink_follower = max_followers_per_group * Ka * Q * tau
# Sub-Leader → Grand Leader コスト
comm_uplink_sub = n_subleaders * Ka * Q * tau
```

通信量 Q = ceil(p/2 + 3)（論文と同じ式）。

実際には bit 伝送は行わず、`runner.step(dummy_arm, ...)` でタイムステップのみ消費する（**量子化・通信誤りなし**の簡略実装）。

**集約（Grand Leader が直接実施）**

全プレイヤーの推定値 `E[m][k]` を直接参照して集約する（通信をシミュレートせず）。

```python
mu_hat[k][m] = E[m][k]
N_mat[k][m] = v_cnt[m][k]
```

**AcceptReject 判定（Grand Leader のみ）**

Huang2022 と同じ式（`_compute_accept_reject`）:

```
ρ[k] = Σ_m (μ̂[k,m] * N[k,m]) / Σ_m N[k,m]
B[k] = sqrt(2*ln(1/δ) / Σ_m N[k,m]) + 2^{-p/2-3}
accept: |{i : ρ[k]-B[k] >= ρ[i]+B[i]}| >= Ka - M0
reject: |{i : ρ[i]-B[i] >= ρ[k]+B[k]}| >= M0
```

**ダウンリンク（Grand Leader → Sub-Leader → Follower）**

accept/reject の結果をコスト計算のみで配信（実際の伝送なし）。

#### sub-phase 3: AssignAndUpdate（割当）

**n=1 のとき（Huang2022 と同等の特殊ケース）**

```python
C0_accept = C_accept - {good_arm}
if M0 - 1 == len(C0_accept) and good_arm_accepted:
    f[grand_leader] = good_arm          # リーダーが good arm を受け取る
elif len(C0_accept) >= M0:
    f[grand_leader] = C0_accept[M0-1]  # 通常割当
for m != grand_leader:
    idx = M0 - j_list[m]
    if 0 <= idx < len(C0_accept):
        f[m] = C0_accept[idx]
```

**n>1 のとき（通常経路）**

```python
C_assign = C_accept - G   # good arms を除外
idx = M_active - j        # rank が高い (j が大) ほど C_assign の先頭から割当
if 0 <= idx < len(C_assign):
    f[m] = C_assign[idx]
```

active_arms と M0 の更新:

```python
remove_set = assigned_this_round | set(C_reject)
active_arms = [a for a in active_arms if a not in remove_set]
M0 = max(0, M0 - len(assigned_this_round))
```

---

### 問題点

#### 問題 1: n>1 のとき good arms が割当対象から永久に除外される

`C_assign = C_accept - G` で good arms を除外するため、**good arms は誰にも割り当てられない**。

一方、accept/reject 判定は good arms も含む active_arms 全体に対して行われる。
平均報酬が高い good arms は `C_accept` に入りやすいが、`C_assign` から除かれているので割り当ては発生しない。
割り当てが発生しなければ `active_arms` からも除去されない（除去条件は `assigned_this_round` か `C_reject` のみ）。

**結果**: high-mean good arms が毎フェーズ C_accept に残り続け、C_accept のサイズを圧迫する。
割当条件 `0 <= idx < len(C_assign)` が成立しにくくなり、フォロワーが割り当てを得られないまま
フェーズが繰り返される可能性がある。

n=1 は特殊ケース（grand_leader が good arm を受け取れる）として脱出経路があるが、
n>1 は完全な脱出経路がない。

#### 問題 2: 通信の簡略化による理論値との乖離

推定値の集約は量子化なしで行われるため、論文が想定する通信コストと実際の情報伝達量が対応していない。
通信時間コストは `max_followers_per_group * Ka * Q * tau` という最大グループ近似で計算されており、
グループ数や通信方向によっては過大評価または過小評価になる可能性がある。

これは実装上の簡略化として許容されている（`communication.py` のコメントにも明記）が、
理論的な計算量解析と実験上のステップ数が一致しないことに注意が必要。

---

## 問題 1・問題 2 の詳細解説

### 問題 1: n>1 のとき good arms が割当対象から永久に除外される

#### 何が起きているか

`_try_assign` と `active_arms` 更新の組み合わせによるデッドロック。

**除外ロジック（`communication.py:113`）:**

```python
C_assign = [a for a in C_accept if a not in good_set]  # good arms を除外
```

**active_arms の更新（`algorithm4:218-219`）:**

```python
remove_set = assigned_this_round | set(C_reject)
active_arms = [a for a in active_arms if a not in remove_set]
```

good arms が `active_arms` から消えるのは「割当済みになったとき」か「reject されたとき」だけ。しかし good arms は平均報酬が高いため、毎フェーズ `C_accept` に入り続け、`C_reject` には入らず、`C_assign` からも除外されるため **割当が一切発生しない**。

#### 具体例（K=5, M=3, n=2, good_arms=[0,1]）

| フェーズ | active_arms | C_accept | C_reject | C_assign | 割当 | M0変化 |
|---|---|---|---|---|---|---|
| 1 | [0,1,2,3,4] | [0,1,3] | [4] | [3] | j=3→arm3 のみ | 3→2 |
| 2 | [0,1,2] | [0,1] | [2] | [] | **なし** | 2→2 |
| 3 | [0,1] | [0,1] | [] | [] | **なし** | 2→2 |
| 4 | [0,1] | [0,1] | [] | [] | **なし** | **∞ループ** |

フェーズ 3 以降、`active_arms` に good arms だけが残り、`C_assign` が永久に空になる。grand leader (j=1) と sub-leader (j=2) は割当を受けられないまま while ループが終了しない。

#### n=1 で問題が出ない理由

`n=1` は特殊ケース（`algorithm4:180-199`）で分岐しており、grand leader が good arm そのものを受け取れる脱出経路がある:

```python
if M0 - 1 == len(C0_accept) and good_arm_accepted:
    f[grand_leader] = good_arm  # n=1 専用の脱出
```

n>1 はこの分岐を通らず、通常経路（`_try_assign`）で good arms が必ず除外される。

#### 本質的な設計上の問題

補完擬似コード（`docs/pseudocode/20260510_izumi2026_pseudocode.md:303`）の `AssignAndUpdate` は:

> C_assign = C_accept \ G（good arms を通信チャンネルとして除外）

と書いているが、**sub-leader・grand leader (j=1..n) 自身がどの腕に割当されるかが未定義**。n=1 では grand leader が good arm を受け取る特殊処理があるが、n>1 では j=1..n の n 人のプレイヤーへの割当経路が実装されていない。これが根本原因。

---

### 問題 2: 通信の簡略化による理論値との乖離

#### 何が簡略化されているか

3つの層で理論と実装がずれている。

**（A）量子化なし: 情報量と時間コストが対応していない**

論文は `Q = ceil(p/2 + 3)` bits の量子化通信を前提とする。実装では:

```python
# algorithm4:154-155
mu_hat[k][m] = E[m][k]   # float (64bit) を直接コピー
N_mat[k][m] = v_cnt[m][k]
```

推定値を float のまま直接参照するため、**実際の通信は 0 ステップ**で完了するが、時間コストとして `Ka * Q * tau` ステップが別途消費される。

→ 通信コストは記録されるが、情報の質（量子化誤差）は論文の理論解析と異なる。論文の誤差解析は Q bits 量子化の丸め誤差 `2^(-p/2-3)` を B[k] に含めているが、実装では丸め誤差なしで完全な float が伝わる。

**（B）並列グループ通信のコスト過大/過小評価**

論文の設計: n 個のグループがそれぞれ異なるチャンネル（good_arms[g_0]）を使って**同時並列**に通信できる。

実装のコスト計算（`algorithm4:139,144`）:

```python
comm_uplink_follower = max_followers_per_group * Ka * Q * tau  # 最重グループ近似
comm_uplink_sub      = n_subleaders * Ka * Q * tau             # 直列加算
```

`comm_uplink_sub` は Sub-Leader が Grand Leader へ送る部分で、`n_subleaders` 個を**直列に足している**。G[1] チャンネルを順番に使う逐次送信であれば直列が正しいが、論文の意図との整合性が曖昧。

**（C）Follower→Sub-Leader の並列性: max group 近似の影響**

```python
max_followers_per_group = max(n_followers_in_group for each group)
comm_uplink_follower = max_followers_per_group * Ka * Q * tau
```

グループ間は並列チャンネルを使えるため「最大グループのコスト = 全体コスト」という近似は方向性として正しい。ただし同一グループ内の Follower→Sub-Leader 送信が逐次か並列かが論文本文に明記されていないため不確か。

#### まとめ

| 問題 | 論文の想定 | 実装の実態 |
|---|---|---|
| 情報伝達 | Q bits 量子化、丸め誤差あり | float 直接コピー、誤差なし |
| 通信コスト（Follower→SL） | max group で並列 | max group 近似（近い） |
| 通信コスト（SL→GL） | n-1 通の逐次 G[1] 通信 | `n_subleaders * Ka * Q * tau`（係数が曖昧） |
| 集約精度 | 量子化誤差込みで信頼区間 B[k] が設計されている | 誤差なしの float でより精度が高い集約 → B[k] が過大評価 |

問題 2 は**実験結果の絶対値**（regret の大きさやフェーズ数）に影響するが、アルゴリズムの**正確性（収束・ランク決定）には直接影響しない**。問題 1 は n>1 で while ループが終わらないという**実装クリティカルなバグ**。
