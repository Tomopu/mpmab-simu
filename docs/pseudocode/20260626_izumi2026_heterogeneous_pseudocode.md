# Izumi 2026 Heterogeneous Pseudocode: Multi-Channel MPMAB with Collision Sensing

作成日: 2026-06-26

このファイルは `docs/pseudocode/20260510_izumi2026_homogeneous_pseudocode.md` の
homogeneous / no-sensing 擬似コードを、heterogeneous / collision-sensing 設定に
移すための実装用 pseudocode です。

## Source Files

- `docs/pseudocode/20260510_izumi2026_homogeneous_pseudocode.md`
- `docs/pseudocode/20260510_shi2021_beacon_pseudocode.md`
- `papers/(2026) Multi-Channel Communication Algorithm for Multi-Player Multi-Armed Bandits without Collision Sensing/src/data.tex`

## Target Model

- 報酬行列は heterogeneous: `mu[m, k]`。
- collision sensing が有効で、各 player は自分が collision したかどうかを観測できる。
- player が時刻 `t` に arm `k` を pull したときの collision indicator を
  `eta_k(t) in {0, 1}` と書く。
  - `eta_k(t) = 1`: collision あり。
  - `eta_k(t) = 0`: collision なし。
- collision 時は reward は 0 になるが、`eta_k(t)` によって stochastic reward 0 と区別できる。
- 学習フェーズは Shi 2021 BEACON を multi-channel / hierarchical communication に拡張した
  `ParallelBEACON` として扱う。

## Key Simplification From No-Sensing

Izumi 2026 の元問題は no-sensing なので、collision と Bernoulli reward 0 を区別できない。
そのため、以下のような確率的保証が必要になる。

- `FindMultipleGoodArms`:
  - 通信用 arm は reward が正になる確率が十分高い必要がある。
  - `mu_tilde[k]` は arm `k` の平均報酬に対する下界として使われる。
  - 後続の repeated sampling 回数は `mu_tilde_min = min_{k in G} mu_tilde[k]` から
    `tau_rank`, `tau_comm` として決まる。

- `ParallelVirtualMusicalChairs`:
  - player が arm を占有できたかどうかを `r > 0` で推定する。
  - stochastic reward 0 と collision を区別できないため、十分な長さの試行が必要。

- `ParallelVirtualNumberPlayers`:
  - `tau` 回同じ virtual collision test を行い、報酬和 `R = 0` なら collision とみなす。

collision-sensing 設定では、`eta_k(t)` を直接観測できるため、これらは以下に置き換えられる。

- positive-reward good arm は不要。通信チャネルは collision を起こせればよい。
- `r > 0` による無衝突判定は `eta_k(t) = 0` に置き換える。
- `tau` 回の `R == 0` collision 判定は 1 回の `eta_k(t) = 1` に置き換える。
- `tau_rank`, `tau_comm`, `mu_tilde_min` は初期化判定には不要になる。

## Why `mu_tilde` Exists in the Homogeneous No-Sensing Algorithm

homogeneous / no-sensing 版では、player は `eta_k(t)` を観測できない。
観測できるのは reward だけなので、reward 0 が以下のどちらで起きたのかを 1 回では区別できない。

- collision が起きた。
- collision は起きていないが Bernoulli reward がたまたま 0 だった。

そのため、通信用 arm には「無衝突なら十分な確率で reward 1 が出る」という下界が必要になる。
`FindMultipleGoodArms` はこのために `G` と `mu_tilde` を返す。
`mu_tilde_min` が小さいほど、reward 0 が偶然続く確率を小さくするために repeated sampling 回数
`tau` を大きくする必要がある。

heterogeneous / collision-sensing 版では `eta_k(t)` により collision の有無を直接観測するため、
この意味での `mu_tilde` は初期化には不要。実装で `mu_tilde` 風の戻り値を残す場合は、
homogeneous 版と戻り値 shape を合わせるための API 互換用 dummy とみなす。
なお、BEACON/ParallelBEACON の epoch loop 内に出てくる `mu_tilde[k, m]` は
player-arm 平均報酬の量子化・推定値であり、ここで不要になる `FindMultipleGoodArms` の
`mu_tilde[k]` とは別の変数である。

## Index Convention

- 擬似コード本文は 1-based index で書く。
- Python 実装では arm/player/rank slot の内部表現を 0-based に統一する。
- 内部 rank `j` は論文と同じく 1-based とする。

## Algorithm 1: SelectCollisionSensingChannels

No-sensing 版の `FindMultipleGoodArms` を置き換える。

```text
Input:
  K: number of arms
  n: number of communication channels

Output:
  G: communication channel arms, |G| = n
  mu_tilde: dummy lower-bound map, kept only for API compatibility

1. G <- {1, 2, ..., n}
2. For each k in G:
     mu_tilde[k] <- 1
3. Return G, mu_tilde
```

### Rationale

No-sensing 版では、collision を報酬 0 から推定するため、通信用 arm は正報酬を返す
確率が必要だった。collision-sensing 版では、bit 1/0 は `eta_k(t)` で直接読めるため、
通信チャネルの選択に reward lower bound は不要。
`mu_tilde` は homogeneous 版と戻り値を合わせるためにだけ残す。

## Algorithm 2: CollisionSensingParallelVirtualMusicalChairs

No-sensing 版 `ParallelVirtualMusicalChairs` の `r > 0` 判定を
`no collision` 判定に置き換える。

```text
Input:
  K: number of arms
  G = {g_1, ..., g_n}: communication channel arms
  delta: confidence parameter

Output per player:
  s: external rank slot in {1, ..., K}

1. s <- -1
2. For each channel i in {1, ..., n}:
     ell_i <- -1

3. T_retry <- ceil(ln(1/delta))
   For t = 1, ..., ceil(K * T_retry / n):
     if t is the first round of a K-slot block:
       if s = -1:
         F_1 <- empty set
         for i = 1, ..., n:
           draw ell_i uniformly from {1, ..., K} \ F_i
           F_{i+1} <- { ((ell_j + (i - j)) mod K) + 1 : 1 <= j <= i }
       else:
         ell_i <- s for all i

     if there exists i such that ((t + i - 2) mod K) + 1 = ell_i:
       pull channel arm g_i
       observe eta_{g_i}(t)
       if eta_{g_i}(t) = 0 and s = -1:
         s <- ell_i
         ell_j <- s for all j in {1, ..., n}
     else:
       pull an arbitrary arm outside G

4. Return s
```

### Rationale

No-sensing 版は `r > 0` を「collision なし」の proxy として使う。
collision-sensing 版では 1 回の pull で `eta_k(t) = 0` を観測すれば、
その virtual slot は他 player と衝突していないと判断できる。

ただし、複数 player が同じ virtual slot をランダムに選んで衝突した場合は、
衝突した全員が次のブロックで別スロットを再試行する必要がある。

no-sensing 版の total_steps = ceil(K * tau_rank / n) と対称的に、collision-sensing 版は
mu_min=1 を代入して total_steps = ceil(K * T_retry / n) とする。これにより:
- ブロック数 ≈ T_retry / n
- 1 ブロックで n チャネル × 1 スロット試行 = n 回の並列試行
- 合計スロット試行 = T_retry 回

T_retry = ceil(ln(1/delta)) 回のスロット試行機会を確保する（T_retry ブロックではない）。
1 ブロックのみでは試行回数が不足し、未確定 player のフォールバックで rank 重複が生じる。

## Algorithm 3: CollisionSensingParallelVirtualNumberPlayers

No-sensing 版 `ParallelVirtualNumberPlayers` の `tau` 回 sampling と
`R = 0` 判定を、1 回の `eta_k(t)` 判定に置き換える。

```text
Input:
  K: number of arms
  G = {g_1, ..., g_n}: communication channel arms
  s: external rank slot

Output per player:
  M_hat: estimated number of players
  j: internal rank

1. M_hat <- 1
2. j <- 1

3. For h = 1, ..., ceil(2K / n):
     For i = 1, ..., n:
       v <- (h - 1)n + i
       if v > 2K:
         ell_i <- -1
       else if v > 2s:
         ell_i <- ((s + (v - 2s) - 1) mod K) + 1
       else:
         ell_i <- s

     For k = 1, ..., K:
       if there exists i such that ell_i != -1 and ((ell_i + i - 2) mod K) + 1 = k:
         i_tilde <- the matching index
         pull channel arm g_{i_tilde}
         observe eta_{g_{i_tilde}}(t)
         if eta_{g_{i_tilde}}(t) = 1:
           M_hat <- M_hat + 1
           v_collision <- (h - 1)n + i_tilde
           if v_collision <= 2s:
             j <- j + 1
       else:
         pull an arbitrary arm outside G

4. Return M_hat, j
```

### Rationale

No-sensing 版は `tau` 回 pull して `R = 0` なら collision と推定する。
collision-sensing 版では、その virtual test における collision は
`eta_k(t) = 1` で直接観測できる。

## Algorithm 4: ParallelBEACON Initial Sampling

BEACON の初期サンプリングと同じく、各 player-arm pair の初期サンプルを 1 回ずつ作る。

```text
Input:
  K: number of arms
  M: number of players
  j_m: internal rank of player m
  G: communication channel arms

Output:
  T[k, m], R[k, m], samples[k, m]

1. Initialize T[k, m] <- 0, R[k, m] <- 0, samples[k, m] <- empty list

2. For offset = 1, ..., K:
     Each player m pulls arm ((j_m + offset - 2) mod K) + 1
     Observe reward r_m
     Let a_m be the arm pulled by m
     T[a_m, m] <- T[a_m, m] + 1
     R[a_m, m] <- R[a_m, m] + r_m
     append r_m to samples[a_m, m]

3. Return T, R, samples
```

## Algorithm 5: ParallelBEACON Epoch Loop

This is the current simplified implementation of heterogeneous ParallelBEACON.
Communication payload is copied directly and only the communication time cost is charged.

```text
Input:
  T[k, m], R[k, m], samples[k, m]
  G = {g_1, ..., g_n}: communication channel arms
  group(m): group id derived from internal rank j_m
  delta

Repeated for epoch r = 1, 2, ... until horizon:

1. For each arm k and player m:
     p[k, m] <- floor(log2(T[k, m]))

2. Uplink aggregation:
     For each group g:
       For each active arm k:
         group_sum[g, k] <- sum over m in group g of mean(samples[k, m]) * T[k, m]
         group_N[g, k] <- sum over m in group g of T[k, m]
       consume communication steps for follower -> sub-leader

3. Sub-leader to grand-leader:
     consume communication steps for sub-leader -> grand leader

4. Grand leader builds reward estimates:
     For each arm k and player m:
       if m is in grand-leader group:
         mu_tilde[k, m] <- individual estimate of player m on arm k
       else:
         mu_tilde[k, m] <- aggregated group mean for group(m), arm k

5. UCB:
     For each arm k and player m:
       mu_bar[k, m] <- mu_tilde[k, m] + sqrt(3 log(t_r) / 2^(p[k, m] + 1))

6. Matching oracle:
     assignment <- maximum weight matching on mu_bar

7. Downlink:
     consume communication steps for grand leader -> sub-leaders -> followers

8. Exploration:
     p_r <- min_m p[assignment[m], m]
     For t = 1, ..., 2^p_r:
       player m pulls assignment[m]
       update T, R, samples with observed reward
```

## Notes and Open Issues

- `SelectCollisionSensingChannels` changes the meaning of `G`: it is now a set of communication
  channels, not necessarily positive-reward good arms.
- The current ParallelBEACON communication remains simplified: it does not yet use forced-collision
  bit transmission for each payload bit.
- The current group aggregation loses player-level heterogeneity for groups outside the grand
  leader group by using a group mean. For faithful heterogeneous BEACON, player-level estimates
  should be preserved through communication.
- Because collision-sensing initialization no longer depends on reward lower bounds, `tau_rank`,
  `tau_comm`, and `mu_tilde_min` are removed from the initialization path.
