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

## Communication Arm Convention

BEACON の Send/Receive プロトコル（Shi 2021 Algorithm 3/4）をそのまま使う。

```text
Send(bits, sender=m, receiver=n):
  c_m: sender m の通信アーム
  c_n: receiver n の通信アーム
  For each bit b in bits:
    if b == 1: m pulls c_n   （c_n に collision → receiver が bit 1 を検出）
    if b == 0: m pulls c_m   （collision なし → receiver が bit 0 を検出）

Receive(n_bits, sender=m, receiver=n):
  For each bit position:
    n pulls c_n
    if collision: received bit = 1
    else:         received bit = 0
```

ParallelBEACON での通信アーム割り当て:

```text
c_m = s_m   for all followers m（Wang 2020 外部 rank = 固有の通信 arm）
c_{sub-leader g} = g_g   for g = 2, ..., n
c_{grand-leader} = g_1
```

これにより、同一グループ内の通信（フォロワー f → sub-leader g）は:
- Receiver (sub-leader g): pulls g_g
- Sender (follower f):     pulls g_g (bit=1) or s_f (bit=0)

異なるグループ g ≠ h では g_g ≠ g_h なので干渉しない。
したがって n グループが同時に uplink Phase A を実行できる。

Sub-leader → grand leader の通信では:
- Receiver (grand leader): pulls g_1
- Sender (sub-leader g):   pulls g_1 (bit=1) or g_g (bit=0)

grand leader は同時に 1 アームしか受信できないため、sub-leader からの
中継は逐次（sequential）になる（後述の Phase B）。

## Algorithm 5a: ParallelBEACON Grand Leader (j = 1)

```text
Input:
  T[k, 0], R[k, 0], samples[k, 0]   （grand leader own data, 0-based index）
  G = {g_1, ..., g_n}: communication channel arms
  group_map: group g -> list of player indices m in group g
  M_hat: estimated number of players

Internal state across epochs:
  prev_p[k, m] <- -1 for all k, m
  prev_q[k, m] <- 0  for all k, m       （累積量子化値）
  mu_tilde[k, m] <- 0.0 for all k, m    （全プレイヤーの推定値）

Repeated for epoch r = 1, 2, ... until horizon:

1. Compute p values:
     For each arm k and player m:
       p[k, m] <- floor(log2(T[k, m]))

   Update own estimate (grand leader, m=0):
     For each arm k where p[k, 0] > prev_p[k, 0]:
       Q <- ceil(1 + p[k, 0] / 2)
       mu_tilde[k, 0] <- mean of first 2^{p[k,0]} samples of arm k (grand leader's own)

2. Uplink Phase A: Follower → group leader（n グループ同時進行）

   n グループが channel arm g_1, ..., g_n を使って並列に通信する。
   Grand leader は g_1 で group 1 内のフォロワーからの送信を受信する。

   For each follower f in group 1, sequentially:
     For each arm k where p[k, f] > prev_p[k, f]:
       Q <- ceil(1 + p[k, f] / 2)
       delta_q <- [received via Receive(Q+1 bits, sender=f, receiver=grand-leader)]
       prev_q[k, f] <- prev_q[k, f] + decode(delta_q)
       mu_tilde[k, f] <- dequantize(prev_q[k, f], Q)

   ※ 同時に group 2..n では sub-leader g が group g 内フォロワーから受信している
      （grand leader は関与しない）

3. Uplink Phase B: Sub-leader → grand leader（逐次）

   Grand leader は g_1 で各 sub-leader から順番に受信する。
   sub-leader ごとに、自分のグループの全プレイヤー分のデータを中継してもらう。

   For sub-leader g = 2, ..., n in order:
     For each player m in group g, sequentially:
       For each arm k where p[k, m] > prev_p[k, m]:
         Q <- ceil(1 + p[k, m] / 2)
         delta_q <- [received via Receive(Q+1 bits, sender=sub-leader_g,
                                          receiver=grand-leader)]
         prev_q[k, m] <- prev_q[k, m] + decode(delta_q)
         mu_tilde[k, m] <- dequantize(prev_q[k, m], Q)

4. Compute UCB index:
     t_r <- current time step
     For each arm k and player m:
       mu_bar[k, m] <- mu_tilde[k, m] + sqrt(3 * ln(t_r) / 2^{p[k,m]+1})

5. Matching oracle:
     assignment <- maximum weight matching on mu_bar
                   (assignment[m] = arm index for player m, 0-based)

6. Downlink Phase A: Grand leader → sub-leaders（逐次）

   For sub-leader g = 2, ..., n in order:
     For each player m in group g:
       Send(int_to_bits(assignment[m], ceil(log2(K))),
            sender=grand-leader, receiver=sub-leader_g)
       （grand leader pulls g_g for bit=1, g_1 for bit=0;  sub-leader g pulls g_g）

7. Downlink Phase B: Grand leader → group 1 followers（逐次）

   For each follower f in group 1:
     Send(int_to_bits(assignment[f], ceil(log2(K))),
          sender=grand-leader, receiver=f)
     （grand leader pulls s_f for bit=1, g_1 for bit=0;  follower f pulls s_f）

8. Signal exploration start （collision pattern on g_1 to synchronize）

9. Exploration:
     p_r <- min_m p[assignment[m], m]
     For t = 1, ..., 2^{p_r}:
       pull assignment[0]   （grand leader's assigned arm）
       update T, R, samples for arm assignment[0]

10. Update prev_p:
      prev_p[k, m] <- p[k, m] for all k, m
```

## Algorithm 5b: ParallelBEACON Sub-Leader (j = 2, ..., n; group = j)

```text
Input:
  T[k, m], R[k, m], samples[k, m]   （sub-leader's own data）
  g_g: own group channel arm
  group_followers: list of follower player indices in own group

Internal state across epochs:
  prev_p[k, f] <- -1 for all k and followers f in own group
  prev_q[k, f] <- 0  for all k, f
  received_deltas: buffer for received updates from followers

Repeated for epoch r = 1, 2, ... until horizon:

1. Update own p:
     p[k, m] <- floor(log2(T[k, m]))

2. Uplink Phase A（parallel with other groups）:
   Sub-leader acts as receiver for own group.

   For each follower f in own group, sequentially:
     For each arm k where p[k, f] > prev_p[k, f]:
       Q <- ceil(1 + p[k, f] / 2)
       delta_q <- [received via Receive(Q+1 bits, sender=f, receiver=sub-leader)]
       Buffer delta_q for arm k, player f

3. Uplink Phase B（sequential, sub-leaders take turns）:
   When it is own group g's turn:
     # Relay own changed estimates
     For each arm k where p[k, m] > prev_p[k, m]:
       Q <- ceil(1 + p[k, m] / 2)
       delta_q_m <- quantize_delta(mu_hat[k, m], prev_q[k, m], Q)
       Send([sign_bit] ++ int_to_bits(|delta_q_m|, Q),
            sender=sub-leader, receiver=grand-leader)
       （sub-leader pulls g_1 for bit=1, g_g for bit=0;  grand leader pulls g_1）

     # Relay buffered follower estimates
     For each follower f in own group, sequentially:
       For each arm k where buffered delta exists:
         Send(buffered_bits[k, f],
              sender=sub-leader, receiver=grand-leader)
         （同上）

4. Wait for Downlink Phase A:
   For each player m in own group:
     assignment[m] <- bits_to_int(Receive(ceil(log2(K)) bits,
                                  sender=grand-leader, receiver=sub-leader))

5. Downlink Phase B（parallel with other groups）:
   For each follower f in own group, sequentially:
     Send(int_to_bits(assignment[f], ceil(log2(K))),
          sender=sub-leader, receiver=f)
     （sub-leader pulls s_f for bit=1, g_g for bit=0;  follower f pulls s_f）

6. Exploration:
     p_r <- [received from grand leader's synchronization signal]
     pull assignment[m] for 2^{p_r} times
     update T, R, samples

7. Update prev_p, prev_q for own and buffered followers.
```

## Algorithm 5c: ParallelBEACON Follower (j > n; group = (j-1) mod n + 1)

```text
Input:
  T[k, m], R[k, m], samples[k, m]
  s_m: own external rank (= own communication arm)
  g_g: group g's channel arm (receive arm of own sub-leader / grand leader)

Internal state across epochs:
  prev_p[k, m] <- -1 for all k
  prev_q[k, m] <- 0  for all k

Repeated for epoch r = 1, 2, ... until horizon:

1. Update own p:
     p[k, m] <- floor(log2(T[k, m]))

2. Uplink Phase A（parallel with other groups）:
   Send changed estimates to own group's leader (sub-leader or grand leader).

   For each arm k where p[k, m] > prev_p[k, m]:
     Q <- ceil(1 + p[k, m] / 2)
     mu_hat[k, m] <- mean of first 2^{p[k,m]} samples of arm k
     new_q <- quantize(mu_hat[k, m], Q)
     delta_q <- new_q - prev_q[k, m]
     prev_q[k, m] <- new_q
     Send([sign_bit(delta_q)] ++ int_to_bits(|delta_q|, Q),
          sender=follower, receiver=group-leader)
     （follower pulls g_g for bit=1, s_m for bit=0;  group-leader pulls g_g）

3. （Idle during Phase B: sub-leaders relay to grand leader）

4. Wait for Downlink:
     assignment[m] <- bits_to_int(Receive(ceil(log2(K)) bits,
                                  sender=group-leader, receiver=follower))
     （follower pulls s_m;  group-leader pulls s_m for bit=1, g_g for bit=0）

5. Exploration:
     p_r <- [from synchronization]
     pull assignment[m] for 2^{p_r} times
     update T, R, samples

6. Update prev_p.
```

## Communication Cost Analysis

エポックごとの通信コスト（time steps）:

```text
Uplink Phase A（n グループ同時進行）:
  Duration = max_{g=1..n} Σ_{f in followers(g)} Σ_{k: p[k,f]>prev} (Q_{k,f} + 1)
  ≈ (M/n) * K * avg_Q  if balanced groups  （n 倍速）

Uplink Phase B（sub-leader → grand leader、逐次）:
  Duration = Σ_{g=2..n} Σ_{m in group(g)} Σ_{k: p[k,m]>prev} (Q_{k,m} + 1)
  ≈ (n-1)/n * M * K * avg_Q  （n=1 では 0、n→∞ では ≈ M * K * avg_Q）

Downlink Phase A（grand leader → sub-leaders、逐次）:
  Duration = (n-1) * ceil(M/n) * ceil(log2(K))
  ≈ M * ceil(log2(K))  （BEACON と同程度）

Downlink Phase B（sub-leaders → followers、n グループ同時進行）:
  Duration = max_g(followers_in_g) * ceil(log2(K))
  ≈ M/n * ceil(log2(K))  （n 倍速）
```

BEACON（n=1）との比較:

| フェーズ | BEACON | ParallelBEACON（n>1） | 速度比 |
|---|---|---|---|
| Uplink A（follower→leader） | M * K * avg_Q | M/n * K * avg_Q | n 倍速 |
| Uplink B（relay、新規追加） | なし | (n-1)/n * M * K * avg_Q | ― |
| Downlink A（leader→follower） | M * ceil(log2(K)) | M * ceil(log2(K)) | 同等 |
| Downlink B（relay、新規追加） | なし | M/n * ceil(log2(K)) | ― |

Phase B（sub-leader relay）は Uplink A の並列化で節約した分を打ち消すため、
**エポック通信コストの合計は BEACON と同程度**になる。

n チャネルの本質的な利点は通信コスト削減ではなく:
1. 初期化フェーズ（ParallelVMC/VNP）の n 倍速
2. 探索フェーズの有効サンプル数増加（n アーム同時 pull による情報収集効率化）

## Notes and Open Issues

- 本疑似コードは BEACON の Send/Receive プロトコルを ParallelBEACON に直接拡張したもの。
  現在の実装（algorithm5_parallel_beacon_epoch.py）は通信ペイロードを直接コピーし
  通信コストのみを近似計算する簡略版であり、以下の点が未実装:
  1. 差分エンコード（prev_p との比較による条件付き送信）
  2. 個人推定値の中継（現在はグループ平均を使用）
  3. Phase A/B の分離とコスト計算
- `SelectCollisionSensingChannels` は G の meaning を変えた（positive-reward arm である必要がない）。
- forced-collision ビット伝送は collision-sensing 環境でのみ有効（eta_k(t) で衝突検出）。
- tau_rank, tau_comm, mu_tilde_min は collision-sensing 初期化では不要。

