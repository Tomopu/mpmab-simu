# BEACON / ParallelBEACON Communication Notes

作成日: 2026-06-29

このメモは、Shi et al. (2021) BEACON と Izumi 2026 ParallelBEACON の通信コストに関する議論を整理したもの。

## 1. BEACON が量子化して送るもの

BEACON では、follower が leader に報酬サンプル列そのものを送るわけではない。

各 follower `m` は epoch `r` で各 arm `k` について、

```text
p^r_{k,m} = floor(log2(T^r_{k,m}))
```

を計算する。ここで `T^r_{k,m}` は player `m` が arm `k` を探索した回数。

`p^r_{k,m}` が前 epoch から増えた場合だけ、follower は arm `k` の最初の

```text
2^{p^r_{k,m}}
```

個の探索サンプルから標本平均

```text
mu_hat^r_{k,m}
```

を計算する。

この標本平均を

```text
Q^r_{k,m} = ceil(1 + p^r_{k,m} / 2)
```

bits で量子化し、量子化済み推定値 `mu_tilde^r_{k,m}` を作る。

ただし通信で送るのは `mu_tilde^r_{k,m}` そのものではなく、前 epoch との差分

```text
delta_tilde^r_{k,m}
  = mu_tilde^r_{k,m} - mu_tilde^{r-1}_{k,m}
```

である。

実装上は、おおよそ次の bit string を送る。

```text
sign bit + Q bits for |delta|
```

したがって、更新された `(k, m)` ごとに必要な通信量はおおよそ

```text
Q + 1 bits
```

である。

leader は受信した差分を足して、

```text
mu_tilde^r_{k,m}
  = mu_tilde^{r-1}_{k,m} + delta_tilde^r_{k,m}
```

を復元する。

その後、leader は

```text
mu_bar^r_{k,m}
  = mu_tilde^r_{k,m}
    + sqrt(3 ln(t_r) / 2^{p^r_{k,m}+1})
```

を作り、`mu_bar` 行列を Oracle に入れて matching assignment を決める。

leader から follower への downlink では、標本平均ではなく、その epoch で follower `m` が引くべき arm

```text
s^r_m
```

を送る。

## 2. BEACON の通信はすでに効率化されている

BEACON の重要な工夫は、次を送らないことにある。

- 生の報酬サンプル列
- 全履歴
- 毎 epoch の全 player-arm 推定値

代わりに送るのは、更新が必要な player-arm pair に対する

```text
標本平均報酬の量子化差分
```

だけである。

このため、BEACON はすでに通信量をかなり削っている。

## 3. ParallelBEACON で n を増やしても改善が出にくい理由

例として `M=8`, `K=20`, `n=2`, 平均 bit 長 `Qbar ~= 3` を考える。

### BEACON

leader 以外の 7 人が leader に直接送る。

```text
cost ~= 7 * K * Qbar
     ~= 7 * 20 * 3
     = 420 bits
```

### ParallelBEACON

2 グループに分ける。

```text
group 1: grand leader + followers 3 人
group 2: sub-leader  + followers 3 人
```

Phase A では各グループ内で follower から group leader へ並列送信する。

```text
group 1: 3 * 20 * 3 = 180 bits
group 2: 3 * 20 * 3 = 180 bits

Phase A duration = max(180, 180) = 180
```

Phase A だけ見れば 2 倍速に見える。

しかし Phase B では、sub-leader が group 2 の全員分を grand leader に中継する。

```text
sub-leader own data + followers 3 人
= 4 players

Phase B cost = 4 * 20 * 3 = 240 bits
```

合計は

```text
Phase A + Phase B = 180 + 240 = 420 bits
```

となり、BEACON と同程度になる。

一般化すると、概算では

| Phase | Cost |
| --- | ---: |
| Phase A, parallel group uplink | `(M/n) * K * Qbar` |
| Phase B, sequential relay | `((n-1)/n) * M * K * Qbar` |
| Total | `[1/n + (n-1)/n] * M * K * Qbar = M * K * Qbar` |

したがって、uplink 通信コストは `n` によらずほぼ一定になる。

Phase A の並列化で節約した分を、Phase B の中継が回収してしまう。

## 4. n の効果が出る場所

ParallelBEACON の `n` の効果が見えやすいのは、主に次の部分である。

- 初期化の高速化
- initial sampling で複数 player が異なる arm を引くことによるカバレッジ

ただし、たとえば `T=100000` に対して初期化が `1100 -> 207 steps` に短縮されても、節約は約 900 steps で全体の 0.9% 程度である。

このスケールでは累積リグレットに大きな差が出にくい。

差を見たいなら、初期化節約が horizon に対して無視できない設定にする必要がある。

目安:

```text
initialization saving / T >= 0.05
```

程度から差が見えやすくなる。

試す設定例:

```text
M = 32, K = 100, n = 4 or 8,  T = 20000..50000
M = 64, K = 200, n = 8 or 16, T = 20000..50000
```

ただし、これは uplink 通信の本質的改善ではなく、初期化と sampling coverage の効果である。

## 5. 単純な group aggregate 改善が難しい理由

一見すると、sub-leader が group 内の情報を集約してから grand leader に送れば通信量を減らせそうに見える。

たとえば homogeneous MPMAB なら、group 内で

```text
N_group(k) = sum_i N_i(k)
S_group(k) = sum_i S_i(k)
```

のような集約統計を送る設計が考えられる。

この場合、通信量は概算で

```text
(M/n + n) * K
```

まで落とせる可能性がある。

しかし BEACON は heterogeneous MPMAB を扱う。

heterogeneous setting では、player ごとに arm の平均報酬が異なる。

```text
player 1 にとって arm k は良い
player 2 にとって arm k は悪い
```

という状況が普通に起こる。

BEACON の Oracle は `K x M` の player-arm 推定値行列

```text
mu_bar[k,m]
```

を使って matching を解く。

そのため leader は、基本的に各 player `m` と各 arm `k` の推定値を必要とする。

sub-leader が group 内の値を 1 本の平均や合計にまとめると、player ごとの差が失われ、Oracle が正しい matching を解けなくなる。

したがって、先ほどのような

```text
sub-leader が group aggregate だけ送る
```

という改善は、homogeneous では成立しやすいが、heterogeneous BEACON にはそのまま適用しにくい。

## 6. BEACON を改良するならどこか

BEACON の量子化差分通信は維持したまま、改善を狙うなら、送る値を集約するよりも

```text
送る player-arm pair を減らす
```

方向が自然である。

### 6.1 Active pair filtering

全ての `(k, m)` を送るのではなく、matching decision に影響しそうな player-arm pair だけ送る。

```text
send only (k,m) pairs whose confidence interval can affect the matching
```

明らかに悪い pair は送らない。

ただし理論保証には、送らなかった pair が最適 matching に影響しないことを UCB/LCB で示す必要がある。

### 6.2 Top-L candidates per player

各 player が全 arm ではなく、自分にとって上位 `L` 個の候補 arm だけ送る。

```text
L << K
communication ~= O(M L)
```

実装は比較的簡単だが、真の最適 matching に必要な arm を落とす可能性がある。

理論保証は弱くなりやすい。

### 6.3 Sub-leader as filter

sub-leader を単なる中継者ではなく filter として使う。

Phase A:

```text
follower -> sub-leader:
  quantized delta of sample mean
```

Sub-leader:

```text
group 内の情報から、grand leader に送る必要がある (k,m) を選ぶ
```

Phase B:

```text
sub-leader -> grand leader:
  selected quantized deltas only
```

この場合、Phase B の通信量は全員分の relay ではなく sparse relay になる。

ただし sub-leader が「何を送らなくても安全か」を判定する必要があり、BEACON の理論からは大きな拡張になる。

### 6.4 Distributed matching

grand leader が完全な `K x M` 行列を持つ設計をやめる。

たとえば、

```text
sub-leader:
  group 内 matching / candidate matching を計算

grand leader:
  group 間で arm が衝突した場合だけ調整
```

という設計が考えられる。

これは通信量を減らせる可能性があるが、BEACON とはかなり別のアルゴリズムになる。

## 7. まとめ

BEACON では、リーダーに送っているのは報酬履歴ではなく

```text
各 follower の各 arm に対する標本平均報酬の量子化差分
```

である。

この時点で通信はかなり効率化されている。

ParallelBEACON で通信チャネル数 `n` を増やしても、現在の設計では Phase A の並列化による短縮分を Phase B の中継が打ち消すため、uplink 通信コストは BEACON と同程度になる。

また、heterogeneous setting では Oracle が player-arm ごとの推定値行列を必要とするため、sub-leader が group 平均に集約して送る改善はそのまま使いにくい。

したがって、本質的に改善するなら次のどちらかになる。

1. BEACON の量子化差分通信は維持しつつ、送信する `(k,m)` pair を減らす。
2. grand leader が全 player-arm 行列を集める設計をやめ、distributed matching 型の別アルゴリズムにする。

BEACON に近い改良としては、まず

```text
Filtered ParallelBEACON
```

を検討するのが自然である。

ただし、その理論保証の核心は

```text
どの (k,m) を送らなくても matching decision に影響しないか
```

を confidence interval に基づいて安全に判定する部分になる。
