# Izumi 2026 Pseudocode: Multi-Channel MPMAB without Collision Sensing

作成日: 2026-05-10

このファイルは Claude Code が PDF を読みに行かずに実装できるよう、TeX に記述された擬似コードを論文ごとに集約したものです。

## Source Files

- `papers/(2026) Multi-Channel Communication Algorithm for Multi-Player Multi-Armed Bandits without Collision Sensing/src/data.tex`

## Implementation Notes

- 対象モデル: homogeneous multi-channel MPMAB。論文タイトルは without collision sensing なので、初回実装では collision flag を意思決定に使わない。
- Python 実装では arm/player index を 0-based に統一する。
- このファイルでは `data.tex` に記述された parallel 版の擬似コードだけを扱う。
- `HierarchicalDistributedExploration` の `ComGrandLeader`, `ComSubLeader`, `ComFollower` は `data.tex` では詳細省略されているため、初回実装では通信時間コストと集約結果を明示的にシミュレートする。
- `tau` は TeX 上の括弧が曖昧なので、2022 版との対応を優先して `ceil(log(1 / delta) / min(mu_tilde))` として実装する。
- multi-channel 版では、同一 player が同一時刻に複数 arm を pull しない制約を必ずテストする。

## Extracted TeX Pseudocode

### `papers/(2026) Multi-Channel Communication Algorithm for Multi-Player Multi-Armed Bandits without Collision Sensing/src/data.tex`

#### 1. FindMultipleGoodArms

Source lines: `papers/(2026) Multi-Channel Communication Algorithm for Multi-Player Multi-Armed Bandits without Collision Sensing/src/data.tex:71-112`

```tex
\begin{algorithm}[t]
   \fontsize{8.5pt}{8.6pt}\selectfont
   \caption{FindMultipleGoodArms}
   \label{alg:FindMultipleGoodArms}
   \begin{algorithmic}
      \REQUIRE $K$: total number of arms, $\delta$: confidence parameter, \\$n$: number of good arms to find ($n < K - M$).
      \ENSURE $\mathcal{G}$: set of find good arms, $\tilde{\mu}$: lower bounds map.

      \STATE $\mathcal{G} \gets \emptyset$, $\mathcal{K} \gets \{1, \dots, K\}$ \textit{\# Initialization}
      \STATE $p \gets 0$ \textit{\# Initialize phase counter}

      \WHILE{$|\mathcal{G}| < n$ and $|\mathcal{K}| > 1$}
         \STATE $p \gets p + 1$, $R[k], N[k] \gets 0$ for $k \in \mathcal{K}$

         \STATE \textit{\# Sub-phase 1: Explore active arms uniformly}
         \FOR {$t = 1, \dots, 6|\mathcal{K}|2^p \lceil\ln{\frac{2n}{\delta}}\rceil$}
            \STATE Select arm $k \in \mathcal{K}$ uniformly at random, observe reward $r$, $R[k] \gets R[k] + r$, $N[k] \gets N[k] + 1$
         \ENDFOR

         \STATE \textit{\# Sub-phase 2: Confirm accepted arms}
         \FOR{each $\ell \in \mathcal{K}$ in ascending order}
            \STATE $R'[k] \gets 0$ for $k \in \mathcal{K}$ \textit{\# rewards of samples}

            \STATE \textit{\# if arm $\ell$ was accepted sample arms uniformly}
            \IF{$N[\ell] > 0$ \textbf{and} $\frac{R[\ell]}{N[\ell]} \ge 2^{1-p}$}
               \FOR{$t = 1, \dots, |\mathcal{K}| 2^p  \lceil\ln{\frac{2n}{\delta}}\rceil$}
                  \STATE Select arm $k \in \mathcal{K}$ uniformly at random, observe reward $r$, $R'[k] \gets R'[k] +  r$
               \ENDFOR
               \IF {$R'[\ell] \geq 1$}
                  % $\tilde{k} \gets \ell$, $\tilde{\mu}[\ell] \gets 2^{p}$ \textbf{break end~if}
                  \STATE Add $\ell$ to $\mathcal{G}$, $\tilde{\mu}[\ell] \gets 2^{-p}$
                  \STATE \textbf{if} $|\mathcal{G}| \ge n$ \textbf{then break end~if} \textit{\# Exit FOR loop}
               \ENDIF
            \ELSE
               \STATE \textbf{for} $t = 1, \dots, |\mathcal{K}| 2^p \lceil\ln{\frac{2n}{\delta}}\rceil$ \textbf{do} Select arm $\ell$ observe reward $r$, $R'[\ell] \gets R'[\ell] +  r$ \textbf{end for}
            \ENDIF
         \ENDFOR
         \STATE $\mathcal{K} \gets \mathcal{K} \setminus \mathcal{G}$ \textit{\# Update active set at the end of the phase}
      \ENDWHILE
      % \STATE \textbf{return} $\mathcal{G}, \tilde{\mu}$
   \end{algorithmic}
\end{algorithm}
```

#### 2. ParallelVirtualMusicalChairs

Source lines: `papers/(2026) Multi-Channel Communication Algorithm for Multi-Player Multi-Armed Bandits without Collision Sensing/src/data.tex:127-166`

```tex
\begin{algorithm}[t]
   \fontsize{8.5pt}{8.2pt}\selectfont
   \caption{ParallelVirtualMusicalChairs}
   \label{alg:ParallelVirtualMusicalChairs}
   \begin{algorithmic}
      \REQUIRE $K$: total number of arms, $\mathcal{G}$: set of good arms, $\tau$: sampling times.
      \ENSURE $s$: external rank of the player.

      \STATE $s \gets -1$ \textit{\# Rank of the player is initially unset}
      \STATE $n = |\mathcal{G}|$ \textit{\# Number of the good arms}

      \STATE \textit{\# Musical chairs on the good arms} $\mathcal{G} = \{\tilde{k}_1, \dots, \tilde{k}_n\}$
      \FOR {$t = 1, \dots, \lceil K\tau/n \rceil$}
         \STATE \textit{\# Select sampling slots at each block start}
         \IF {$t \bmod K = 1$}
            \IF {$s = -1$}
               \STATE $\mathcal{F}_{1} = \emptyset$ \textit{\# Initialization of the sets}
               \FOR {$i = 1, \dots, n$}
                  \STATE \textit{\# Choose the $i$-th candidate arm}
                  \STATE Draw $\ell_i \in \{1, \dots, K\} \setminus \mathcal{F}_i$ uniformly at random
                  \STATE $\mathcal{F}_{i+1} \gets \{\big((\ell_j + (i - j)) \bmod K\big) + 1\mid 1 \leq j \leq i\}$
               \ENDFOR
            \ELSE
               \STATE $\ell_j \gets s$ for all $j \in \{1, \dots, n\}$ \textit{\# Wait at rank $s$}
            \ENDIF
         \ENDIF
         \STATE \textit{\# sample the corresponding time slot}
         \IF {$\exists i \in \{1, \dots, n\}$ s.t. $((t + i - 2) \bmod K)+1  = \ell_i$}
            \STATE Select arm $\tilde{k}_i$, and observe reward $r_i$
            \STATE \textit{\# set the rank if it was not set yet and a non zero reward was obtained}
            \IF {$r_i > 0$ and $s = -1$}
               \STATE $s \gets \ell_i$
               \STATE $\ell_j \gets s$ for all $j \in \{1, \dots, n\}$
            \ENDIF
         \ELSE
            \STATE Select an arbitrary arm in $\{1, \dots, K\} \setminus \mathcal{G}$
         \ENDIF
      \ENDFOR
   \end{algorithmic}
\end{algorithm}
```

#### 3. ParallelVirtualNumberPlayers

Source lines: `papers/(2026) Multi-Channel Communication Algorithm for Multi-Player Multi-Armed Bandits without Collision Sensing/src/data.tex:182-217`

```tex
\begin{algorithm}[t]
   \fontsize{8.5pt}{8.2pt}\selectfont
   \caption{ParallelVirtualNumberPlayers}
   \label{alg:ParallelVirtualNumberPlayers}
   \begin{algorithmic}
      \REQUIRE $K$: total number of arms, $\mathcal{G} = \{\tilde{k}_1, \dots, \tilde{k}_n\}$: set of good arms, $s$: external rank, $\tau$: sampling times.
      \ENSURE $\hat{M}$: estimated number of players, $j$: internal rank.

      \STATE $\hat{M} \gets 1$, $j \gets 1$, $n \gets |\mathcal{G}|$ \textit{\# Initialization}

      \FOR {$h = 1, \dots, \lceil 2K/n \rceil$}
         \FOR {$i = 1, \dots, n$}
            \STATE $v \gets (h - 1)n + i$ \textit{\# Calculate virtual time}
            \STATE \textbf{if} $v > 2K$ \textbf{then} $\ell_i \gets -1$; \textbf{continue end if}
            \IF {$v > 2s$}
               \STATE $\ell_i \gets \big((s + (v - 2s) - 1) \bmod K\big) + 1$
            \ELSE
               \STATE $\ell_i \gets s$ \textit{\# Wait at rank $s$}
            \ENDIF
         \ENDFOR
         \FOR {$k = 1, \dots, K$}
            \STATE $R \gets 0$
            \IF {$\exists i \in \{1, \dots, n\}$ s.t. $\ell_i \neq -1$ \textbf{and} \\ $\quad \big((\ell_i - 1 + (i - 1)) \bmod K\big) + 1 = k$}
               \STATE $\tilde{i} \gets \text{the matching index}$
               \STATE \textbf{for} $t = 1, \dots, \tau$ \textbf{do} Select arm $\tilde{k}_{\tilde{i}}$, observe reward $r$, $R \gets R + r$ \textbf{end for}
               \IF {$R = 0$}
                  \STATE $\hat{M} \gets \hat{M} + 1$; $v \gets (h - 1)n + \tilde{i}$
                  \STATE \textbf{if} $v \leq 2s$ \textbf{then} $j \gets j + 1$ \textbf{end if}
               \ENDIF
            \ELSE
               \STATE \textbf{for} $t = 1, \dots, \tau$ \textbf{do} Select an arbitrary arm in $\{1, \dots, K\}\setminus\mathcal{G}$ \textbf{end for}
            \ENDIF
         \ENDFOR
      \ENDFOR
   \end{algorithmic}
\end{algorithm}
```

#### 4. HierarchicalDistributedExploration

Source lines: `papers/(2026) Multi-Channel Communication Algorithm for Multi-Player Multi-Armed Bandits without Collision Sensing/src/data.tex:233-275`

```tex
\begin{algorithm}[t]
   \fontsize{8.5pt}{8.2pt}\selectfont
   \caption{HierarchicalDistributedExploration}
   \label{alg:HierarchicalDistributedExploration}
   \begin{algorithmic}
      \REQUIRE $K$: number of arms, $j$: internal rank of a player, \\$\hat{M}$: estimated number of players, $\mathcal{G} = \{\tilde{k}_1, \dots, \tilde{k}_n\}$: set of good arms, $\tau$: sampling times
      \ENSURE $f$: an arm amongst the $M$ best arms assigned to the player

      \STATE $p \gets 0$, $f \gets -1$, $n \gets |\mathcal{G}|$ \textit{\# Initialization}
      \STATE $R[k], v[k] \gets 0$ for $k=1, \dots, K$ \textit{\# Rewards and number of samples for each arm}

      \STATE \textit{\# Initialize memory for Grand Leader and Sub-Leaders}
      \IF {$j \le n$}
         \STATE \textbf{for} $m=1, \dots, \hat{M}$ and $k=1, \dots, K$ \textbf{do} \\ $\quad\hat{\mu}[k, m], N[k, m] \gets 0$ \textbf{end for}
      \ENDIF

      \STATE $M' \gets \hat{M}$, $\mathcal{K} \gets \{1, \dots, K\}$ \textit{\# Number of active players and set of active arms}

      \WHILE {$f = -1$}
         \STATE $p \gets p + 1$
         \STATE \textit{\# Sub-phase 1: Explore arms by sequential hopping}
         \STATE $k \gets j$
         \FOR {$t = 1, \dots, |\mathcal{K}| 2^p \lceil \ln \frac{1}{\delta} \rceil$}
            \STATE $k \gets (k + 1) \bmod{|\mathcal{K}|}$
            \STATE Select arm $k$, observe reward $r$, $R[k] \gets R[k] + r$, $v[k] \gets v[k] + 1$, $E[k] \gets R[k] / v[k]$
         \ENDFOR
         \STATE \textit{\# Sub-phase 2: Hierarchical Information Exchange}
         \STATE $Q \gets \lceil \frac{p}{2} + 3 \rceil$ \textit{\# Message length}
         \IF {$j = 1$}
            \STATE \textit{\# Grand-Leader}
            \STATE $(f, \mathcal{K}, M', \hat{\mu}, N)$
            \STATE \hspace{4em} $\gets \texttt{ComGrandLeader}(\hat{\mu}, N, E, \mathcal{G}, M', Q, \tau, p)$
         \ELSIF {$j \le n$}
            \STATE \textit{\# Sub-Leader: Send to Grand-Leader}
            \STATE $(f, \mathcal{K}, M', \hat{\mu}, N)$
            \STATE \hspace{4em} $\gets \texttt{ComSubLeader}(\hat{\mu}, N, E, j, \mathcal{G}, M', Q, \tau, p)$
         \ELSE
            \STATE \textit{\# Follower: Send to Sub-Leader}
            \STATE $(f, \mathcal{K}, M') \gets \texttt{ComFollower}(E, j, \mathcal{G}, M', Q, \tau)$
         \ENDIF
      \ENDWHILE
   \end{algorithmic}
\end{algorithm}
```

#### 5. ProposedParallelAlgorithm

Source lines: `papers/(2026) Multi-Channel Communication Algorithm for Multi-Player Multi-Armed Bandits without Collision Sensing/src/data.tex:280-295`

```tex
\begin{algorithm}[t]
   \fontsize{8.5pt}{8.2pt}\selectfont
   \caption{ProposedParallelAlgorithm}
   \label{alg:ProposedParallelAlgorithm}
   \begin{algorithmic}
      \REQUIRE $K$: number of arms, $\delta$: confidence level, $n$: number of good arms to find
      \ENSURE $\bar{k}$: an arm amongst the $M$ best arms assigned to the player
      \STATE $(\mathcal{G}, \tilde{\mu}) \gets \texttt{FindMultipleGoodArms}(K, \delta, n)$
      \STATE $\tilde{\mu}_{\min} \gets \min_{k \in \mathcal{G}} \tilde{\mu}[k]$
      \STATE $\tau \gets \lceil \ln{(1/\delta)/\tilde{\mu}_{\min}} \rceil$ \textit{\# Calculate sampling times}
      % \STATE $\tau \gets \lceil \ln{(1/\delta)/\tilde{\mu}} \rceil$ \textit{\# Calculate sampling times}
      \STATE $s \gets \texttt{ParallelVirtualMusicalChairs}(K, \mathcal{G}, \tau)$
      \STATE $(\hat{M}, j) \gets \texttt{ParallelVirtualNumberPlayers}(K, \mathcal{G}, s, \tau)$
      \STATE $\bar{k} \gets \texttt{HierarchicalDistributedExploration}(K, j, \hat{M}, \mathcal{G}, \tau)$
   \end{algorithmic}
\end{algorithm}
```

## Supplemental Pseudocode for Omitted Communication Routines

この節は `data.tex` に明示されていない `ComGrandLeader`, `ComSubLeader`, `ComFollower` を、Huang 2022 の `ComLeader` / `ComFollow` と本論文本文の階層的リーダー制の説明から実装用に補完したものである。

注意:

- これは TeX から抽出した原文擬似コードではなく、実装のための補完仕様である。
- 初回実装では forced-collision bit 通信を完全再現しなくてよい。Huang 2022 実装と同様に、通信結果は明示的に集約し、通信時間コストだけを `Trace` に記録してよい。
- accept/reject 判定は Huang 2022 の `ComLeader` と同じ式を使う。
- group id は論文本文に従い `g = ((j - 1) mod n) + 1` とする。
- sub-leader は `j <= n` の player。grand leader は `j = 1` の player。

### Shared Helper: AcceptReject

```text
Input:
  K_active: active arms
  M_active: active players count M'
  mu_hat[k, m]: player m's estimate for arm k
  N[k, m]: player m's sample count for arm k
  p: phase index
  delta: confidence parameter

Output:
  C_accept: accepted arms
  C_reject: rejected arms

For each k in K_active:
  1. 集約推定値を計算する
     rho[k] <- sum_m(mu_hat[k, m] * N[k, m]) / sum_m(N[k, m])

  2. Huang 2022 と同じ信頼半径を計算する
     B[k] <- sqrt(2 * log(1 / delta) / sum_m(N[k, m])) + 2^(-p/2 - 3)

  3. top-M_active に入ることが確実な arm を accept する
     if |{i in K_active: rho[k] - B[k] >= rho[i] + B[i]}| >= |K_active| - M_active:
       add k to C_accept

  4. top-M_active に入らないことが確実な arm を reject する
     if |{i in K_active: rho[i] - B[i] >= rho[k] + B[k]}| >= M_active:
       add k to C_reject
```

### Shared Helper: AssignAndUpdate

```text
Input:
  player rank j
  good arms G
  active arms K_active
  active players count M_active
  C_accept
  C_reject

Output:
  f: assigned arm or -1
  K_next: updated active arms
  M_next: updated active players count

1. 通信用 good arms を follower に割り当てないため、accepted set から G を除外する
   C_assign <- C_accept \ G

2. rank の大きい active player から accepted arm を割り当てる
   if M_active - j + 1 <= |C_assign|:
      f <- C_assign[M_active - j + 1]
      return f, K_active, M_active

3. 割り当てがない場合は、accepted/rejected arms を inactive にする
   K_next <- K_active \ (C_accept union C_reject)
   M_next <- M_active - |C_accept|
   f <- -1
   return f, K_next, M_next
```

### ComGrandLeader

```text
Input:
  mu_hat, N: grand leader が保持する player-arm 統計
  E: grand leader 自身の current estimates
  v: grand leader 自身の sample counts
  G = {k_tilde_1, ..., k_tilde_n}: good arms used as channels
  groups: mapping from group id to player ids
  K_active: active arms
  M_active: active players count M'
  Q: message length
  tau: sampling time
  p: phase index
  delta: confidence parameter

Output:
  f: grand leader assigned arm or -1
  K_next
  M_next
  mu_hat_next
  N_next
  C_accept
  C_reject

1. grand leader 自身の統計を mu_hat, N に反映する
   For each k in K_active:
     mu_hat[k, grand_leader] <- E[k]
     N[k, grand_leader] <- v[k]

2. group 1 の follower から統計を受け取る
   For each follower m in groups[1] excluding grand_leader:
     For each k in K_active:
       receive E_m[k], v_m[k] through channel G[1]
       mu_hat[k, m] <- E_m[k]
       N[k, m] <- v_m[k]
       consume communication cost |K_active| * Q * tau

3. sub-leader から各 group の集約統計を受け取る
   For each group g = 2, ..., n:
     Let subleader be the player with internal rank j = g
     If subleader is active:
       For each k in K_active:
         receive group_mu[g, k], group_N[g, k] through channel G[1]
         store them as aggregated group statistics
         consume communication cost |K_active| * Q * tau

4. player-level 統計と group-level 統計を合わせて global estimates を作る
   Implementation option A:
     Keep group statistics as weighted pseudo-player entries.
   Implementation option B:
     Expand group statistics back into mu_hat/N if individual follower values are available.
   初回実装では option A でよい。

5. AcceptReject を実行する
   C_accept, C_reject <- AcceptReject(K_active, M_active, mu_hat, N, p, delta)

6. 決定内容を sub-leader と group 1 follower に送る
   Send |C_accept|, |C_reject|, C_accept, C_reject.
   The downlink follows the reverse route of steps 2 and 3.
   consume communication cost for each receiver.

7. 自身の割当と active set を更新する
   f, K_next, M_next <- AssignAndUpdate(j=1, G, K_active, M_active, C_accept, C_reject)

8. Return f, K_next, M_next, mu_hat, N, C_accept, C_reject
```

### ComSubLeader

```text
Input:
  E: sub-leader 自身の current estimates
  v: sub-leader 自身の sample counts
  j: sub-leader internal rank, where 2 <= j <= n
  G: good arms used as channels
  groups: mapping from group id to player ids
  K_active
  M_active
  Q
  tau
  p

Output:
  f: sub-leader assigned arm or -1
  K_next
  M_next
  group_mu
  group_N

1. 自分の group id を決める
   g <- j
   channel_to_followers <- G[g]
   channel_to_grand_leader <- G[1]

2. 自身の統計で group aggregate を初期化する
   For each k in K_active:
     group_sum[k] <- E[k] * v[k]
     group_N[k] <- v[k]

3. 同じ group の follower から統計を受け取る
   For each follower m in groups[g] excluding sub-leader:
     For each k in K_active:
       receive E_m[k], v_m[k] through channel_to_followers
       group_sum[k] <- group_sum[k] + E_m[k] * v_m[k]
       group_N[k] <- group_N[k] + v_m[k]
       consume communication cost |K_active| * Q * tau

4. group aggregate を計算する
   For each k in K_active:
     if group_N[k] > 0:
       group_mu[k] <- group_sum[k] / group_N[k]
     else:
       group_mu[k] <- 0

5. group aggregate を grand leader に送る
   For each k in K_active:
     send group_mu[k], group_N[k] through channel_to_grand_leader
     consume communication cost |K_active| * Q * tau

6. grand leader から accept/reject 決定を受け取る
   receive C_accept, C_reject through channel_to_grand_leader
   consume downlink communication cost

7. 決定内容を同じ group の follower に中継する
   For each follower m in groups[g] excluding sub-leader:
     send C_accept, C_reject through channel_to_followers
     consume downlink communication cost

8. 自身の割当と active set を更新する
   f, K_next, M_next <- AssignAndUpdate(j, G, K_active, M_active, C_accept, C_reject)

9. Return f, K_next, M_next, group_mu, group_N
```

### ComFollower

```text
Input:
  E: follower current estimates
  v: follower sample counts
  j: follower internal rank, where j > n
  G: good arms used as channels
  K_active
  M_active
  Q
  tau

Output:
  f: follower assigned arm or -1
  K_next
  M_next

1. 所属 group を決める
   g <- ((j - 1) mod n) + 1
   channel <- G[g]

2. 自分の推定値と sample count を sub-leader に送る
   For each k in K_active:
     send E[k], v[k] through channel
     consume communication cost |K_active| * Q * tau

3. sub-leader から accept/reject 決定を受け取る
   receive C_accept, C_reject through channel
   consume downlink communication cost

4. 自身の割当と active set を更新する
   f, K_next, M_next <- AssignAndUpdate(j, G, K_active, M_active, C_accept, C_reject)

5. Return f, K_next, M_next
```

### Implementation Notes for Simplified Simulator

```text
初回実装では、以下の簡略化を許容する。

1. Communication payload is copied directly.
   forced-collision bit protocol は再現せず、E/v や accept/reject set は関数呼び出しで直接渡す。

2. Communication cost is still recorded.
   Runner/Trace には communication phase の dummy steps を追加し、Huang 2022 と Izumi 2026 の所要時間比較ができるようにする。

3. Parallel group uplink/downlink can be charged by max group cost.
   n groups が別 channel を使うため、同時に通信できる uplink/downlink は「全 group の合計」ではなく「最も長い group の通信時間」で近似してよい。

4. Grand leader aggregation remains the only accept/reject decision point.
   accept/reject の計算は grand leader だけが行い、sub-leader/follower は受け取った決定に従う。
```

## Extraction Summary

Total algorithm blocks: 5
