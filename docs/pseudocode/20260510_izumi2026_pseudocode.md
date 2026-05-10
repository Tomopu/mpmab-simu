# Izumi 2026 Pseudocode: Multi-Channel MPMAB without Collision Sensing

作成日: 2026-05-10

このファイルは Claude Code が PDF を読みに行かずに実装できるよう、TeX に記述された擬似コードを論文ごとに集約したものです。

## Source Files

- `papers/(2026) Multi-Channel Communication Algorithm for Multi-Player Multi-Armed Bandits without Collision Sensing/src/data.tex`
- `papers/(2026) Multi-Channel Communication Algorithm for Multi-Player Multi-Armed Bandits without Collision Sensing/src/algorihtm.tex`
- `papers/(2026) Multi-Channel Communication Algorithm for Multi-Player Multi-Armed Bandits without Collision Sensing/src/algorithm2.tex`

## Implementation Notes

- 対象モデル: homogeneous multi-channel MPMAB。論文タイトルは without collision sensing なので、初回実装では collision flag を意思決定に使わない。
- Python 実装では arm/player index を 0-based に統一する。
- `data.tex` は最終的に実装したい parallel 版の擬似コードを含む。
- `algorihtm.tex` と `algorithm2.tex` は sequential 版や通信サブルーチンを含む。ファイル名 `algorihtm.tex` は原文の typo のまま。
- `HierarchicalDistributedExploration` の `ComGrandLeader`, `ComSubLeader`, `ComFollower` は `data.tex` では詳細省略、`algorihtm.tex` に詳細擬似コードがある。
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

### `papers/(2026) Multi-Channel Communication Algorithm for Multi-Player Multi-Armed Bandits without Collision Sensing/src/algorihtm.tex`

#### 1. SequentialFindGoodArms (for player $m=1, \dots, M$)

Source lines: `papers/(2026) Multi-Channel Communication Algorithm for Multi-Player Multi-Armed Bandits without Collision Sensing/src/algorihtm.tex:1-39`

```tex
\begin{algorithm}[t]
   \caption{SequentialFindGoodArms (for player $m=1, \dots, M$)}
   \label{alg:SequentialFindGoodArms}
   \begin{algorithmic}
      \REQUIRE $K$: total number of arms, $\delta$: confidence parameter, $n$: number of good arms to find ($n < K - M$).
      \ENSURE $\mathcal{G}$: set of find good arms, $\tilde{\mu}$: lower bounds map.

      \STATE $\mathcal{G} \gets \emptyset$, $\mathcal{K} \gets \{1, \dots, K\}$ \textit{\# Initialization}

      \STATE \textit{\# Iterate until $n$ arms are found or no arms left}
      \WHILE{$|\mathcal{G}| < n$ and $|\mathcal{K}| > 1$}
         \STATE $p \gets 0$, $\tilde{k} \gets -1$
         \WHILE{$\tilde{k} = -1$}
            \STATE $p \gets p + 1$, $R[k], N[k] \gets 0$ for $k \in \mathcal{K}$
            \STATE \textit{\# Sub-phase 1: Explore active arms uniformly}
            % \STATE \textit{\# Note: Time complexity scales current set size $|\mathcal{K}|$}
            \FOR {$t = 1, \dots, 6|\mathcal{K}|2^p \ln{\frac{2}{\delta}}$}
               \STATE Select arm $k \in \mathcal{K}$ uniformly at random, observe reward $r$, $R[k] += r$, $N[k] += 1$
            \ENDFOR
            \STATE \textit{\# Sub-phase 2: Confirm accepted arms}
            \FOR{$\ell \in \mathcal{K}$}
               \STATE $R'[k] \gets 0$ for $k \in \mathcal{K}$ \textit{\# rewards of samples}
               \STATE \textit{\# if arm $\ell$ was accepted sample arms uniformly}
               \IF{$\frac{R[\ell]}{N[\ell]} \ge 2^{1-p}$}
                  \FOR{$t = 1, \dots, |\mathcal{K}| 2^p \ln{\frac{2}{\delta}}$}
                     \STATE Select arm $k \in \mathcal{K}$ uniformly at random, observe reward $r$, $R'[k] += r$
                  \ENDFOR
                  \STATE \textbf{if} $R'[\ell] \geq 1$ \textbf{then} $\tilde{k} \gets \ell$, $\tilde{\mu}[\ell] \gets 2^{p}$ \textbf{break end~if}
               \ELSE
                  \STATE \textbf{for} $t = 1, \dots, |\mathcal{K}| 2^p \ln{\frac{2}{\delta}}$ \textbf{do} Select arm $\ell$ observe reward $r$, $R'[\ell] += r$ \textbf{end for}
               \ENDIF
            \ENDFOR
         \ENDWHILE
         % \STATE \textit{\# Update sets for next iteration}
         \STATE Add $\tilde{k}$ to $\mathcal{G}$, remove $\tilde{k}$ from $\mathcal{K}$
      \ENDWHILE
      \STATE \textbf{return} $\mathcal{G}, \tilde{\mu}$
   \end{algorithmic}
\end{algorithm}
```

#### 2. SequentialVirtualMusicalChairs (for player $m=1, \dots, M$)

Source lines: `papers/(2026) Multi-Channel Communication Algorithm for Multi-Player Multi-Armed Bandits without Collision Sensing/src/algorihtm.tex:41-79`

```tex
\begin{algorithm}[t]
   \caption{SequentialVirtualMusicalChairs (for player $m=1, \dots, M$)}
   \label{alg:SequentialVirtualMusicalChairs}
   \begin{algorithmic}
      \REQUIRE $K$: total number of arms, $\mathcal{G}$: set of good arms, $\tau$: sampling times.
      \ENSURE $s$: external rank of the player.

      \STATE $s \gets -1$ \textit{\# Rank of the player is initially unset}
      \STATE $n = |\mathcal{G}|$ \textit{\# Number of the good arms}

      \STATE \textit{\# Musical chairs on the good arms} $\mathcal{G} = \{\tilde{k}_1, \dots, \tilde{k}_n\}$
      \FOR {$t \gets 1, \dots, \lceil K\tau/n \rceil$}
         \STATE \textit{\# Time is split in blocks of size $K$ and we select when to sample at the start of a block}
         \IF {$t \bmod K = 1$}
            \IF {$s = -1$}
               \STATE $\mathcal{L}_{0} = \emptyset$, $\mathcal{F}_{1} = \emptyset$ \textit{\# Initialization of the sets}
               \FOR {$i \gets 1, \dots, n$}
                  \STATE Draw $\ell_i \in \{1, \dots, K\} \setminus (\mathcal{L}_{i-1} \cup \mathcal{F}_i)$ uniformly at random \textit{\# Choose the $i$-th candidate arm}
                  \STATE $\mathcal{L}_{i} \gets \mathcal{L}_{i-1} \cup \{\ell_i\}$
                  \STATE $\mathcal{F}_{i+1} \gets \{\big((\ell_j + (i - j)) \bmod K\big) + 1\mid 1 \leq j \leq i\}$
               \ENDFOR
            \ELSE
               \STATE $\ell_j \gets s$ for all $j \in \{1, \dots, n\}$ \textit{\# Choose the rank as a slot if it is set}
            \ENDIF
         \ENDIF
         \STATE \textit{\# sample the corresponding time slot}
         \IF {$\exists i \in \{1, \dots, n\}$ s.t. $(t \bmod K) + (i - 1) = \ell_i$}
            \STATE Select arm $\tilde{k}_i$, and observe reward $r_i$
            \STATE \textit{\# set the rank if it was not set yet and a non zero reward was obtained}
            \IF {$r_i > 0$ and $s = -1$}
               \STATE $s \gets \ell_i$
               \STATE $\ell_j \gets s$ for all $j \in \{1, \dots, n\}$
            \ENDIF
         \ELSE
            \STATE Select an arbitrary arm in $\{1, \dots, K\} \setminus \{\tilde{k}_i\}$
         \ENDIF
      \ENDFOR
   \end{algorithmic}
\end{algorithm}
```

#### 3. SequentialVirtualNumberPlayers (for player $m$)

Source lines: `papers/(2026) Multi-Channel Communication Algorithm for Multi-Player Multi-Armed Bandits without Collision Sensing/src/algorihtm.tex:81-116`

```tex
\begin{algorithm}[t]
   \caption{SequentialVirtualNumberPlayers (for player $m$)}
   \label{alg:SequentialVirtualNumberPlayers}
   \begin{algorithmic}
      \REQUIRE $K$: total number of arms, $\mathcal{G} = \{\tilde{k}_1, \dots, \tilde{k}_n\}$: set of good arms, $s$: external rank, $\tau$: sampling times.
      \ENSURE $\hat{M}$: estimated number of players, $j$: internal rank.

      \STATE $\hat{M} \gets 1$, $j \gets 1$, $n \gets |\mathcal{G}|$ \textit{\# Initialization}

      \FOR {$h = 1, \dots, \lceil 2K/n \rceil$}
         \FOR {$i = 1, \dots, n$}
            \STATE $v \gets (h - 1)n + i$ \textit{\# Calculate virtual time}
            \STATE \textbf{if} $v > 2K$ \textbf{then} $\ell_i \gets -1$; \textbf{continue end if}
            \IF {$v > 2s$}
               \STATE $b_i \gets \big((s + (v - 2s) - 1) \bmod K\big) + 1$
            \ELSE
               \STATE $b_i \gets s$ \textit{\# Wait at rank $s$}
            \ENDIF
            \STATE $\ell_i \gets \big((b_i - 1 + (i - 1)) \bmod K\big) + 1$
         \ENDFOR
         \FOR {$k = 1, \dots, K$}
            \STATE $R \gets 0$
            \IF {$\exists i \in \{1, \dots, n\}$ s.t. $\ell_i = k$}
               \STATE \textbf{for} $t = 1, \dots, \tau$ \textbf{do} Select arm $\tilde{k}_i$, observe reward $r$, $R \gets R + r$ \textbf{end for}
               \IF {$R = 0$}
                  \STATE $\hat{M} \gets \hat{M} + 1$
                  \STATE $v \gets (h - 1)n + i$ \textit{\# Recalculate virtual time for the matching arm}
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

#### 4. SequentialDistributedExploration (Proposed Method)

Source lines: `papers/(2026) Multi-Channel Communication Algorithm for Multi-Player Multi-Armed Bandits without Collision Sensing/src/algorihtm.tex:118-154`

```tex
\begin{algorithm}[t]
   \caption{SequentialDistributedExploration (Proposed Method)}
   \label{alg:SequentialDistributedExploration}
   \begin{algorithmic}
      \REQUIRE $K$: number of arms, $j$: internal rank of a player, $\hat{M}$: estimated number of players, $\mathcal{G} = \{\tilde{k}_1, \dots, \tilde{k}_n\}$: set of good arms, $\tau$: sampling times
      \ENSURE $f$: an arm amongst the $M$ best arms assigned to the player

      \STATE $p \gets 0$, $f \gets -1$, $n \gets |\mathcal{G}|$ \textit{\# Initialization}
      \STATE $R[k], v[k] \gets 0$ for $k=1, \dots, K$ \textit{\# Rewards and number of samples for each arm}

      \STATE \textit{\# Initialize memory for Grand Leader and Sub-Leaders}
      \IF {$j \le n$}
         \STATE \textbf{for} $m=1, \dots, \hat{M}$ and $k=1, \dots, K$ \textbf{do} $\hat{\mu}[k, m], N[k, m] \gets 0$ \textbf{end for}
      \ENDIF

      \STATE $M' \gets \hat{M}$, $\mathcal{K} \gets \{1, \dots, K\}$ \textit{\# Number of active players and set of active arms}

      \WHILE {$f = -1$}
         \STATE $p \gets p + 1$
         \STATE \textit{\# Sub-phase 1: Explore arms by sequential hopping (Same as original)}
         \STATE $k \gets j$
         \FOR {$t = 1, \dots, |\mathcal{K}| 2^p \lceil \ln \frac{1}{\delta} \rceil$}
            \STATE $k \gets (k + 1) \bmod{|\mathcal{K}|}$
            \STATE Select arm $k$, observe reward $r$, $R[k] \gets R[k] + r$, $v[k] \gets v[k] + 1$, $E[k] \gets R[k] / v[k]$
         \ENDFOR
         \STATE \textit{\# Sub-phase 2: Hierarchical Information Exchange}
         \STATE $Q \gets \lceil \frac{p}{2} + 3 \rceil$ \textit{\# Message length}
         \IF {$j = 1$}
            \STATE $(f, \mathcal{K}, M') \gets \texttt{ComGrandLeader}(E, \mathcal{G}, M', Q, \tau, p)$ \textit{\# Grand Leader: Aggregates from Sub-Leaders}
         \ELSIF {$j \le n$}
            \STATE $(f, \mathcal{K}, M') \gets \texttt{ComSubLeader}(E, j, \mathcal{G}, M', Q, \tau, p)$ \textit{\# Sub-Leader: Aggregates from Followers, sends to Grand Leader}
         \ELSE
            \STATE $(f, \mathcal{K}, M') \gets \texttt{ComFollower}(E, j, \mathcal{G}, M', Q, \tau)$ \textit{\# Follower: Sends to Sub-Leader}
         \ENDIF
      \ENDWHILE
   \end{algorithmic}
\end{algorithm}
```

#### 5. ProposedSequentialAlgorithm (for player $m = 1, \dots, M$)

Source lines: `papers/(2026) Multi-Channel Communication Algorithm for Multi-Player Multi-Armed Bandits without Collision Sensing/src/algorihtm.tex:156-167`

```tex
\begin{algorithm}[t]
   \caption{ProposedSequentialAlgorithm (for player $m = 1, \dots, M$)}
   \label{alg:ProposedSequentialAlgorithm}
   \begin{algorithmic}
      \REQUIRE $K$: number of arms, $\delta$: confidence level
      \ENSURE $\bar{k}$: an arm amongst the $M$ best arms assigned to the player
      \STATE $(\mathcal{G}, \tilde{\mu}) \gets \texttt{SequentialFindGoodArm}(K, \delta)$
      \STATE $s \gets \texttt{SequentialVirtualMusicalChair}(K, \mathcal{G}, K \ln{(1/\delta)/\tilde{\mu}})$
      \STATE $(\hat{M}, j) \gets \texttt{SequentialVirtualNumberPlayers}(K, \mathcal{G}, s, \ln{(1/\delta)/\tilde{\mu}})$
      \STATE $\bar{k} \gets \texttt{SequentialDistributedExploration}(K, j, \hat{M}, \mathcal{G}, \ln{(1/\delta)/\tilde{\mu}})$
   \end{algorithmic}
\end{algorithm}
```

#### 6. ComGrandLeader (for player $j=1$)

Source lines: `papers/(2026) Multi-Channel Communication Algorithm for Multi-Player Multi-Armed Bandits without Collision Sensing/src/algorihtm.tex:170-211`

```tex
\begin{algorithm}[t]
   \caption{ComGrandLeader (for player $j=1$)}
   \label{alg:ComGrandLeader}
   \begin{algorithmic}
      \REQUIRE $E$: estimates, $\mathcal{G}$: set of good arms, $M'$: active players, $Q$: msg size, $\tau$: sampling time, $p$: phase
      \ENSURE $f$: assigned arm, $\mathcal{K}$: active arms, $M'$: updated active players

      \STATE $\overline{\mu} \leftarrow E$, $\overline{N} \leftarrow v$ \textit{\# Initialize with own stats}
      \STATE $n \leftarrow |\mathcal{G}|$

      \STATE \textit{\# 1. Receive from own group (Group 1) followers}
      \FOR {$i \in \text{Followers}(Group 1)$}
         \FOR {$k \in \mathcal{K}$}
            \STATE $(\mu_{val}, N_{val}) \leftarrow \text{DecoderReceiveFloat}(\tilde{k}_1, \tau, Q)$
            \STATE Update $\overline{\mu}[k], \overline{N}[k]$ with received values
         \ENDFOR
      \ENDFOR

      \STATE \textit{\# 2. Receive aggregated stats from Sub-Leaders (Sequential to avoid collision)}
      \FOR {$g = 2, \dots, n$}
         \FOR {$k \in \mathcal{K}$}
            \STATE $(\mu_{agg}, N_{agg}) \leftarrow \text{DecoderReceiveFloat}(\tilde{k}_1, \tau, Q)$
            \STATE Update $\overline{\mu}[k], \overline{N}[k]$ with aggregated values from Group $g$
         \ENDFOR
      \ENDFOR

      \STATE \textit{\# 3. Make Decisions (Accept / Reject)}
      \FOR {$k \in \mathcal{K}$}
         \STATE Compute global average $\rho[k]$ and confidence $B[k]$ using $\overline{\mu}, \overline{N}$
         \IF {Arm $k$ is optimal w.r.t others} \STATE Add $k$ to $AcceptedSet$ \ENDIF
         \IF {Arm $k$ is suboptimal w.r.t others} \STATE Add $k$ to $RejectedSet$ \ENDIF
      \ENDFOR

      \STATE \textit{\# 4. Broadcast Decisions to Sub-Leaders and Group 1}
      \STATE \textit{\# Send sizes and contents of Accepted/Rejected sets on $\tilde{k}_1$}
      \STATE Broadcast sets via $\text{EncoderSendInt}(\tilde{k}_1, \dots)$

      \STATE \textit{\# 5. Update State and Assign Arm}
      \STATE Update $\mathcal{K}, M'$ based on decisions
      \STATE \textbf{if} Assigning arm \textbf{then} $f \leftarrow \text{AssignedArm}$ \textbf{end if}
   \end{algorithmic}
\end{algorithm}
```

#### 7. ComSubLeader (for player $2 \le j \le n$)

Source lines: `papers/(2026) Multi-Channel Communication Algorithm for Multi-Player Multi-Armed Bandits without Collision Sensing/src/algorihtm.tex:213-247`

```tex
\begin{algorithm}[t]
   \caption{ComSubLeader (for player $2 \le j \le n$)}
   \label{alg:ComSubLeader}
   \begin{algorithmic}
      \REQUIRE $E$: estimates, $j$: rank, $\mathcal{G}$: good arms, $M'$: active players, $Q$: msg size, $\tau$: sampling time, $p$: phase
      \ENSURE $f, \mathcal{K}, M'$

      \STATE $\mu_{group} \leftarrow E$, $N_{group} \leftarrow v$
      \STATE $g \leftarrow j$ \textit{\# My Group ID is my rank}

      \STATE \textit{\# 1. Receive from my group followers (Parallel Uplink)}
      \FOR {$i \in \text{Followers}(Group g)$}
         \FOR {$k \in \mathcal{K}$}
            \STATE $(\mu_{val}, N_{val}) \leftarrow \text{DecoderReceiveFloat}(\tilde{k}_g, \tau, Q)$
            \STATE Update $\mu_{group}[k], N_{group}[k]$
         \ENDFOR
      \ENDFOR

      \STATE \textit{\# 2. Send aggregated stats to Grand Leader (on $\tilde{k}_1$)}
      \STATE Wait for turn (based on group ID $g$)
      \FOR {$k \in \mathcal{K}$}
         \STATE $\text{EncoderSendFloat}(\tilde{k}_1, \tau, Q, \mu_{group}[k], N_{group}[k])$
      \ENDFOR

      \STATE \textit{\# 3. Receive Decisions from Grand Leader (on $\tilde{k}_1$)}
      \STATE Receive Accepted/Rejected sets via $\text{DecoderReceiveInt}(\tilde{k}_1, \dots)$

      \STATE \textit{\# 4. Relay Decisions to my group followers (Parallel Downlink)}
      \STATE Broadcast sets on $\tilde{k}_g$ via $\text{EncoderSendInt}(\tilde{k}_g, \dots)$

      \STATE \textit{\# 5. Update State}
      \STATE Update $\mathcal{K}, M'$ based on received sets
      \textbf{if} Assigning arm \textbf{then} $f \leftarrow \text{AssignedArm}$ \textbf{end if}
   \end{algorithmic}
\end{algorithm}
```

#### 8. ComFollower (for player $j > n$)

Source lines: `papers/(2026) Multi-Channel Communication Algorithm for Multi-Player Multi-Armed Bandits without Collision Sensing/src/algorihtm.tex:249-271`

```tex
\begin{algorithm}[t]
   \caption{ComFollower (for player $j > n$)}
   \label{alg:ComFollower}
   \begin{algorithmic}
      \REQUIRE $E$: estimates, $j$: rank, $\mathcal{G}$: good arms, $M'$: active players, $Q$: msg size, $\tau$: sampling time
      \ENSURE $f, \mathcal{K}, M'$

      \STATE $n \leftarrow |\mathcal{G}|$
      \STATE $g \leftarrow ((j - 1) \pmod n) + 1$ \textit{\# Identify my group ID}

      \STATE \textit{\# 1. Send estimates to my Group Leader (on $\tilde{k}_g$)}
      \FOR {$k \in \mathcal{K}$}
         \STATE $\text{EncoderSendFloat}(\tilde{k}_g, \tau, Q, E[k], v[k])$
      \ENDFOR

      \STATE \textit{\# 2. Receive Decisions from my Group Leader (on $\tilde{k}_g$)}
      \STATE Receive Accepted/Rejected sets via $\text{DecoderReceiveInt}(\tilde{k}_g, \dots)$

      \STATE \textit{\# 3. Update State}
      \STATE Update $\mathcal{K}, M'$ based on received sets
      \STATE \textbf{if} Assigning arm \textbf{then} $f \leftarrow \text{AssignedArm}$ \textbf{end if}
   \end{algorithmic}
\end{algorithm}
```

### `papers/(2026) Multi-Channel Communication Algorithm for Multi-Player Multi-Armed Bandits without Collision Sensing/src/algorithm2.tex`

#### 1. SequentialFindGoodArms

Source lines: `papers/(2026) Multi-Channel Communication Algorithm for Multi-Player Multi-Armed Bandits without Collision Sensing/src/algorithm2.tex:1-30`

```tex
\begin{algorithm}[t]
   \caption{SequentialFindGoodArms}
   \label{alg:SequentialFindGoodArms}
   \begin{algorithmic}[1]
      \REQUIRE $K$, $\delta$, $n$
      \ENSURE $\mathcal{G}$, $\tilde{\mu}$
      \STATE $\mathcal{G} \gets \emptyset$, $\mathcal{K} \gets \{1, \dots, K\}$
      \WHILE{$|\mathcal{G}| < n$ and $|\mathcal{K}| > 1$}
         \STATE $p \gets 0$, $\tilde{k} \gets -1$
         \WHILE{$\tilde{k} = -1$}
            \STATE $p \gets p + 1$, Reset counters $R, N$
            \STATE \textit{\# Sub-phase 1: Uniform Exploration}
            \FOR {$t = 1, \dots, 6|\mathcal{K}|2^p \ln{\frac{2}{\delta}}$}
               \STATE Sample $k \in \mathcal{K}$ uniformly, update $R[k], N[k]$
            \ENDFOR
            \STATE \textit{\# Sub-phase 2: Confirmation}
            \FOR{$\ell \in \mathcal{K}$}
               \IF{$R[\ell]/N[\ell] \ge 2^{1-p}$}
                  \STATE Sample $k \in \mathcal{K}$ uniformly (Confirm step)
                  \STATE \textbf{if} observed reward on $\ell \ge 1$ \textbf{then} $\tilde{k} \gets \ell$ \textbf{break} \textbf{end if}
               \ELSE
                  \STATE Sample $\ell$ (Reject step)
               \ENDIF
            \ENDFOR
         \ENDWHILE
         \STATE Add $\tilde{k}$ to $\mathcal{G}$, remove from $\mathcal{K}$
      \ENDWHILE
      \RETURN $\mathcal{G}, \tilde{\mu}$
   \end{algorithmic}
\end{algorithm}
```

#### 2. SequentialVirtualMusicalChairs

Source lines: `papers/(2026) Multi-Channel Communication Algorithm for Multi-Player Multi-Armed Bandits without Collision Sensing/src/algorithm2.tex:32-52`

```tex
\begin{algorithm}[t]
   \caption{SequentialVirtualMusicalChairs}
   \label{alg:SequentialVirtualMusicalChairs}
   \begin{algorithmic}[1]
      \REQUIRE $K$, $\mathcal{G}$, $\tau$
      \ENSURE $s$
      \STATE $s \gets -1$, $n \gets |\mathcal{G}|$
      \FOR {$t \gets 1, \dots, \lceil K\tau/n \rceil$}
         \IF {$t \bmod K = 1$}
            \STATE Update candidate targets $\ell_1, \dots, \ell_n$ based on history
            \STATE \textbf{if} $s \ne -1$ \textbf{then} $\ell_j \gets s$ for all $j$ \textbf{end if}
         \ENDIF
         \IF {$\exists i$ s.t. $(t \bmod K) + (i - 1) = \ell_i$}
            \STATE Select $\tilde{k}_i \in \mathcal{G}$, observe reward $r$
            \STATE \textbf{if}$r > 0$ and $s = -1$ \textbf{then} $s \gets \ell_i$ \textbf{end if}
         \ELSE
            \STATE Select arbitrary arm in $\{1, \dots, K\} \setminus \mathcal{G}$
         \ENDIF
      \ENDFOR
   \end{algorithmic}
\end{algorithm}
```

#### 3. SequentialVirtualNumberPlayers

Source lines: `papers/(2026) Multi-Channel Communication Algorithm for Multi-Player Multi-Armed Bandits without Collision Sensing/src/algorithm2.tex:54-79`

```tex
\begin{algorithm}[b]
   \caption{SequentialVirtualNumberPlayers}
   \label{alg:SequentialVirtualNumberPlayers}
   \begin{algorithmic}[1]
      \REQUIRE $K$, $\mathcal{G}$, $s$, $\tau$
      \ENSURE $\hat{M}, j$
      \STATE $\hat{M} \gets 1$, $j \gets 1$, $n \gets |\mathcal{G}|$
      \FOR {$h = 1, \dots, \lceil 2K/n \rceil$}
         \FOR {$i = 1, \dots, n$}
            \STATE $v \gets (h - 1)n + i$
            \STATE Calculate target $\ell_i$ based on $v$ and $s$ (Hopping)
         \ENDFOR
         \FOR {$k = 1, \dots, K$}
            \STATE $R \gets 0$
            \IF {$\exists i$ s.t. $\ell_i = k$}
               \STATE \textbf{for} $\tau$ times \textbf{do} Select $\tilde{k}_i$, accumulate $R$ \textbf{end for}
               \IF {$R = 0$}
                  \STATE $\hat{M} \gets \hat{M} + 1$; Update $j$ if $v \le 2s$
               \ENDIF
            \ELSE
               \STATE \textbf{for} $\tau$ times \textbf{do} Select random arm \textbf{end for}
            \ENDIF
         \ENDFOR
      \ENDFOR
   \end{algorithmic}
\end{algorithm}
```

## Extraction Summary

Total algorithm blocks: 16
