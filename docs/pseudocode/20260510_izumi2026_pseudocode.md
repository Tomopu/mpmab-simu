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

## Extraction Summary

Total algorithm blocks: 5
