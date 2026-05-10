# Huang 2022 Pseudocode: Multi-Player Bandits without Collision Sensing

作成日: 2026-05-10

このファイルは Claude Code が PDF を読みに行かずに実装できるよう、TeX に記述された擬似コードを論文ごとに集約したものです。

## Source Files

- `papers/(2022) Towards Optimal Algorithms for Multi-Player Bandits without Collision Sensing Information/arXiv-2103.13059v2/main.tex`

## Implementation Notes

- 対象モデル: homogeneous MPMAB without collision sensing。
- Python 実装では arm/player index を 0-based に統一する。TeX 内の `1,...,K` と `1,...,M` は実装時に変換する。
- アルゴリズムには collision flag を渡さない。観測できるのは reward のみ。
- `EncoderSendInt` と `DecoderReceiveInt` は TeX では省略されている。`EncoderSendFloat` / `DecoderReceiveFloat` と同じ通信パターンで、固定長 binary integer を送受信する補助関数として実装する。
- `ComLeader` と `ComFollow` は TeX 上で複数の `algorithm` 環境に分割されている。実装では 1 つの関数に統合する。
- `tau_rank = ceil(K * log(1 / delta) / mu_tilde)`、`tau_comm = ceil(log(1 / delta) / mu_tilde)` を基本にする。

## Extracted TeX Pseudocode

### `papers/(2022) Towards Optimal Algorithms for Multi-Player Bandits without Collision Sensing Information/arXiv-2103.13059v2/main.tex`

#### 1. \alg{FindGoodArm} (for player $m=1,...,M)$)

Source lines: `papers/(2022) Towards Optimal Algorithms for Multi-Player Bandits without Collision Sensing Information/arXiv-2103.13059v2/main.tex:142-172`

```tex
\begin{algorithm}[h]
\caption{\alg{FindGoodArm} (for player $m=1,...,M)$)}
\label{algo:FindGoodArm}
\begin{algorithmic}
\REQUIRE $K$: number of arms , $\delta$: confidence parameter
\ENSURE   $\tilde{k}$: good arm, $\tilde{\mu}$: lower bound on reward of $\tilde{k}$
\STATE $p \gets 0$, $\tilde{k} \gets -1$ {\it \# initialization} 
\WHILE{$\tilde{k} = -1$}  
    \STATE $p \pluseq 1$, $R[k],N[k] \gets 0$ for $k=1,...,K$ {\it \# current phase, rewards and number of samples}
    \STATE{\it \# sub-phase 1: explore arms uniformly at random}
    \FOR{$t = 1,...,6 K 2^{p} \ln {2 \over \delta}$} 
        \STATE Select arm $k \in \{1,\dots,K\}$ uniformly at random, observe reward $r$, $R[k] \pluseq  r$, $N[k] \pluseq 1$
    \ENDFOR
    \STATE{\it \# sub-phase 2: confirm accepted arms}
    \FOR{$\ell \gets 1,\dots,K$}
        \STATE $R'[k] \gets 0$ for $k=1,...,K$ {\it \#  rewards of samples} 
        \STATE{\it \# if arm $\ell$ was accepted sample arms uniformly}
        \IF{${R[\ell] \over N[\ell]} \ge 2^{1-p}$}
            \FOR{$t = 1,\dots, 2^{p} K \ln {2 \over \delta}$}
                \STATE Select arm $k \in \{1,\dots,K\}$ uniformly at random and observe reward $r$,  $R'[k] \pluseq r$
            \ENDFOR
            \STATE{\it \# if a non-zero reward is obtained, confirm arm $\ell$}
            \ifthen{$R'[\ell] \ge 1$}{$\tilde{k} \gets \ell$, $\tilde{\mu} \gets 2^{-p}$ {\bf break}} 
            \STATE{\it \hspace{-0.4cm} \# if arm $\ell$ was rejected only sample $\ell$}
        \ELSE
            \for{$t =1,\dots,2^{p} K \ln {2 \over \delta}$}{select arm $k=\ell$, observe reward $r$, $R'[k] \pluseq r$} 
        \ENDIF
    \ENDFOR
\ENDWHILE
\end{algorithmic}
\end{algorithm}
```

#### 2. \alg{VirtualMusicalChairs} (for player $m=1,...,M$)

Source lines: `papers/(2022) Towards Optimal Algorithms for Multi-Player Bandits without Collision Sensing Information/arXiv-2103.13059v2/main.tex:193-220`

```tex
\begin{algorithm}
\caption{\alg{VirtualMusicalChairs} (for player $m=1,...,M$)}
\label{algo:VirtualMusicalChairs}
\begin{algorithmic}
\REQUIRE $K$: number of arms, $\tilde{k}$: a good arm, $\tau$: sampling times
\ENSURE $s$:  external rank of the player
\STATE $s \gets -1$; { \it \# rank of the player is initially unset}
\STATE{ \it \# musical chairs on the arm $\tilde{k}$}
\FOR{$t \gets 1,\dots,K \tau$}
    \STATE{ \it \# time is split in blocks of size $K$ and we select when to sample at the start of a block.}
    \IF{$\mod{(t,K)} = 1$}
        \IF{$s = -1$}
            \STATE Draw $\ell \in \{1,...,K\}$ uniformly at random { \it \# Choose a random slot if rank is unset}
        \ELSE
            \STATE $\ell \gets s$ { \it \# Choose the rank as a slot if it is set}
        \ENDIF
    \ENDIF
    \STATE{ \it \# sample the corresponding time slot}
    \IF{$\mod{(t,K)} = \ell$}
        \STATE Select arm $\tilde{k}$, and observe reward $r$
        \STATE{ \it \# set rank if it was not set yet and a non zero reward was obtained}
        \ifthen{$r > 0$ and $s=-1$}{$s \gets \ell$}
    \ELSE 
        \STATE Select an arbitrary arm in $\{1,...,K\} \setminus \{\tilde{k}\}$ 
    \ENDIF
\ENDFOR
\end{algorithmic}
\end{algorithm}
```

#### 3. \alg{VirtualNumberPlayers} (for player $m=1,...,M$)

Source lines: `papers/(2022) Towards Optimal Algorithms for Multi-Player Bandits without Collision Sensing Information/arXiv-2103.13059v2/main.tex:235-261`

```tex
\begin{algorithm}
\caption{\alg{VirtualNumberPlayers} (for player $m=1,...,M$)}
\label{alg:NumberPlayers}
\begin{algorithmic}
\REQUIRE $K$: number of arms, $\tilde{k}$ a good arm, $s$: external rank of a player, $\tau$: sampling times
\ENSURE $\hat{M}$: estimated number of players, $j$: internal rank of the player
\STATE $\hat{M} \gets 1$, $\ell \gets s$, $j \gets 1$ {\it \# initialization}
\FOR{$n=1,\dots,2K$}
    \ifthen{$n > 2s$}{$\ell \gets \mod{(\ell+1,K)}$ {\it \# sequential hopping}}
    \STATE $R \gets 0$ {\it \# sum of rewards from the good arm}
    \STATE {\it \# sample from the good arm}
    \FOR{$k = 1,\dots,K$}
        \IF{$\ell \ne k$}
            \for{$t=1,\dots,\tau$}{Select an arbitrary arm in $\{1,...,K\} \setminus \{\tilde{k}\}$}
        \ELSE
            \STATE {\it \# sample from virtual arm $\tau$ times}
            \for{$t=1,\dots,\tau$}{Select arm $\tilde{k}$, observe reward $r$, $R \pluseq  r$}            \STATE {\it \# if no non-zero reward was obtained increase the estimated number of players}
            \IF{$R  = 0$}
                \STATE $\hat{M} \pluseq 1$, 
                \ifthen{$n \le 2s$}{$j \pluseq 1$}
            \ENDIF
        \ENDIF
    \ENDFOR
\ENDFOR

\end{algorithmic}
\end{algorithm}
```

#### 4. \alg{DistributedExploration} (for players $m=1,...,M$)

Source lines: `papers/(2022) Towards Optimal Algorithms for Multi-Player Bandits without Collision Sensing Information/arXiv-2103.13059v2/main.tex:278-306`

```tex
\begin{algorithm}
\caption{\alg{DistributedExploration} (for players $m=1,...,M$)}
\label{algo}
\begin{algorithmic}
\REQUIRE $K$: number of arms, $j$: internal rank of a player, $\hat{M}$: number of players, $\tilde{k}$: a good arm, $\tau$: sampling times
\ENSURE $f$ an arm amongst the $M$ best arms assigned to the player
\STATE Initialize $p \gets 0$; $f \gets -1$;  {\it \# initialization}
\STATE $R[k], v[k] \gets 0$ for $k=1,...,K$ {\it \# rewards and number of samples for each arm}
\STATE {\it \# rewards and number of samples for each arm held by each players, only stored by the leader}
\IF{$j=1$}
    \for{$m=1,...,\hat{M}$ and $k=1,...,K$}{$\hat\mu[k,m],N[k,m] \gets 0$}
\ENDIF
\STATE $M' \gets \hat{M}$, ${\cal K} \gets \{1,\dots,K\}$ {\it \# number of active players and set of active arms}
\WHILE{$f = -1$}
\STATE $p \pluseq 1$ {\it \# start phase $p$}
\STATE $k \gets j$ {\it \# first sub-phase explore arms by sequential hopping}
\FOR{$t \gets 1,\dots, |\mathcal K| 2^{p} \left\lceil  \ln {1 \over \delta} \right\rceil$}
\STATE $k \gets (k + 1) \mod |\mathcal K|$ 
\STATE Select arm $k$, observe reward $r$, $R[k] \pluseq  r$, $v[k]\pluseq 1$, $E[k] \gets {R[k] \over v[k]}$ 
\ENDFOR
\STATE $Q \gets \lceil {p \over 2} + 3 \rceil$ {\it \# second sub-phase: share estimates between players}
\IF{$j = 1$}
\STATE $(f,{\cal K},M',\hat{\mu},N) \gets  $\alg{ComLeader}$(\hat{\mu},N,{\cal K},M',Q,\tau,$ $ \tilde{k},p,\delta)$ {\# player is a leader}
\ELSE 
\STATE $(f,\mathcal {\cal K},M') \gets $\alg{ComFollow}$(E,j,\mathcal {\cal K},M',Q,\tau,\tilde{k})$  {\# player is a follower}
\ENDIF
\ENDWHILE
\end{algorithmic}
\end{algorithm}
```

#### 5. Proposed algorithm (for player $m=1,...,M$)

Source lines: `papers/(2022) Towards Optimal Algorithms for Multi-Player Bandits without Collision Sensing Information/arXiv-2103.13059v2/main.tex:322-333`

```tex
\begin{algorithm}
\caption{Proposed algorithm (for player $m=1,...,M$)}
\label{alg:proposed}
\begin{algorithmic}
\REQUIRE $K$: number of arms, $\delta$: confidence level 
\ENSURE $\bar{k}$ an arm amongst the $M$ best arms assigned to the player
\STATE $(\tilde{k}, \tilde{\mu}) \gets $\alg{FindGoodArm}$(K,\delta)$  {\it \# find a good arm and a lower bound on its reward}
\STATE $s \gets $\alg{VirtualMusicalChairs}$(K, \tilde{k}, K \ln(\frac{1}{\delta})/\tilde{\mu})$  {\it \# assign external rank to each player}
\STATE $(\hat{M},j) \gets $\alg{VirtualNumberPlayers}$(K, \tilde{k}, s, \ln(\frac{1}{\delta}) / \tilde{\mu})$  {\it \# estimate the number of players and assign internal rank}
\STATE $\bar{k} \gets $\alg{DistributedExploration}$(K,j,\hat{M},\tilde{k}, \ln(\frac{1}{\delta}) / \tilde{\mu}))$ {\it \# find one arms out of the $M$ best arms}
\end{algorithmic}
\end{algorithm}
```

#### 6. \alg{ComLeader}

Source lines: `papers/(2022) Towards Optimal Algorithms for Multi-Player Bandits without Collision Sensing Information/arXiv-2103.13059v2/main.tex:569-612`

```tex
\begin{algorithm}[t]
\label{alg:communicationleader}
\caption{\alg{ComLeader}}
\begin{algorithmic}[H]
\REQUIRE 
$\hat{\mu}$: estimates held by each player, $N$: number of samples held by players, ${\cal K}$: set of active arms, $M'$: number of active players, $Q$ message length, $\tau$: sampling time, $\tilde{k}$: good arm, $p$: phase, $\delta$: confidence parameter
\ENSURE $f$ one of the $M$ best arms to be assigned to player, $\bar{{\cal K}}$ updated version set of active arms, $\bar{M}'$ updated number of active players, $\bar{\mu}$ updated estimates held by players, $\bar{N}$ updates number of samples held by players

\STATE $\bar{\mu} \gets \hat{\mu}$, $\bar{N} \gets N$
\STATE {\# \it receive the updated values of the estimates held by active players}
\FOR{$i \gets 2,\dots,M$} 
    \FOR{$k \in \mathcal K$}
        \STATE $\bar{\mu}[k,i] \gets $ \alg{DecoderReceiveFloat}$(\tilde{k}, \tau, Q)$ 
        \STATE $\bar{N}[k,i] \gets \bar{N}[k,i] + 2^{p} \left\lceil  \ln {1 \over \delta} \right\rceil$
    \ENDFOR
\ENDFOR
\STATE {\# \it compute the estimates aggregated across all players decide which arms to accept / reject}
\FOR{$k \in {\cal K}$}
    \STATE {\# \it compute the estimate aggregated across all players}
    \STATE $\rho[k] \gets (\sum_{i=1}^M \bar{\mu}[k,i] \bar{N})/(\sum_{i=1}^M \bar{N}[k,i])$
    \STATE {\# \it compute confidence radius}
    \STATE $B[k] \gets \sqrt{(2 \ln {1 \over \delta}) / (\sum_{i=1}^M \bar{N}[k,i])} + 2^{-{p \over 2} - 3}$
    \STATE {\# \it accept arm}
    \IF{$|\{i \in \mathcal K : \rho[k] - B[k] \geq  \rho[i] + B[i]\}| \geq |\mathcal K| - M'$}
        \STATE Add $k$ to $C[.,1]$
    \ENDIF    
    \STATE {\# \it reject arm}
    \IF{$|\{i \in \mathcal K : \rho[i] - B[i]  \geq  \rho[k]  + B[k] \}| \geq M'$}
        \STATE Add $k$ to $C[.,2]$
    \ENDIF
\ENDFOR
\STATE {\# \it message size to send accepted / rejected arms}
\STATE $Q' \gets \lceil \log_2 |{\cal K}|\rceil$
\STATE {\# \it send the size of the sets of accepted / rejected arms}
\FOR{$i \gets 2, \dots, M'$ and $s=1,2$}
\STATE \alg{EncoderSendInt}$(\mathcal K,\tilde{k}, \tau, Q', {\bf length}(C[.,s]))$
\ENDFOR
\STATE {\# \it send the contents of the sets of accepted / rejected arms}
\FOR{$i \gets 2, \dots, M'$ and $s=1,2$ and $k \in C[.,s]$}
\STATE \alg{EncoderSendInt}$(\mathcal K,\tilde{k},\tau,Q',k)$
\ENDFOR
\STATE $C' \gets C$ and remove $\tilde{k}$ from $C'[,.1]$
\end{algorithmic}
\end{algorithm}
```

#### 7. Uncaptioned algorithm block

Source lines: `papers/(2022) Towards Optimal Algorithms for Multi-Player Bandits without Collision Sensing Information/arXiv-2103.13059v2/main.tex:614-630`

```tex
\begin{algorithm}                     
\begin{algorithmic}
\IF{$M' = {\bf length}(C'[.,1])$ and $C=C'$}
    \STATE $f \gets C[M',1]$
\ELSIF{$M'-1 = {\bf length}(C'[.,1])$ and $C \ne C'$}
\STATE {\# \it assign the good arm to the leader}
    \STATE $f \gets \tilde{k}$
\ELSE
\STATE {\# \it otherwise make accepted and rejected arms inactive and update number of active players}
\STATE $\bar{M} \gets M' - {\bf length}(C[.,1])$
\STATE ${\mathcal K}' \gets {\mathcal K}$
\FOR{$k \in C'$}
    \STATE Remove $k$ from ${\cal K}'$
\ENDFOR
\ENDIF
\end{algorithmic}
\end{algorithm}
```

#### 8. \alg{ComFollow}

Source lines: `papers/(2022) Towards Optimal Algorithms for Multi-Player Bandits without Collision Sensing Information/arXiv-2103.13059v2/main.tex:632-678`

```tex
\begin{algorithm}[t]
\caption{\alg{ComFollow}}
\begin{algorithmic}[H]
\REQUIRE $E$: estimated reward of each arm by the player, $j$: rank of the player, ${\cal K}$: set of active arms, $M'$: number of active players, $Q$: message size, $\tau$: sampling time, $\tilde{k}$: a good arm
\ENSURE ,
$f$ one of the $M$ best arms to be assigned to player, $\bar{{\cal K}}$ updated version set of active arms, $\bar{M}'$ updated number of active players
\STATE ${\cal K}' \gets {\cal K} \setminus \{ \tilde{k}\}$
\STATE {\# \it send reward estimates of active arms to the leader}
\FOR{$i = 2,\dots,M'$}
\IF{$j = i$}
\FOR{$k \in {\cal K}$}
\STATE \alg{EncoderSendFloat}$({\cal K},\tilde{k},\tau,Q, E[k])$
\ENDFOR
\ELSE
\FOR{$t=1,..., |{\cal K}| \tau Q$}
    \STATE Select the $(j \mod |{\cal K}'|)$-th  arm in set ${\cal K}'$
\ENDFOR
\ENDIF
\ENDFOR
\STATE {\# \it message size to send accepted / rejected arms}
\STATE $Q' \gets \lceil \log_2 |{\cal K}|\rceil$
\STATE {\# \it receive the sizes of the sets of accepted and rejected arms}
\FOR{$i \gets 2, \dots, M'$}
\IF{$j = i$}
\FOR{$s=1,2$}
    \STATE $N[s] \gets$ \alg{DecoderReceiveInt}$(\tilde{k}, \tau, Q'$)
\ENDFOR
\ELSE
\FOR{$t=1,...,2 \tau Q'$}
    \STATE Select the $(j \mod |{\cal K}'|)$-th  arm in set ${\cal K}'$
\ENDFOR
\ENDIF
\ENDFOR
\STATE {\# \it receive the contents of the sets of accepted and rejected arms}
\FOR{$i \gets 2, \dots, M'$}
\IF{$j = i$}
\FOR{$s=1,2$ and $q=1,...,N[s]$}
        \STATE $C[q,s] \gets$ \alg{DecoderReceiveInt} $(\tilde{k},\tau,Q')$ 
\ENDFOR
\ELSE
\FOR{$t=1,...,\tau Q'(N[1]+N[2])$}
    \STATE Select the $(j \mod |{\cal K}'|)$-th  arm in set ${\cal K}'$
\ENDFOR
\ENDIF
\ENDFOR
\end{algorithmic}
\end{algorithm}
```

#### 9. Uncaptioned algorithm block

Source lines: `papers/(2022) Towards Optimal Algorithms for Multi-Player Bandits without Collision Sensing Information/arXiv-2103.13059v2/main.tex:680-700`

```tex
\begin{algorithm}
\begin{algorithmic}

\STATE {\# \it update the set of active arms, and the active players}
\STATE {\# \it avoid assigning the good arm to followers}
\STATE $C \gets C'$
\STATE Remove $\tilde{k}$ from $C'[.,1]$
\STATE {\# \it if an accepted arm can be assigned to player then do so}
\IF{$M' - j + 1 \leq {\bf length}(C'[.,1])$}
\STATE $f \gets C'[M'-j+1,1]$
\STATE {\# \it otherwise make accepted and rejected arms inactive and update number of active players}
\ELSE
\STATE $\bar{M} \gets M' - {\bf length}(C'[.,1])$
\STATE $\bar{\mathcal K} \gets {\mathcal K}$
\FOR{$k \in C'$}
    \STATE Remove $k$ from $\bar{\mathcal K}$
\ENDFOR
\ENDIF
\label{alg:communicationfollow}
\end{algorithmic}
\end{algorithm}
```

#### 10. \alg{EncoderSendFloat}

Source lines: `papers/(2022) Towards Optimal Algorithms for Multi-Player Bandits without Collision Sensing Information/arXiv-2103.13059v2/main.tex:703-725`

```tex
\begin{algorithm}[t]
\caption{\alg{EncoderSendFloat}}
\begin{algorithmic}[H]
\REQUIRE ${\mathcal K}$: a subset of arms, $\tilde{k}$: a good arm, $\tau$: a sampling time, $Q$: the message size, $\mu \in [0,1]$ a real number to send
\STATE {\# \it convert the number to send to a binary message of size $Q$}
\STATE $S \gets $\alg{FloatToBinary}$(\mu,Q)$ 
\STATE {\# \it send each bit of the binary message}
\FOR{$q=1,\dots, Q$}
    \IF{$S[q] = 1$}
		\STATE {\# \it send a $1$ bit}        
        \STATE $\ell \gets q \mod |\mathcal K \backslash \tilde{k}|$
        \STATE $k \gets$ the $\ell$-th arm in set ${\cal K} \backslash \tilde{k}$ 
    \ELSE
        \STATE {\# \it send a $0$ bit}
        \STATE $k \gets \tilde{k}$
    \ENDIF
    \FOR{$t=1,\dots,\tau$}
        \STATE Select arm $k$ 
    \ENDFOR
\ENDFOR
\label{alg:encodersendfloat}
\end{algorithmic}
\end{algorithm}
```

#### 11. \alg{DecoderReceiveFloat}

Source lines: `papers/(2022) Towards Optimal Algorithms for Multi-Player Bandits without Collision Sensing Information/arXiv-2103.13059v2/main.tex:727-747`

```tex
\begin{algorithm}[t]
\caption{\alg{DecoderReceiveFloat}}
\begin{algorithmic}[H]
\REQUIRE $\tilde{k}$: a good arm, $\tau$: a sampling time, $Q$: the message size
\ENSURE $\mu \in [0,1]$ a received real number to send
\STATE {\# \it decode each bit of the binary message}
\FOR{$q=1,\dots, Q$}
    \STATE $B[q] \gets 0$
    \FOR{$t=1,\dots,\tau$}
        \STATE Select arm $\tilde{k}$ and observe reward $r$
        \STATE {\# \it decode a $1$ bit if at least one non zero reward is obtained}
        \IF{$r > 0$}
            \STATE $B[q] \gets 1$
        \ENDIF
    \ENDFOR
\ENDFOR
\STATE {\# \it convert the received binary message to a real number}
\STATE $\mu \gets $\alg{BinaryToFloat}$(B,Q)$
\label{alg:decoderreceivefloat}
\end{algorithmic}
\end{algorithm}
```

## Extraction Summary

Total algorithm blocks: 11
