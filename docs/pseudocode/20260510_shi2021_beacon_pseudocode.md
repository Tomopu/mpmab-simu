# Shi 2021 Pseudocode: BEACON for Heterogeneous MPMAB

作成日: 2026-05-10

このファイルは Claude Code が PDF を読みに行かずに実装できるよう、TeX に記述された擬似コードを論文ごとに集約したものです。

## Source Files

- `papers/(2021) Heterogeneous Multi-player Multi-armed Bandits Closing the Gap and Generalization/arXiv-2110.14622v2/CR_BEACON_NeurIPS.tex`

## Implementation Notes

- 対象モデル: heterogeneous MPMAB with collision sensing。
- BEACON は player-arm 平均報酬行列 `mu[m, k]` を前提にする。
- 初期化の orthogonalization / rank assignment は本文説明であり、TeX の `algorithm` 環境としては載っていないため、下の実装用 pseudocode を使う。
- BEACON の主処理は leader / follower / send / receive の `algorithm` 環境として載っている。
- `Oracle(mu_bar)` は matching oracle。linear reward なら最大重み matching として実装できる。
- 通信は collision sensing を使う。bit 1 は受信側通信 arm に collision を作る、bit 0 は送信側が自分の通信 arm に退避する。

## Synthesized Initialization Pseudocode

The TeX source describes orthogonalization and rank assignment in prose. Use this implementation-oriented pseudocode for BEACON initialization.

### Orthogonalization

```text
Input: K arms, unknown M players, collision sensing enabled
Output per player: external rank/state s in {1, ..., K}

state <- 0  # 0 means unsettled
repeat blocks of K + 1 rounds:
  if state == 0:
    candidate <- uniformly random arm in {1, ..., K}
    pull candidate
    if no collision:
      state <- candidate
  else:
    pull state

  if using reserved broadcast arm:
    if state == 0:
      pull broadcast arm during all K broadcast rounds
    else:
      pull state except in the state-th broadcast round, where pull broadcast arm
    if no broadcast collision is observed in the block:
      terminate orthogonalization
  else using sequential broadcast over arms 1..K:
    implement equivalent block-level test that all players are settled
    terminate when the block indicates no unsettled player
```

### Rank Assignment

```text
Input per player: unique external state s in {1, ..., K}
Output per player: internal rank m in {1, ..., M}, estimate M_hat

M_hat <- 0
rank <- 1
for block k in 1..K:
  for round q in 1..K:
    if state == k:
      pull arm q
    else:
      pull own state
  if a collision pattern indicates that state k exists:
    M_hat <- M_hat + 1
    if k < state:
      rank <- rank + 1
return M_hat, rank
```

## Extracted TeX Pseudocode

### `papers/(2021) Heterogeneous Multi-player Multi-armed Bandits Closing the Gap and Generalization/arXiv-2110.14622v2/CR_BEACON_NeurIPS.tex`

#### 1. BEACON: Leader

Source lines: `papers/(2021) Heterogeneous Multi-player Multi-armed Bandits Closing the Gap and Generalization/arXiv-2110.14622v2/CR_BEACON_NeurIPS.tex:208-237`

```tex
	\begin{algorithm}[htb]
	    \caption{BEACON: Leader}
	    \label{alg:leader}
	    \begin{algorithmic}[1]
		\State Initialization: $r\gets 0$; $\forall (k,m), p^r_{k,m}\gets -1, T^r_{k,m}\gets 0, \tilde{\mu}^r_{k,m}\gets 0$
		\State Play each arm $k\in[K]$ and $T^{r+1}_{k,1}\gets T^r_{k,1}+1$
		\While{not reaching the time horizon}
		\State $r\gets r+1$
		\State $\forall (k,m), p^r_{k,m}\gets \left\lfloor\log_2(T^r_{k,m})\right\rfloor$
		\State $\forall k\in[K]$, update sample mean $\hat{\mu}^{r}_{k,1}$ with the first $2^{p^r_{k,1}}$ exploratory samples from arm $k$
		\Statex $\triangleright$ \textit{Communication Phase}
		\For{$(k,m)\in[K]\times [M]$} 
		\If{$p^r_{k,m}>p^{r-1}_{k,m}$}
		\State $\tilde{\delta}^r_{k,m}\gets \texttt{Receive}(\tilde{\delta}^r_{k,m},m)$
		\State $\tilde{\mu}_{k,m}^r\gets \tilde{\mu}_{k,m}^{r-1}+\tilde{\delta}_{k,m}^r$
		\Else 
		\State $\tilde{\mu}_{k,m}^r\gets \tilde{\mu}_{k,m}^{r-1}$
		\EndIf
		\EndFor
		\State $\forall (k,m), \bar{\mu}^r_{k,m}\gets \tilde{\mu}_{k,m}^r+\sqrt{3\ln t_r/2^{p^r_{k,m}+1}}$
		\State $S_r=[s^r_1,...,s^r_M]\gets \texttt{Oracle}(\boldsymbol{\bar{\mu}_r})$
		\State $\forall m\in [M], \texttt{Send}(s^r_m,m)$
		\Statex $\triangleright$ \textit{Exploration Phase}
		\State $p_r \gets \min_{m\in[M]}p^r_{s^r_m,m}$
		\State Play arm $s^r_1$ for $2^{p_r}$ times
		\State Signal followers to stop exploration
		\State Update $\forall m\in[M], T^{r+1}_{s_m,m}\gets T^r_{s_m,m}+2^{p_r}$
		\EndWhile
		\end{algorithmic}
	\end{algorithm}
```

#### 2. BEACON: Follower $m$

Source lines: `papers/(2021) Heterogeneous Multi-player Multi-armed Bandits Closing the Gap and Generalization/arXiv-2110.14622v2/CR_BEACON_NeurIPS.tex:466-492`

```tex
\begin{algorithm}[htb]
	\caption{BEACON: Follower $m$}
	\label{alg:follower}
	\begin{algorithmic}[1]
		\State Set epoch counter $r\gets 0$; arm counter $[p^r_{k,m}]_{k\in[K]}\gets 0$; sample time $[T^r_{k,m}]_{k\in[M]}\gets 0$; communicated statistics $[\tilde{\mu}^r_{k,m}]_{k\in[M]}\gets 0$
		\State In order $k\in [K]$, play arm $[(m-1+k) \text{ mod } K]$ once  and update sample time $T^{r+1}_{k,m}\gets T^r_{k,m}+1$
		\While{not reaching the time horizon $T$}
		\State $r\gets r+1$
		\State $\forall k\in[K], p^r_{k,m}\gets \left\lfloor\log_2(T^r_{k,m})\right\rfloor$
		\State Update $\hat{\mu}^{r}_{k,m}$ as the sample mean from the first $2^{p^r_{k,m}}$ exploratory samples from arm $k$
		\Statex $\triangleright$ \textit{Communication Phase}
		\For{$k\in [K]$} 
		\If{$p^r_{k,m}>p^{r-1}_{k,m}$}
		\State {$\tilde{\mu}^r_{k,m} \gets \texttt{ceil}(\hat{\mu}^r_{k,m})$ with $\lceil 1+ p^r_{k,m}/2\rceil$} bits
		\State $\tilde{\delta}^r_{k,m}\gets \tilde{\mu}_{k,m}^r - \tilde{\mu}_{k,m}^{r-1}$
		\State $\texttt{Send}(\tilde{\delta}^r_{k,m}, 1)$
		\Else
		\State $\tilde{\mu}_{k,m}^r\gets \tilde{\mu}_{k,m}^{r-1}$
		\EndIf
		\EndFor
		\State $s^r_m \gets \texttt{Receive}(s^r_m,1)$
		\Statex $\triangleright$ \textit{Exploration Phase}
		\State Play arm $s^r_m$ until signaled
		\State Update $T^{r+1}_{s^r_m,m}\gets T^r_{s^r_m,m}+2^{p_r}$
		\EndWhile
	\end{algorithmic}
\end{algorithm}
```

#### 3. $\texttt{Send()}$ for Player $m$

Source lines: `papers/(2021) Heterogeneous Multi-player Multi-armed Bandits Closing the Gap and Generalization/arXiv-2110.14622v2/CR_BEACON_NeurIPS.tex:497-511`

```tex
	\begin{algorithm}[htb]
	    \caption{$\texttt{Send()}$ for Player $m$}
	    \label{alg:send}
	    \begin{algorithmic}[1]
		\Require bit string $\boldsymbol{u} = [u_1, u_2, ..., u_{l}]$ with length $l$, receiver index $n$
		\State Initialization: player $m$'s communication arm $c_m$, player $n$'s communication arm $c_n$
		\For{$i = 1, 2, \cdots , l$}
		\If{$u_i=1$}
		\State Pull arm $c_n$ \Comment{collision for bit $1$}
		\Else
		\State Pull arm $c_m$ \Comment{no collision for bit $0$}
		\EndIf
		\EndFor
		\end{algorithmic}
	\end{algorithm}
```

#### 4. $\texttt{Receive()}$ for Player $n$

Source lines: `papers/(2021) Heterogeneous Multi-player Multi-armed Bandits Closing the Gap and Generalization/arXiv-2110.14622v2/CR_BEACON_NeurIPS.tex:512-528`

```tex
	\begin{algorithm}[htb]
	    \caption{$\texttt{Receive()}$ for Player $n$}
	    \label{alg:receive}
	    \begin{algorithmic}[1]
		\Require bit string $\boldsymbol{u}'$ with length $l$, sender index $m$
		\State Initialization: player $n$'s communication arm $c_n$
		\For{$i = 1, 2, \cdots , l$}
		\State Pull arm $c_n$
		\If{ collision}
		\State $u'_i \gets 1$ \Comment{collision for bit $1$}
		\Else
		\State $u'_i \gets 0$ \Comment{no collision for bit $0$}
		\EndIf
		\EndFor
		\Ensure $\boldsymbol{u}'$
		\end{algorithmic}
	\end{algorithm}
```

## Extraction Summary

Total algorithm blocks: 4
