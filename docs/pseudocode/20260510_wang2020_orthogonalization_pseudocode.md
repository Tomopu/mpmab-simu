# Wang 2020 Pseudocode: DPE Orthogonalization and Exploration

作成日: 2026-05-10

このファイルは Claude Code が PDF を読みに行かずに実装できるよう、TeX に記述された擬似コードを論文ごとに集約したものです。

## Source Files

- `papers/(2020) An Optimal Algorithm for Multiplayer Multi-Armed Bandits/arXiv-1909.13079v2/main.tex`

## Implementation Notes

- 対象モデル: homogeneous collision-sensing MPMAB / DPE。
- 2020 論文の初期化は TeX の `algorithm` 環境ではなく本文説明で記述されているため、下の実装用 pseudocode を使う。
- DPE の exploration-exploitation phase は `algorithm2e` 形式の `algorithm` 環境として TeX に存在する。
- DPE は collision sensing を使い、leader から follower への implicit communication を collision で行う。
- README で参照している Wang et al. 2020 の orthogonalization procedure は、ここでは `DPE initialization` として整理する。

## Synthesized Initialization Pseudocode

The TeX source gives the initialization procedure in prose. Use this implementation-oriented pseudocode when implementing orthogonalization and rank assignment.

### Orthogonalization

```text
Input: K arms, unknown M players, collision sensing enabled
Output per player: unique external state s in {1, ..., K-1}

state <- 0  # 0 means unsatisfied
repeat blocks of K + 1 rounds:
  if state == 0:
    a <- uniformly random element of {1, ..., K-1}
    pull a
    if no collision:
      state <- a
  else:
    pull state

  for q in 1..K:
    if state == 0:
      pull K
    else if q == state:
      pull K
    else:
      pull state

  if no collision is observed in the K broadcast rounds:
    terminate
```

### Rank Assignment

```text
Input per player: unique state s in {1, ..., K-1}
Output per player: M_hat and internal rank rank in {1, ..., M}

M_hat <- 0
rank <- 1
for block k in 1..K-1:
  collision_seen_in_block <- false
  for q in 1..K-1:
    if state == k:
      pull q
    else:
      pull state
    if collision:
      collision_seen_in_block <- true

  if collision_seen_in_block:
    M_hat <- M_hat + 1
    if k < state:
      rank <- rank + 1

return M_hat, rank
```

### Communication When Best Empirical Set Changes

```text
When ordered set N(t) changes, leader communicates to followers.
For each follower rank i+1, use one block of M + K + 1 rounds:
  1. Signal start by colliding with that follower.
  2. Use next M rounds to communicate the rank in N(t) of the leaving arm.
  3. Use next K rounds to communicate the index of the entering arm.
Followers keep their old exploration-exploitation actions during communication and update after the communication block ends.
```

## Extracted TeX Pseudocode

### `papers/(2020) An Optimal Algorithm for Multiplayer Multi-Armed Bandits/arXiv-1909.13079v2/main.tex`

#### 1. The DPE algorithm: Exploration-exploitation phase

Source lines: `papers/(2020) An Optimal Algorithm for Multiplayer Multi-Armed Bandits/arXiv-1909.13079v2/main.tex:213-234`

```tex
\begin{algorithm}[htb]
	\SetAlgoLined
	\textbf{Initialization:} Set $\hat{\nu}(1)=d(1)=0$. Initialize the set of best empirical arms ${\cal N}(1)$  and $\hat{M}(1)$ arbitrarily.\\
	For round $t\ge 1$:\\
	{\bf Leader.}\\
	\qquad	1. If $t=0 (\hbox{mod }M)$, update $\hat{\nu}_k(t)$, $d_k(t)$ for each arm $k$, and $\hat{M}(t)$\\
	\qquad	\quad update the ordered set ${\cal N}(t)\gets\left\{\ell_{1}(t),\ell_{2}(t),\ldots,\ell_{M}(t)\right\}$\\
	\qquad  \quad (the set of the $M$ best empirical arms)\\
	\qquad    2. If ${\cal N}(t)\neq {\cal N}(t-1)$, communicate ${\cal N}(t)$ to the followers\\
	\qquad   3. ${\cal B}(t)\gets \left\{k\notin {\cal N}(t):d_k(t)\geq \hat{\nu}_{\hat{M}(t)}(t)\right\}$;\\
		\qquad  \quad  $m\gets \left[(t+1)(\hbox{mod } M)\right] +1$\\
	\qquad  \quad If (${\cal B}(t)=\emptyset$ or $\ell_m(t)\neq \hat{M}(t)$), $\rho(t)\gets \ell_{m}(t)$\\
		\qquad  \quad Else \\
		\qquad  \quad \quad w.p. $1/2$, $\rho(t)\gets  \hat{M}(t)$\\
		\qquad  \quad \quad w.p. $1/2$, $\rho(t)\gets  k $ where $k$ is drawn from ${\cal B}(t)$
		    uniformly\\
		\qquad  \quad  Select arm $\rho(t)$\\
		
{\bf Follower with rank $i\in \{2,\ldots,M\}$.}\\
		  \qquad $m_i\gets \left[(t+i)(\hbox{mod } M)\right] +1$, Select arm ${\ell}_{m_i}(t)$\\
    \caption{The DPE algorithm: Exploration-exploitation phase}~\label{alg:main}
\end{algorithm}
```

## Extraction Summary

Total algorithm blocks: 1
