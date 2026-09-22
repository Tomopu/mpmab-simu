# HDE のサブリーダー段の通信時間の課金を論文に合わせる（2026-09-22）

## 何が違っていたか

`HierarchicalDistributedExploration`（Algorithm 4）の通信サブフェーズは、実際の bit 伝送を行わず、
時間コストだけをダミー step で消費する簡略実装である。このうちサブリーダー段
（Sub-Leader → Grand Leader の上りと、その逆向きの下り）の本数を、旧実装は

```
n_subleaders = sum(1 for m in range(M) if 2 <= j_list[m] <= n and f[m] == -1)
```

のように**未割当のサブリーダーだけ**数えていた。

論文は違う。補足 B の AssignAndUpdate の説明は
「assigned subleaders and the grand leader keep their communication role, namely aggregation,
relay, and the test, until the assignment of all players is fixed」と述べ、補足 D の Lemma 4 の
証明 (3) はサブリーダー段を毎フェーズ `(n-1)|K|Qτ` と数えている。

主設定（K=20, M=8）では通信路が上位 4 本なので、サブリーダーは早いフェーズで割当が確定する。
以後、境界の腕（μ=0.58 対 0.52）を分けるまで続く多くのフェーズで、旧実装はサブリーダー段を
課金していなかった。この差は n=1 にはサブリーダーがいないので現れず、
「n=1 が Huang 2022 と完全一致する」という検証では検出できない。

## 直したこと

- `n_subleaders = max(0, n - 1)` にした。割当済みかどうかによらず、HDE が終わるまで毎フェーズ課金する。
- フォロワー段は変えていない（未割当のフォロワーだけが送る。Huang 2022 の ComFollow と同じ）。
- 検証用に `hde_comm_log`（フェーズごとの内訳）を追加した。乱数は消費しない。
- `tests/algorithms/homogeneous/izumi2026/test_algorithm4_relay_cost.py` を追加した。

## 結果への影響

修正前の実装で、主設定・降順・10 試行について「数え落とした中継ステップ」を数えると、
n=4 で 49,929 ステップ（通信中は全員がダミー腕を引くので 1 ステップの損失は上位 8 本の期待値の和 6.23、
換算で約 311,000 のリグレット）だった。n=1 に対する削減率は約 50.8% から約 35.7% に下がる。

**`outputs/camera_ready_20260917/` 以下の Huang 2022・Izumi 2026 の結果は旧実装の値である。**
Randomized Selfish KL-UCB の結果はこの修正の影響を受けない。修正後の結果は
`outputs/camera_ready_20260922/` 以下に置く。

## 関連

- `reviews/codex_review_verification.md`（論文リポジトリ）の指摘 2 と実験 B
