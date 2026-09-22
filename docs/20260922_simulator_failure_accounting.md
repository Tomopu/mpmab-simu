# 失敗した試行の評価を直す（F24、2026-09-22）

Codex のシミュレーターレビュー（論文リポジトリ `reviews/simulator_code_review.md`）が挙げた 5 件への対応。
**9 月 22 日の主実験（`outputs/camera_ready_20260922/`）の数値は変わらない**（保存 CSV に該当する試行がない）。
失敗を含む実験を正しく評価するための修正である。

## 直した 5 件

| # | 問題 | 対応 |
|---|---|---|
| 1 | 割当が決まると残りの horizon を引かず、曲線を水平に延ばす。誤った割当の残り損失が落ちる | `compute_metrics(..., horizon=T)` が「残り時間 × 固定割当の期待損失」を `expected_tail_regret` として `cumulative_regret` に加える。正しい割当なら 0。重複した腕は衝突で報酬 0、未割当は報酬 0 として数える。`regret_at_stop` に止めた時点の値を残す。曲線は `build_curve_rows(..., stop_time, tail_loss_per_step)` で同じ傾きで延ばす |
| 2 | FindMultipleGoodArms がプレイヤー 0 の G だけを返し、不一致が記録されない | 各プレイヤーの G（順序つき）と各腕の下界を `fmga_player_good_arms` / `fmga_player_mu_tilde` に残し、`PlayerStateIzumi.good_arms` と `mu_tilde` に自分の値を入れる。`compute_metrics` が `good_arm_set_agreement`（集合の一致）と `initialization_agreement`（通信路の順序つきリストと各腕の下界の一致）を計算し、成功判定には後者を使う。Huang 2022 も同様（`fga_player_k_tilde` / `fga_player_mu_tilde`） |
| 3 | 内部ランクの誤推定で j=1 がいないと ValueError で試行が止まる | `run()` が VMC・VNP・HDE の ValueError を捕まえ、`init_failure_reason` を結果に入れて返す。以後は未割当のまま止まるので、1 の規則で残り時間は最大損失になる |
| 4 | 重複割当を成功と数える | `final_assignment_success` に `not assignment_duplicate` を加えた |
| 5 | n ≤ M、2n ≤ K を検査しない | コンストラクターが既定で拒む。`allow_out_of_range_n=True`（CLI は `--allow-out-of-range-n`）で許し、summary の `n_within_assumptions=0` で区別する |

## 不一致時の評価規約（2026-09-22 追記）

VMC・VNP・HDE は `good_arms[j-1]` の**順序**で通信路を選び、τ は下界から計算するので、集合が一致しても順序か下界が違えば同じプロトコル状態ではない。そこで

- `good_arm_set_agreement`：G の集合が全員で一致したか（弱い指標、参考用）
- `initialization_agreement`：通信路の順序つきリストと各腕の下界が全員で一致したか（成功判定に使う）

を分けて記録する。**不一致を検出した試行は、FindMultipleGoodArms の直後に失敗として止め**（`init_failure_reason` に理由を残す）、残り時間は未割当として最大損失で数える。これはプレイヤーが使える通信操作ではなく、評価用の保守的な規約である。不一致のままプレイヤー 0 の G で続けた軌跡は元の分散アルゴリズムを表さないので評価しない。なお FindMultipleGoodArms の内部では各フェーズの終わりにプレイヤー 0 の確定集合で active 集合を更新しているので、検査は関数の終了時に行う。完全な分散実装（各プレイヤーのローカル状態で個別に進める）ではない。

## summary.csv に増えた列

`good_arm_set_agreement`、`initialization_agreement`、`regret_at_stop`、`expected_tail_regret`、`init_failure_reason`、`n_within_assumptions`。

## 数値が変わらないことの確認

主設定（K=20, M=8, T=10^6、降順）の trial 0〜9、Huang 2022 と Izumi 2026 の n=1〜4（計 50 行）を
修正後のコードで実行し、`outputs/camera_ready_20260922/` の同じ行と比べた（2026-09-22）。
`cumulative_regret`、`init_duration`、`find_good_duration`、`rank_duration`、`number_players_duration`、
`collision_count`、`final_assignment_success` の不一致は 0 行。全行で `initialization_agreement=1`（当時の列名は `good_arm_agreement`）、
`expected_tail_regret=0`、`n_within_assumptions=1`。乱数の消費は変えていないので、主実験の値は変わらない。

Codex のレビューが挙げた再現条件（δ=0.9）で修正後の動作も確かめた。seed 0 の誤った割当 [2,0] では
`expected_tail_regret=9,667.5` が加わり `cumulative_regret=10,457.0`、seed 52 の G の不一致は
`initialization_agreement=0` で失敗、seed 2 の j=1 なしは例外にならず `init_failure_reason` に残り、
(K, M, n)=(7, 2, 4) はコンストラクターが拒む。

## 実行の注意

`(K, M) = (10, 4)` で n=5 を走らせるには `--allow-out-of-range-n` が要る（論文では (A1) の範囲外の参考値として扱っている）。

## 関連

- `reviews/codex_second_review_verification.md`（論文リポジトリ）第 3 章
- `tests/invariants/test_failure_accounting.py`
