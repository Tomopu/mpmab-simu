# Fallback Inventory

作成日: 2026-05-11

このドキュメントは、論文手順から外れる安全装置 fallback の所在を記録する。現在の通常実行経路では、割当を補完する fallback は呼び出さない。割当失敗は実験結果の `final_assignment_success=0` として記録し、必要に応じて実験スクリプト側の retry 機能で再試行する。

## 実験側の再試行

`experiments/compare_homogeneous.py` に以下のオプションを追加した。

```bash
python experiments/compare_homogeneous.py \
  --experiment speedup \
  --trials 50 \
  --horizon 1000000 \
  --retry-on-failure \
  --max-attempts 5
```

- `--retry-on-failure`: `final_assignment_success=0` の trial を seed を変えて再実行する。
- `--max-attempts`: 1 trial あたりの最大 attempt 数。
- 成功した attempt だけを summary / curve CSV に保存する。
- 全 attempt が失敗した場合は最後の失敗 attempt を保存し、`retry_exhausted=1` を立てる。

この方針により、アルゴリズム本体で失敗を隠さず、長時間実験では成功 trial を自動採用できる。

## 残している fallback

| 場所 | 関数 | 現在の通常経路 | 内容 |
|------|------|----------------|------|
| `algorithms/homogeneous_huang2022.py` | `_fallback_assign_remaining_accepts` | 未使用 | 未使用の accepted arm を未割当 player に rank 降順で補完割当する旧実験用処理 |
| `algorithms/homogeneous_multichannel_izumi2026.py` | `_fallback_assign_remaining_accepts` | 未使用 | Huang 版と同じ補完割当の Izumi 版 |
| `algorithms/homogeneous_multichannel_izumi2026.py` | `_try_assign_with_accepted_good_arm_fallback` | 未使用 | `C_accept - G` で割当できない場合に accepted good arms も割当候補に含める旧実験用処理 |
| `algorithms/homogeneous_multichannel_izumi2026.py` | `parallel_virtual_musical_chairs` 内の spreading 候補枯渇時処理 | 通常は到達しない | `n < K-M` の前提では候補が枯渇しない想定。防御的な分岐として残っている |

## 呼び出さない理由

Huang 2022 の ComLeader / ComFollow では、accepted arm で割当できない player はその場で補完されず、active set を更新して次 phase に進む。したがって `C_accept` 補完割当は論文手順ではない。

Izumi 2026 でも、現在の HDE はまだ forced-collision bit 通信を完全再現していない。ここで補完割当を使うと、通信・割当規則の未完成部分を成功率で隠してしまうため、通常経路から外している。

## 将来再利用する場合

fallback を使った実験を行う場合は、通常の論文再現結果とは別名の experiment / figure として保存する。論文再現の主結果には混ぜない。
