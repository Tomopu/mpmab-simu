# Paper Pseudocode References

このディレクトリには、MPMAB シミュレーター実装で参照する論文擬似コードをまとめています。

PDF ではなく TeX の `algorithm` 環境を抽出しているため、Claude Code に実装を依頼するときはこのディレクトリの Markdown を優先して参照させます。

## Files

- [20260510_huang2022_pseudocode.md](20260510_huang2022_pseudocode.md)
  - Homogeneous MPMAB without collision sensing。
  - 初回実装の single-channel baseline。

- [20260510_izumi2026_pseudocode.md](20260510_izumi2026_pseudocode.md)
  - Homogeneous multi-channel MPMAB。
  - `data.tex` に記述された parallel 版のみ。
  - 初回実装の proposed multi-channel model。

- [20260510_shi2021_beacon_pseudocode.md](20260510_shi2021_beacon_pseudocode.md)
  - Heterogeneous MPMAB / BEACON。
  - 後続実装で使う。

- [20260510_wang2020_orthogonalization_pseudocode.md](20260510_wang2020_orthogonalization_pseudocode.md)
  - Orthogonalization, rank assignment, DPE。
  - collision-sensing initialization の参照用。

## Notes

- Python 実装では arm/player index を 0-based に統一する。
- TeX 内の擬似コードは基本的に 1-based index で書かれている。
- 本文説明しか存在しない初期化手順は、実装用 pseudocode として書き起こしている。
