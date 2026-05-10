# 論文まとめ

## はじめに

現在、私は (2026) Multi-Channel Communication Algorithm for Multi-Player Multi-Armed Bandits without Collision Sensing という研究を行っています。

参考にしている MPMAB (Multi-Player Multi-Armed Bandits) の研究は 2つある。1つ目は (2021) Heterogeneous Multi-player Multi-armed Bandits Closing the Gap and Generalization、2つ目は (2022) Towards Optimal Algorithms for Multi-Player Bandits without Collision Sensing Information である。

今回は、これらの研究のアルゴリズムをもとに、MPMAB のシミュレーターを作成したい。
README.md に記載されているディレクトリ構造と、以下「参考文献一覧」の論文およびコード (Tex) を参考に、各アルゴリズムのコードを読み取り、Pythonでどう実装すればいいのかを考えて欲しい。

ただし、[私の論文](papers/(2026) Multi-Channel Communication Algorithm for Multi-Player Multi-Armed Bandits without Collision Sensing/C000063_Izumi_fin.pdf) に書かれているコードは、(2022) Towards Optimal Algorithms for Multi-Player Bandits without Collision Sensing Information のコードをベースに、複数の通信チャネルに拡張したものであり、(2021) Heterogeneous Multi-player Multi-armed Bandits Closing the Gap and Generalization のコードとは異なるため、(2021) のコードの複数チャネルへの改良版も別途作成しなければならない。

具体的には、(2021) Heterogeneous Multi-player Multi-armed Bandits Closing the Gap and Generalization での [orthogonalization procedure (Wang et al., 2020) の部分]((2020) An Optimal Algorithm for Multiplayer Multi-Armed Bandits) を、(2022) Towards Optimal Algorithms for Multi-Player Bandits without Collision Sensing Information での Algorithm 1. FindGoodArm、Algorithm 2. VirtualMusicalChairs、Algorithm 3. VirtualNumberPlayers をHeterogeneous Multi-player Multi-armed Bandits に拡張したものを実装しようと考えている。

つまり、私は4つの論文に記載されているアルゴリズムから、4つの問題設定に関するモデルを作成しようとしている。

### 作成しようとしている4つのモデル

1. Homogeneous な MPMAB without Collision Sensing (no-sensing) のモデル [Huang et.al., 2022]
    - 初期化フェーズ： [Huang et.al., 2022]
       - Algorithm 1. FindGoodArm
       - Algorithm 2. VirtualMusicalChairs
       - Algorithm 3. VirtualNumberPlayers
    - 学習フェーズ： [Huang et.al., 2022]
       - Algorithm 4. DistributedExploration

2. Homogeneous かつ Multi-Channel な MPMAB without Collision Sensing のモデル [Izumi et.al., 2026]
    - 初期化フェーズ： [Izumi et.al., 2026]
       - Algorithm 1. FindMultipleGoodArms
       - Algorithm 2. ParallelVirtualMusicalChairs
       - Algorithm 3. ParallelVirtualNumberPlayers
    - 学習フェーズ： [Izumi et.al., 2026]
       - Algorithm 2. HierarchicalDistributedExploration

3. Heterogeneous な MPMAB with Collision Sensing (collision-sensing) のモデル [Shi et.al., 2021]
    - 初期化フェーズ [Wang et.al., 2020]：
       - Orthogonalization
       - Rank assignment
    - 学習フェーズ： [Shi et.al., 2021]
       - Algorithm 1 BEACON: Leader
       - Algorithm 2 BEACON: Follower m

4. Heterogeneous かつ Multi-Channel な MPMAB with Collision Sensing のモデル [Izumi et.al., 2026] (コード未設定中)
    - 初期化フェーズ (Heterogeneous版に改良必要) [Izumi et.al., 2026]：
       - Algorithm 1. FindMultipleGoodArms
       - Algorithm 2. ParallelVirtualMusicalChairs
       - Algorithm 3. ParallelVirtualNumberPlayers
    - 学習フェーズ ([Shi et.al., 2021] を leader, sub-leader, follower の n グループ版に改良) [Izumi et.al., 2026]：
       - Algorithm 1 ParallelBEACON: Leader
       - Algorithm 2 ParallelBEACON: Follower m

まずは、「作成しようとしている4つのモデル」のうち、1. Homogeneous な MPMAB without Collision Sensing (no-sensing) のモデル [Huang et.al., 2022] と 2. Homogeneous かつ Multi-Channel な MPMAB with Collision Sensing のモデル [Izumi et.al., 2026] のコードを実装して、その性能差を比較してみたいと考えている。

その実装が終わった後は、3. と 4. のモデルのコードを実装して、性能差を比較してみたいと考えている。

## 参考文献一覧

1. (2020) An Optimal Algorithm for Multiplayer Multi-Armed Bandits
    - [論文](papers/(2020) An Optimal Algorithm for Multiplayer Multi-Armed Bandits/1909.13079v2.pdf)
    - [コード](papers/(2020) An Optimal Algorithm for Multiplayer Multi-Armed Bandits/arXiv-1909.13079v2/main.tex)

2. (2021) Heterogeneous Multi-player Multi-armed Bandits Closing the Gap and Generalization
   - [論文](papers/(2021) Heterogeneous Multi-player Multi-armed Bandits Closing the Gap and Generalization/2110.14622v2.pdf)
   - [コード](papers/(2021) Heterogeneous Multi-player Multi-armed Bandits Closing the Gap and Generalization/arXiv-2110.14622v2/CR_BEACON_NeurIPS.tex)

3. (2022) Towards Optimal Algorithms for Multi-Player Bandits without Collision Sensing Information
    - [論文](papers/(2022) Towards Optimal Algorithms for Multi-Player Bandits without Collision Sensing Information/2103.13059v2.pdf)
    - [コード](papers/(2022) Towards Optimal Algorithms for Multi-Player Bandits without Collision Sensing Information/arXiv-2103.13059v2/main.tex)

4. (2026) Multi-Channel Communication Algorithm for Multi-Player Multi-Armed Bandits without Collision Sensing
    - [論文](papers/(2026) Multi-Channel Communication Algorithm for Multi-Player Multi-Armed Bandits without Collision Sensing/C000063_Izumi_fin.pdf)
    - [コード](papers/(2026) Multi-Channel Communication Algorithm for Multi-Player Multi-Armed Bandits without Collision Sensing/src/data.tex)