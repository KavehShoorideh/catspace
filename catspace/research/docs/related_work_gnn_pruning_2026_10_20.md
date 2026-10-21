# Related work: pruning and learned guidance over a growing heterogeneous AND-OR graph (2026-10-20)

A literature map made for plan 60 (the policy over the findings graph), at Kaveh's request: "can we prune a graph / subgraph based on something, so that
I don't multiply the attention weights there?" Compiled by a research agent from primary sources; items it could not verify are marked.

## The answer in one paragraph
Yes, and it is the cheap part. Message passing over an edge list (PyG `MessagePassing`, DGL) costs O(|E| d) over the edges that are present, so edges
removed from `edge_index` are never multiplied (`subgraph`, `k_hop_subgraph`, `HeteroData.subgraph`, `edge_type_subgraph`); any symbolic predicate can
drive the removal. A -inf mask in DENSE attention zeroes weights but still computes every QK^T entry and saves nothing, unless a block-sparse kernel
skips masked blocks (FlexAttention). A LEARNED pruner still visits every candidate; it saves compute only downstream of itself. Under the standing rule
(the learned part only orders): prune the SCORER'S CONTEXT, never the search's open set; a node leaves the open set only by a symbolic proof, refutation
or kill, so a pruning mistake costs efficiency, never the outcome.

## Closest analogues to plan 60
- RetroGraph (Xie et al., KDD 2022, arXiv:2206.11477): a GNN over the whole AND-OR search graph scores the open nodes; trained on A* snapshots with
  per-node cross-entropy plus a pairwise ranking loss (no code found).
- HTPS (Lample et al., NeurIPS 2022, arXiv:2205.11491): learned policy/value on single goals; the hypergraph bookkeeping is symbolic (solved 1,
  invalid 0, a tactic's value the product of its children's) -- our learned / symbolic split.
- Retro* (Chen et al., ICML 2020, arXiv:2006.15820): neural cost-to-go at OR nodes, symbolic aggregation, best-first.
- Labassi, Chetelat, Lodi (NeurIPS 2022, arXiv:2210.16934): a siamese GNN picks the next open branch-and-bound node.
- Gasse et al. (NeurIPS 2019, arXiv:1906.01629): a GNN imitates strong branching; candidates restricted symbolically.
- NGDS (Kalyan et al., ICLR 2018, arXiv:1804.01186): symbolic deduction prunes, the network ranks branches.
- LevinTS / PHS (Orseau et al., NeurIPS 2018 arXiv:1811.10928; AAAI 2021 arXiv:2103.11505): expansions bounded by the policy's probability of the
  solution -- the formal version of "the learned part only orders".
- Gap (not verified): no primary source found that masks GNN attention by proof status or by a read set; search systems do it by construction.

## Mechanisms
- Sparsification: NeuralSparse (ICML 2020), PTDNet (WSDM 2021, arXiv:2011.07057), SGAT (TKDE 2021, arXiv:1912.00552: L0 gates, 50-80% edges removed on
  assortative graphs, authors' numbers); lottery tickets UGS (ICML 2021, arXiv:2102.06790, transductive), ICPG (inductive, JCST 2024), DGLT (ICLR 2023);
  top-k / SAGPool / ASAP pooling (prune mid-forward); Spexphormer (NeurIPS 2024, arXiv:2411.16278): a narrow scout on the full graph keeps the edges it
  attends to, a wide model runs on the sparse graph. DropEdge (ICLR 2020) is a training regulariser, not pruning.
- Sparse graph transformers: Dwivedi & Bresson GT (arXiv:2012.09699) and PyG `TransformerConv` attend over edges, O(E d); Graphormer (arXiv:2106.05234) is
  dense O(N^2); GraphGPS (arXiv:2205.12454, PyG `GPSConv`), Exphormer (arXiv:2303.06147) O(N+E); NodeFormer, DIFFormer, SGFormer, Polynormer linear O(N)
  but every node contributes; NAGphormer (arXiv:2206.04910) K+1 hop tokens per node, per-target cost independent of graph size.
- Heterogeneous: R-GCN (arXiv:1703.06103, PyG `RGCNConv`), HAN (metapaths), HGT (arXiv:2003.01332, PyG `HGTConv`: type-specific Q/K/V, edges only),
  Simple-HGN (KDD 2021, arXiv:2112.14936: a tuned GAT with edge-type embeddings matches most heterogeneous GNNs -- start simple); prune by relation =
  leave it out of `HeteroConv` or `edge_type_subgraph`.
- Scoring a few targets: shaDow-GNN (NeurIPS 2021, arXiv:2201.07858, PyG/DGL `ShaDowKHopSampler`): a bounded scope per target, any depth -- for us the
  scope can be the symbolic read set; the labeling trick (NeurIPS 2021, arXiv:2010.16103): mark the target inside its subgraph.
- Incremental: an L-layer network's embeddings change only within L hops of a new node or edge (exact local recompute); TGN (arXiv:2006.10637, PyG
  `TGNMemory`); InkStream (arXiv:2309.11071, exact for min/max aggregators, 2.5-427x on CPU, authors' numbers); Ripple (ICDCS 2025, arXiv:2505.12112);
  InstantGNN (KDD 2022); GNNAutoScale (ICML 2021); Gupta et al. hybrid (NeurIPS 2020, arXiv:2006.15212): a heavy GNN once at the root, a cheap MLP per node.
- Rule-based masking: invalid-action masking keeps the policy gradient valid (arXiv:2006.14171); DPMPN (ICLR 2020) learned query-conditioned expansion;
  NSMP (arXiv:2501.14661, preprint, pruning condition not verified); FlexAttention (MLSys 2025, arXiv:2412.05496).
- A CPU baseline: WL-GOOSE (ICAPS 2024, arXiv:2403.16508): WL features + a linear / GP model beat GNN planning heuristics with ~100x fewer parameters.

## Recommendation for us
1. A symbolic scope + a typed edge-list GNN scoring only the open nodes: per call a `HeteroData` from the open nodes and their read sets (or L-hop
   neighbourhoods); each closed subtree collapsed into ONE status node (status + proof word) rather than deleted, so "this sibling route was refuted"
   costs O(1); a Simple-HGN-style GAT or `HGTConv`, 2-3 layers, small width; logits masked to open nodes; a pairwise / listwise ranking loss on logged
   snapshots (RetroGraph, Labassi). No global attention at these sizes.
2. Incremental caching: per-layer embeddings, recompute only the L-hop neighbourhood of what changed (exact); or Gupta's hybrid (heavy context pass when
   the question changes, a cheap per-node MLP between).
3. A WL-features + linear ranker baseline (WL-GOOSE): cheap, deterministic, inspectable -- the bar the GNN must clear.
Defer: dense or linear global transformers, lottery tickets, learned sparsifiers (revisit a Spexphormer scout only if scopes grow large).
Not covered here: conditional computation inside one model (mixtures of experts, early exit) -- asked about the same evening, to be checked separately.
Could not verify: official code for NeuralSparse, HTPS, RetroGraph; code for InkStream and Ripple; PyG layers for Exphormer, NodeFormer, NAGphormer;
FlexAttention block skipping on CPU.
