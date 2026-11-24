"""ChainSleuth: illicit Bitcoin transaction detection on the Elliptic graph.

macOS OpenMP note: XGBoost/LightGBM link Homebrew's libomp, while torch ships its own copy.
With both loaded in one process, importing torch first makes XGBoost segfault, and importing
the tree libraries first makes multithreaded torch ops deadlock. So:
  * chainsleuth.tabular refuses to load after torch (a clear error instead of a segfault);
  * chainsleuth.gnn drops torch to one thread if a tree library is already loaded;
  * the pipeline runs every stage in its own process, so GNN training keeps all cores.
"""
