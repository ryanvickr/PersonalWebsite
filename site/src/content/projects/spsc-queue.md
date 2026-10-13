---
title: "Lock-free SPSC ring buffer"
summary: "Placeholder: a single-producer/single-consumer queue with cache-line-padded indices and cached head/tail, benchmarked against std::mutex and boost::lockfree."
stack: ["C++20", "atomics", "perf", "Google Benchmark"]
metric: { value: "~8 ns", label: "push+pop round trip (target)" }
order: 2
placeholder: true
---

**Placeholder.** Replace this with a real project, or build it. It's a classic
interview topic for trading firms.

Ideas for the write-up: memory ordering choices (`acquire`/`release` and why
`seq_cst` isn't needed), false sharing and `alignas(std::hardware_destructive_interference_size)`,
caching the remote index to avoid cross-core traffic, and latency histograms
(p50/p99/p99.9) rather than averages.
