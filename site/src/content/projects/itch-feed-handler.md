---
title: "Market-data feed handler"
summary: "Placeholder: a zero-allocation parser for NASDAQ TotalView-ITCH 5.0 that rebuilds a full-depth order book from a day of historical messages."
stack: ["C++20", "binary parsing", "order books", "mmap"]
metric: { value: "—", label: "messages/sec (to measure)" }
order: 3
placeholder: true
---

**Placeholder.** Replace this with a real project.

Ideas: memory-map the raw file, decode big-endian fields without copies,
dispatch on message type with a jump table, keep price levels in a flat array
indexed by tick, and report throughput and per-message latency.
