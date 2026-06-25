---
name: repo-understanding
description: Map the architecture and execution flow of the Effort/AIGI ML repository. Use when tracing training or evaluation code, locating entrypoints and configuration, understanding data-to-metric flow, or assessing hidden dependencies and duplicated logic.
---

# Repository understanding

1. Identify training and evaluation entrypoints, configuration loading, data
   pipelines, model definitions, losses, and metrics.
2. Trace the concrete flow from input data through preprocessing, model
   execution, loss calculation, and reported metrics.
3. Record key files and dependency boundaries.
4. Flag dead code, duplicated logic, hidden coupling, and unresolved questions.

Return an architecture map, key-file list, and concise risk/unknown summary.
