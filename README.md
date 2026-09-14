# TWCRAC

Public code for **TWCRAC: Leveraging Temporal Cycle Consistency and Dynamic Thresholding for Robust Repetitive Action Counting**.

## Overview

TWCRAC is a pose-based framework for repetitive action counting that learns from video-level action labels. It combines a temporal Transformer with intra-video Temporal Cycle Consistency (TCC) to learn periodic motion representations. An adaptive dual-window counting mechanism converts the resulting score curves into repetition counts, helping handle changes in motion tempo and intensity.

## Performance

The paper reports the following results on three benchmarks:

| Dataset | MAE ↓ | OBO ↑ |
| --- | ---: | ---: |
| RepCount | 0.083 | 0.790 |
| Countix | 0.116 | 0.854 |
| UCFRep | 0.201 | 0.688 |

TWCRAC outperforms the skeleton-based methods compared in the paper on all three benchmarks while using only video-level action labels. The paper also reports a median end-to-end throughput of **43.83 FPS** and latency of **22.81 ms** on a single **NVIDIA RTX 4060 Ti**, including pose estimation, preprocessing, model inference and counting.

## Code Release

Due to current constraints, only part of the code is publicly available at this stage. The complete code will be released in the future.

## License

This project is released under the [MIT License](LICENSE).
