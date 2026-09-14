# TWCRAC

This repository contains the public code for our paper, **TWCRAC: Leveraging Temporal Cycle Consistency and Dynamic Thresholding for Robust Repetitive Action Counting**.

## Overview

We propose **TWCRAC**, a pose-based framework for repetitive action counting that learns from video-level action labels. We combine a temporal Transformer with intra-video Temporal Cycle Consistency (TCC) to learn periodic motion representations. Our adaptive dual-window counting mechanism converts the resulting score curves into repetition counts, helping handle changes in motion tempo and intensity.

## Performance

We evaluate our method on three benchmarks and achieve the following results:

| Dataset | MAE ↓ | OBO ↑ |
| --- | ---: | ---: |
| RepCount | 0.083 | 0.790 |
| Countix | 0.116 | 0.854 |
| UCFRep | 0.201 | 0.688 |

Our method outperforms the skeleton-based baselines in our comparison on all three benchmarks while using only video-level action labels. On a single **NVIDIA RTX 4060 Ti**, our system achieves a median end-to-end throughput of **43.83 FPS** and latency of **22.81 ms**, including pose estimation, preprocessing, model inference and counting.

## Code Release

Due to current constraints, we are releasing only part of our code at this stage. We will release the complete code in the future.

## License

This project is released under the [MIT License](LICENSE).
