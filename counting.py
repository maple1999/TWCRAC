class AdaptiveRepetitionCounter:
    def __call__(self, period_scores):
        smoothed = self.smooth(period_scores)
        thresholds = self.estimate_dual_window_thresholds(smoothed)
        peaks = self.detect_valid_peaks(smoothed, thresholds)
        return len(peaks)

    def smooth(self, period_scores):
        raise NotImplementedError("Signal smoothing is not included in this preview.")

    def estimate_dual_window_thresholds(self, smoothed_scores):
        raise NotImplementedError("Dual-window threshold estimation is not included in this preview.")

    def detect_valid_peaks(self, smoothed_scores, thresholds):
        raise NotImplementedError("Constrained peak detection is not included in this preview.")
