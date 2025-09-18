from utils.metric_bundles import MetricMonitorBundle
from utils.metrics_monitor import MetricAlert


class PPOAlerts(MetricMonitorBundle):
    """PPO-specific metric alert bundle.

    Encapsulates alert monitors for KL, clip fraction, and explained variance.
    """

    def __init__(self, agent) -> None:
        self.agent = agent

    def _monitor_approx_kl_oob(self, history: dict):
        metric_key = "train/approx_kl"
        metric_values = history.get(metric_key)
        if not metric_values:
            return None
        _, last_value = metric_values[-1]

        # If the KL divergence is too low, it means that the updates are too weak
        min_threshold, max_threshold = 1e-3, 5e-2
        if last_value < min_threshold:
            return MetricAlert(
                metric=metric_key, 
                message=f"< {min_threshold} is very low; updates may be too weak", 
                tip="Increase the learning rate or decrease the clip range"
            )
        
        # If the KL divergence is too high, it means that the updates are too aggressive
        if last_value > max_threshold:
            return MetricAlert(
                metric=metric_key, 
                message=f"> {max_threshold} is high; updates may be too aggressive", 
                tip="Decrease the learning rate or increase the clip range"
            )

    def _monitor_clip_fraction_oob(self, history: dict):
        metric_key = "train/clip_fraction"
        metric_values = history.get(metric_key)
        if not metric_values:
            return None
        _, last_value = metric_values[-1]

        # If the clip fraction is too low, it means that the updates are too weak
        min_threshold, max_threshold = 0.05, 0.5
        if last_value < min_threshold:
            return MetricAlert(
                metric=metric_key, 
                message=f"< {min_threshold} is very low; likely under-updating", 
                tip="Increase the learning rate or decrease the clip range"
            )
        
        # If the clip fraction is too high, it means that the updates are too aggressive
        if last_value > max_threshold:
            return MetricAlert(
                metric=metric_key, 
                message=f"> {max_threshold} is very high; many updates are clipped", 
                tip="Decrease the learning rate or increase the clip range"
            )

