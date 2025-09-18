from utils.metric_bundles import MetricMonitorBundle
from utils.metrics_monitor import MetricAlert


class PPOAlerts(MetricMonitorBundle):
    """PPO-specific metric alert bundle.

    Encapsulates alert monitors for KL, clip fraction, and explained variance.
    """

    def __init__(self, agent) -> None:
        self.agent = agent

    def _monitor_approx_kl_oob(self, history: dict):
        alert_msg = None
        metric_key = "train/approx_kl"
        metric_values = history.get(metric_key)
        if not metric_values:
            return None
        _, last_value = metric_values[-1]

        min_threshold, max_threshold = 1e-3, 5e-2
        tip = None
        if last_value < min_threshold:
            alert_msg = f"< {min_threshold} is very low; updates may be too weak"
            tip = "Increase the learning rate or decrease the clip range"
        if last_value > max_threshold:
            alert_msg = f"> {max_threshold} is high; updates may be too aggressive"
            tip = "Decrease the learning rate or increase the clip range"

        if not alert_msg:
            return None
        return MetricAlert(metric=metric_key, message=alert_msg, tip=tip)

    def _monitor_clip_fraction_oob(self, history: dict):
        alert_msg = None
        metric_key = "train/clip_fraction"
        metric_values = history.get(metric_key)
        if not metric_values:
            return None
        _, last_value = metric_values[-1]

        min_threshold, max_threshold = 0.05, 0.5
        tip = None
        if last_value < min_threshold:
            alert_msg = f"< {min_threshold} is very low; likely under-updating"
            tip = "Increase the learning rate or decrease the clip range"
        if last_value > max_threshold:
            alert_msg = f"> {max_threshold} is very high; many updates are clipped"
            tip = "Decrease the learning rate or increase the clip range"

        if not alert_msg:
            return None
        return MetricAlert(metric=metric_key, message=alert_msg, tip=tip)

