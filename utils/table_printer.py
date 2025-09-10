"""Namespace-aware metrics table printer utilities (shim).

The implementation now lives under `loggers.print_metrics_logger` to co-locate
printing logic with the Lightning logger. This module re-exports the public API
for backward compatibility.
"""

from loggers.print_metrics_logger import (
    NamespaceTablePrinter,
    print_namespaced_dict,
)

__all__ = ["NamespaceTablePrinter", "print_namespaced_dict"]

