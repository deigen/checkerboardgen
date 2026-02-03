"""
Timing utility for measuring code block execution times.

Usage:
    from cbgen.timer import timer

    timer.start("ClassName.method_name/010")
    # ... code block ...
    timer.stop("ClassName.method_name/010")

    # Get statistics
    timer.print_stats()
    timer.reset()
"""

import time
import threading
from collections import defaultdict
from typing import Dict, List, Optional
import torch


class TimingStats:
    """Statistics for a single timing block."""
    def __init__(self):
        self.count = 0
        self.total_time = 0.0
        self.min_time = float('inf')
        self.max_time = 0.0
        self.times: List[float] = []

    def add(self, elapsed: float):
        self.count += 1
        self.total_time += elapsed
        self.min_time = min(self.min_time, elapsed)
        self.max_time = max(self.max_time, elapsed)
        self.times.append(elapsed)

    @property
    def mean_time(self) -> float:
        return self.total_time / self.count if self.count > 0 else 0.0

    @property
    def std_time(self) -> float:
        if self.count < 2:
            return 0.0
        mean = self.mean_time
        variance = sum((t - mean) ** 2 for t in self.times) / self.count
        return variance ** 0.5


class Timer:
    """Global timer for tracking code block execution times."""

    def __init__(self):
        self.stats: Dict[str, TimingStats] = defaultdict(TimingStats)
        self.active_timers: Dict[str, float] = {}
        self.lock = threading.Lock()
        self.enabled = True

    def start(self, block_name: str):
        """Start timing a code block."""
        if not self.enabled:
            return
        with self.lock:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self.active_timers[block_name] = time.perf_counter()

    def stop(self, block_name: str):
        """Stop timing a code block."""
        if not self.enabled:
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = time.perf_counter()

        with self.lock:
            if block_name not in self.active_timers:
                print(f"Warning: timer.stop('{block_name}') called without matching start()")
                return
            start_time = self.active_timers.pop(block_name)
            elapsed = end_time - start_time
            self.stats[block_name].add(elapsed)

    def reset(self):
        """Reset all timing statistics."""
        with self.lock:
            self.stats.clear()
            self.active_timers.clear()

    def enable(self):
        """Enable timing."""
        self.enabled = True

    def disable(self):
        """Disable timing."""
        self.enabled = False

    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """Get timing statistics as a dictionary."""
        with self.lock:
            result = {}
            for block_name, stats in self.stats.items():
                result[block_name] = {
                    'count': stats.count,
                    'total': stats.total_time,
                    'mean': stats.mean_time,
                    'std': stats.std_time,
                    'min': stats.min_time,
                    'max': stats.max_time,
                }
            return result

    def print_stats(self, sort_by: str = 'total', top_n: Optional[int] = None):
        """
        Print timing statistics grouped by function.

        Args:
            sort_by: Sort by 'total', 'mean', 'count', 'max', or 'min'
            top_n: Only show top N function groups (None = show all)
        """
        stats = self.get_stats()
        if not stats:
            print("No timing data collected.")
            return

        # group by function name (part before last '/')
        function_groups = defaultdict(dict)
        for block_name, block_stats in stats.items():
            function_name = block_name.rsplit('/', 1)[0] if '/' in block_name else block_name
            function_groups[function_name][block_name] = block_stats

        sorted_functions = sorted(
            ((fn, max(bs.get(sort_by, 0) for bs in blocks.values())) for fn, blocks in function_groups.items()),
            key=lambda x: x[1],
            reverse=True
        )

        if top_n is not None:
            sorted_functions = sorted_functions[:top_n]

        # Print header
        print()
        print(f"{'Block Name':<50} {'Count':>8} {'Total(s)':>10} {'Mean(ms)':>10} {'Std(ms)':>10} {'Min(ms)':>10} {'Max(ms)':>10}")
        print("=" * 130)

        # Print each function group
        for i, (function_name, _) in enumerate(sorted_functions):
            blocks = function_groups[function_name]

            # Sort blocks within this function by the sort key
            sorted_blocks = sorted(
                blocks.items(),
                key=lambda x: x[1].get(sort_by, 0),
                reverse=True
            )
            if top_n is not None:
                sorted_blocks = sorted_blocks[:top_n]

            # Print all blocks in this function
            for block_name, block_stats in sorted_blocks:
                print(
                    f"{block_name:<50} "
                    f"{block_stats['count']:>8} "
                    f"{block_stats['total']:>10.4f} "
                    f"{block_stats['mean']*1000:>10.2f} "
                    f"{block_stats['std']*1000:>10.2f} "
                    f"{block_stats['min']*1000:>10.2f} "
                    f"{block_stats['max']*1000:>10.2f}"
                )

            # Add blank line between function groups (except after the last one)
            if i < len(sorted_functions) - 1:
                print()

        print("=" * 130)
        total_time = sum(s['total'] for s in stats.values())
        print(f"Total time across all blocks: {total_time:.4f}s\n")

    def save_stats(self, filepath: str):
        """Save timing statistics to a JSON file."""
        import json
        with open(filepath, 'w') as f:
            json.dump(self.get_stats(), f, indent=2)


# Global timer instance
timer = Timer()
