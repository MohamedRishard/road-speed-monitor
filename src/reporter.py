# src/reporter.py
"""
Reporting Module

Generates:
1. CSV log of all vehicle speeds
2. CSV log of overspeed violations
3. Summary statistics
4. Matplotlib charts (speed distribution, violation timeline)
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import SPEED_LIMIT_KMH, REPORTS_OUT


class Reporter:
    """
    Collects speed and violation data throughout the pipeline
    and generates reports at the end.
    """

    def __init__(self, video_filename="unknown"):
        self.speed_limit    = SPEED_LIMIT_KMH
        self.video_filename = video_filename
        self.speed_records  = []      # all vehicle speeds
        self.violations     = []      # overspeed violations only
        self.session_time   = datetime.now().strftime('%Y%m%d_%H%M%S')

        os.makedirs(REPORTS_OUT, exist_ok=True)
        print(f"[Reporter] Initialized. Reports → {REPORTS_OUT}")

    def add_speed_record(self, track_id, class_name, speed_kmh, frame_number):
        """
        Record a confirmed speed measurement.
        Call this once per vehicle when speed is first confirmed.
        """
        self.speed_records.append({
            'track_id':    track_id,
            'class_name':  class_name,
            'speed_kmh':   round(speed_kmh, 1),
            'frame':       frame_number,
            'timestamp':   datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'status':      'OVERSPEED' if speed_kmh > self.speed_limit else 'NORMAL'
        })

    def add_violation(self, violation_dict):
        """Record an overspeed violation."""
        self.violations.append(violation_dict)

    def generate_all(self):
        """
        Generate all reports. Call this at end of pipeline.
        Returns dict of file paths created.
        """
        print(f"\n[Reporter] Generating reports...")
        paths = {}

        if not self.speed_records:
            print("[Reporter] No speed data to report.")
            return paths

        paths['speeds_csv']     = self._save_speeds_csv()
        paths['violations_csv'] = self._save_violations_csv()
        paths['summary_txt']    = self._save_summary()
        paths['chart']          = self._save_chart()

        print(f"[Reporter] All reports saved to: {REPORTS_OUT}")
        return paths

    def _save_speeds_csv(self):
        """Save all vehicle speed records to CSV."""
        df   = pd.DataFrame(self.speed_records)
        path = os.path.join(REPORTS_OUT,
                            f"speeds_{self.session_time}.csv")
        df.to_csv(path, index=False)
        print(f"  [CSV] Speeds log     : {path}")
        return path

    def _save_violations_csv(self):
        """Save overspeed violations to CSV."""
        path = os.path.join(REPORTS_OUT,
                            f"violations_{self.session_time}.csv")
        if self.violations:
            # Remove bbox from CSV (not useful in tabular format)
            clean = [{k: v for k, v in v.items() if k != 'bbox'}
                     for v in self.violations]
            df = pd.DataFrame(clean)
        else:
            df = pd.DataFrame(columns=[
                'track_id', 'class_name', 'speed_kmh',
                'speed_limit', 'excess_kmh', 'frame', 'timestamp'
            ])

        df.to_csv(path, index=False)
        print(f"  [CSV] Violations log : {path}")
        return path

    def _save_summary(self):
        """Save plain-text summary statistics."""
        speeds = [r['speed_kmh'] for r in self.speed_records]
        path   = os.path.join(REPORTS_OUT,
                              f"summary_{self.session_time}.txt")

        lines = [
            "=" * 50,
            "  ROAD SPEED MONITORING — SESSION REPORT",
            "=" * 50,
            f"  Video file       : {self.video_filename}",
            f"  Session time     : {self.session_time}",
            f"  Speed limit      : {self.speed_limit} km/h",
            "-" * 50,
            f"  Total vehicles   : {len(speeds)}",
            f"  Average speed    : {np.mean(speeds):.1f} km/h",
            f"  Maximum speed    : {np.max(speeds):.1f} km/h",
            f"  Minimum speed    : {np.min(speeds):.1f} km/h",
            f"  Std deviation    : {np.std(speeds):.1f} km/h",
            "-" * 50,
            f"  Violations       : {len(self.violations)}",
            f"  Violation rate   : "
            f"{len(self.violations)/len(speeds)*100:.1f}%",
            "-" * 50,
        ]

        if self.violations:
            lines.append("  Violating vehicles:")
            for v in self.violations:
                lines.append(
                    f"    ID:{v['track_id']:>3} | "
                    f"{v['class_name']:<12} | "
                    f"{v['speed_kmh']:>6.1f} km/h | "
                    f"+{v['excess_kmh']:.1f} over limit | "
                    f"Frame {v['frame']}"
                )

        lines.append("=" * 50)

        with open(path, 'w') as f:
            f.write('\n'.join(lines))

        # Also print to terminal
        print()
        for line in lines:
            print(line)

        print(f"\n  [TXT] Summary saved  : {path}")
        return path

    def _save_chart(self):
        """Generate and save a speed distribution chart."""
        speeds     = [r['speed_kmh'] for r in self.speed_records]
        track_ids  = [r['track_id']  for r in self.speed_records]
        statuses   = [r['status']    for r in self.speed_records]
        colors     = ['red' if s == 'OVERSPEED' else 'green'
                      for s in statuses]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Road Speed Monitoring Report',
                     fontsize=14, fontweight='bold')

        # ── Chart 1: Speed per Vehicle (bar chart) ──────────────────
        ax1 = axes[0]
        bars = ax1.bar(
            [f"ID:{tid}" for tid in track_ids],
            speeds,
            color=colors,
            edgecolor='black',
            linewidth=0.5
        )

        # Speed limit line
        ax1.axhline(y=self.speed_limit, color='red',
                    linestyle='--', linewidth=2,
                    label=f'Speed Limit ({self.speed_limit} km/h)')

        # Value labels on bars
        for bar, spd in zip(bars, speeds):
            ax1.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.5,
                     f'{spd:.1f}', ha='center', va='bottom',
                     fontsize=8)

        ax1.set_xlabel('Vehicle ID')
        ax1.set_ylabel('Speed (km/h)')
        ax1.set_title('Speed per Vehicle')
        ax1.tick_params(axis='x', rotation=45)

        normal_patch    = mpatches.Patch(color='green', label='Normal')
        overspeed_patch = mpatches.Patch(color='red',   label='Overspeed')
        ax1.legend(handles=[normal_patch, overspeed_patch,
                             plt.Line2D([0], [0], color='red',
                                        linestyle='--',
                                        label=f'Limit {self.speed_limit}km/h')])

        # ── Chart 2: Speed Distribution (histogram) ─────────────────
        ax2 = axes[1]
        ax2.hist(speeds, bins=8, color='steelblue',
                 edgecolor='black', alpha=0.7)
        ax2.axvline(x=self.speed_limit, color='red',
                    linestyle='--', linewidth=2,
                    label=f'Speed Limit ({self.speed_limit} km/h)')
        ax2.axvline(x=np.mean(speeds), color='orange',
                    linestyle='-', linewidth=2,
                    label=f'Average ({np.mean(speeds):.1f} km/h)')

        ax2.set_xlabel('Speed (km/h)')
        ax2.set_ylabel('Number of Vehicles')
        ax2.set_title('Speed Distribution')
        ax2.legend()

        plt.tight_layout()

        path = os.path.join(REPORTS_OUT,
                            f"chart_{self.session_time}.png")
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  [PNG] Chart saved    : {path}")
        return path