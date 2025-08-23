"""
Process Mining Pipeline for SAM Dataset

Applies process mining techniques for sequence-of-actions insights using PM4Py.
Converts SAM data to event logs, discovers process models, and analyzes performance.

References:
- PM4Py documentation: https://pm4py-source.readthedocs.io/
- PM4Py GitHub: https://github.com/process-intelligence-solutions/pm4py
"""

import logging
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from logging_config import setup_logging

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# PM4Py imports
try:
    import pm4py
    from pm4py.objects.conversion.log import converter as log_converter
    from pm4py.objects.log.exporter.xes import exporter as xes_exporter
    from pm4py.algo.discovery.inductive import algorithm as inductive_miner
    from pm4py.algo.discovery.dfg import algorithm as dfg_discovery
    from pm4py.visualization.dfg import visualizer as dfg_visualization
    from pm4py.visualization.petri_net import visualizer as pn_vis
    from pm4py.algo.conformance.alignments import algorithm as alignments
    from pm4py.statistics.traces.generic.log import case_statistics
    from pm4py.util import constants as pm4py_constants
    HAS_PM4PY = True
except ImportError:
    HAS_PM4PY = False
    pm4py = None

# Configure logging
logger = logging.getLogger(__name__)


class SAMProcessMiner:
    """Process mining pipeline for the SAM (Student Action Mining) dataset."""

    def __init__(
        self,
        output_dir: Path,
        figures_dir: Path = Path("figures"),
        tables_dir: Path = Path("tables"),
    ):
        """Initialize process miner.

        Args:
            output_dir: Directory to save general outputs
            figures_dir: Directory to export visual artifacts
            tables_dir: Directory to export tabular reports
        """

        if not HAS_PM4PY:
            raise ImportError(
                "PM4Py is required for process mining. Install with: pip install pm4py"
            )

        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.figures_dir = figures_dir
        self.figures_dir.mkdir(parents=True, exist_ok=True)

        self.tables_dir = tables_dir
        self.tables_dir.mkdir(parents=True, exist_ok=True)

        self.event_log = None
        self.dfg = None
        self.petri_net = None
        self.initial_marking = None
        self.final_marking = None

        logger.info(
            "Initialized SAM process miner with output directory: %s", output_dir
        )

    def _parse_schema(self, schema_path: Path) -> Dict[str, List[str]]:
        """Parse schema markdown file to retrieve required and optional fields."""

        schema_text = schema_path.read_text(encoding="utf-8")
        required: List[str] = []
        optional: List[str] = []
        current: Optional[List[str]] = None

        for line in schema_text.splitlines():
            line = line.strip()
            if re.match(r"##+\s+Required Fields", line, re.IGNORECASE):
                current = required
            elif re.match(r"##+\s+Optional Fields", line, re.IGNORECASE):
                current = optional
            elif line.startswith("###") and current is not None:
                field = line.strip("# ")
                current.append(field)

        return {"required": required, "optional": optional}

    def load_and_validate_sam_data(self, sam_csv_path: Path, schema_path: Path) -> pd.DataFrame:
        """Load SAM CSV and validate schema according to spec."""

        logger.info("Loading SAM data from %s", sam_csv_path)

        if not sam_csv_path.exists():
            raise FileNotFoundError(f"SAM file not found: {sam_csv_path}")

        df = pd.read_csv(sam_csv_path)
        logger.info("Loaded %s records from SAM dataset", len(df))

        schema = self._parse_schema(schema_path)
        required_columns = schema.get("required", ["case_id", "activity", "timestamp"])
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            logger.warning("Missing required columns: %s", missing_columns)
            df = self._infer_column_mapping(df)

        df = self._validate_timestamps(df)

        if "resource" not in df.columns and "resource" in schema.get("optional", []):
            df["resource"] = "system"

        initial_size = len(df)
        df = df.dropna(subset=["case_id", "activity"])
        if len(df) < initial_size:
            logger.warning(
                "Dropped %s records with missing case_id or activity",
                initial_size - len(df),
            )

        logger.info(
            "Validated SAM data: %s records, %s cases",
            len(df),
            df["case_id"].nunique(),
        )
        return df
    
    def _infer_column_mapping(self, df: pd.DataFrame) -> pd.DataFrame:
        """Attempt to infer column mappings for SAM schema.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with mapped columns
        """
        column_mappings = {
            # Case ID variations
            'case_id': ['student_id', 'session_id', 'user_id', 'id', 'case'],
            # Activity variations  
            'activity': ['action', 'event', 'activity_name', 'task'],
            # Timestamp variations
            'timestamp': ['time', 'datetime', 'date', 'created_at']
        }
        
        for target_col, candidates in column_mappings.items():
            if target_col not in df.columns:
                for candidate in candidates:
                    if candidate in df.columns:
                        df[target_col] = df[candidate]
                        logger.info(f"Mapped {candidate} -> {target_col}")
                        break
        
        return df
    
    def _validate_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and convert timestamps to proper format.
        
        Args:
            df: DataFrame with timestamp column
            
        Returns:
            DataFrame with validated timestamps
        """
        if 'timestamp' not in df.columns:
            logger.error("No timestamp column found")
            return df
        
        # Try to parse timestamps
        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        except Exception as e:
            logger.warning(f"Failed to parse timestamps automatically: {e}")
            # Try common formats
            for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d', '%d/%m/%Y %H:%M:%S']:
                try:
                    df['timestamp'] = pd.to_datetime(df['timestamp'], format=fmt)
                    logger.info(f"Successfully parsed timestamps with format: {fmt}")
                    break
                except:
                    continue
            else:
                logger.error("Could not parse timestamps")
                # Create dummy timestamps if all else fails
                df['timestamp'] = pd.date_range(start='2023-01-01', periods=len(df), freq='H')
        
        # Sort by case_id and timestamp
        df = df.sort_values(['case_id', 'timestamp'])
        
        return df
    
    def convert_to_event_log(self, sam_df: pd.DataFrame) -> Any:
        """Convert SAM DataFrame to PM4Py event log format.
        
        Args:
            sam_df: Validated SAM DataFrame
            
        Returns:
            PM4Py event log object
        """
        logger.info("Converting SAM data to PM4Py event log format...")
        
        # Rename columns to PM4Py standard names
        log_df = sam_df.copy()
        log_df = log_df.rename(columns={
            'case_id': 'case:concept:name',
            'activity': 'concept:name', 
            'timestamp': 'time:timestamp'
        })
        
        # Add optional attributes
        if 'resource' in sam_df.columns:
            log_df['org:resource'] = sam_df['resource']
        
        # Convert to event log
        parameters = {log_converter.Variants.TO_EVENT_LOG.value.Parameters.CASE_ID_KEY: 'case:concept:name'}
        self.event_log = log_converter.apply(log_df, parameters=parameters, variant=log_converter.Variants.TO_EVENT_LOG)
        
        logger.info(f"Created event log with {len(self.event_log)} traces")
        return self.event_log
    
    def export_xes(self, output_path: Optional[Path] = None) -> Path:
        """Export event log to XES format.
        
        Args:
            output_path: Path for XES file
            
        Returns:
            Path to exported XES file
        """
        if self.event_log is None:
            raise ValueError("No event log available. Call convert_to_event_log() first.")
        
        if output_path is None:
            output_path = self.output_dir / 'sam_event_log.xes'
        
        logger.info(f"Exporting event log to XES: {output_path}")
        xes_exporter.apply(self.event_log, str(output_path))
        
        return output_path
    
    def discover_dfg(self) -> Tuple[Dict, Dict]:
        """Discover Directly-Follows Graph from event log.
        
        Returns:
            Tuple of (dfg, start_activities, end_activities)
        """
        if self.event_log is None:
            raise ValueError("No event log available")
        
        logger.info("Discovering Directly-Follows Graph...")
        
        # Discover DFG
        dfg = dfg_discovery.apply(self.event_log)
        start_activities = pm4py.get_start_activities(self.event_log)
        end_activities = pm4py.get_end_activities(self.event_log)
        
        self.dfg = dfg
        
        logger.info(f"DFG discovered with {len(dfg)} edges")
        return dfg, start_activities, end_activities
    
    def discover_petri_net(self) -> Tuple[Any, Any, Any]:
        """Discover Petri net using Inductive Miner.
        
        Returns:
            Tuple of (petri_net, initial_marking, final_marking)
        """
        if self.event_log is None:
            raise ValueError("No event log available")
        
        logger.info("Discovering Petri net using Inductive Miner...")
        
        net, initial_marking, final_marking = inductive_miner.apply(self.event_log)
        
        self.petri_net = net
        self.initial_marking = initial_marking
        self.final_marking = final_marking
        
        logger.info(f"Petri net discovered with {len(net.places)} places and {len(net.transitions)} transitions")
        return net, initial_marking, final_marking
    
    def analyze_performance(self) -> Dict[str, Any]:
        """Analyze performance metrics of the process.
        
        Returns:
            Dictionary with performance analysis results
        """
        if self.event_log is None:
            raise ValueError("No event log available")
        
        logger.info("Analyzing process performance...")
        
        results = {}
        
        # Case duration statistics
        case_durations = case_statistics.get_case_arrival_avg_time(self.event_log)
        
        # Activity frequency
        activity_counts = {}
        for trace in self.event_log:
            for event in trace:
                activity = event['concept:name']
                activity_counts[activity] = activity_counts.get(activity, 0) + 1
        
        results['activity_frequency'] = activity_counts
        
        # Trace variants analysis
        variants = pm4py.get_variants(self.event_log)
        variant_stats = []
        for variant, traces in variants.items():
            variant_stats.append({
                'variant': ' -> '.join(variant),
                'frequency': len(traces),
                'percentage': len(traces) / len(self.event_log) * 100
            })
        
        variant_stats = sorted(variant_stats, key=lambda x: x['frequency'], reverse=True)
        results['top_variants'] = variant_stats[:10]  # Top 10 variants
        
        # Throughput time analysis
        throughput_times = []
        for trace in self.event_log:
            if len(trace) > 1:
                start_time = trace[0]['time:timestamp']
                end_time = trace[-1]['time:timestamp']
                duration = (end_time - start_time).total_seconds() / 3600  # Convert to hours
                throughput_times.append(duration)
        
        if throughput_times:
            results['throughput_stats'] = {
                'mean_hours': np.mean(throughput_times),
                'median_hours': np.median(throughput_times),
                'std_hours': np.std(throughput_times),
                'min_hours': np.min(throughput_times),
                'max_hours': np.max(throughput_times)
            }
        
        return results
    
    def analyze_conformance(self) -> Dict[str, Any]:
        """Analyze conformance between event log and discovered model.
        
        Returns:
            Dictionary with conformance analysis results
        """
        if self.event_log is None or self.petri_net is None:
            logger.warning("Event log or Petri net not available for conformance analysis")
            return {}
        
        logger.info("Analyzing conformance...")
        
        try:
            # Compute alignments
            aligned_traces = alignments.apply_log(
                self.event_log, 
                self.petri_net, 
                self.initial_marking, 
                self.final_marking
            )
            
            # Calculate fitness scores
            fitness_scores = []
            for alignment in aligned_traces:
                if 'fitness' in alignment:
                    fitness_scores.append(alignment['fitness'])
            
            results = {
                'average_fitness': np.mean(fitness_scores) if fitness_scores else 0.0,
                'fitness_scores': fitness_scores,
                'conforming_traces': sum(1 for score in fitness_scores if score == 1.0),
                'total_traces': len(fitness_scores)
            }
            
            if results['total_traces'] > 0:
                results['conformance_rate'] = results['conforming_traces'] / results['total_traces']
            
            return results
            
        except Exception as e:
            logger.error(f"Conformance analysis failed: {e}")
            return {'error': str(e)}
    
    def identify_bottlenecks(self) -> List[Dict[str, Any]]:
        """Identify process bottlenecks based on waiting times.
        
        Returns:
            List of bottleneck information
        """
        if self.dfg is None:
            logger.warning("DFG not available for bottleneck analysis")
            return []
        
        logger.info("Identifying process bottlenecks...")
        
        # Calculate waiting times between activities
        waiting_times = {}
        
        for trace in self.event_log:
            for i in range(len(trace) - 1):
                current_activity = trace[i]['concept:name']
                next_activity = trace[i + 1]['concept:name']
                
                current_time = trace[i]['time:timestamp']
                next_time = trace[i + 1]['time:timestamp']
                
                waiting_time = (next_time - current_time).total_seconds() / 60  # Minutes
                
                edge = (current_activity, next_activity)
                if edge not in waiting_times:
                    waiting_times[edge] = []
                waiting_times[edge].append(waiting_time)
        
        # Calculate statistics for each edge
        bottlenecks = []
        for edge, times in waiting_times.items():
            if len(times) > 1:  # Need multiple observations
                bottleneck_info = {
                    'from_activity': edge[0],
                    'to_activity': edge[1],
                    'median_waiting_time_minutes': np.median(times),
                    'mean_waiting_time_minutes': np.mean(times),
                    'max_waiting_time_minutes': np.max(times),
                    'frequency': len(times)
                }
                bottlenecks.append(bottleneck_info)
        
        # Sort by median waiting time
        bottlenecks = sorted(bottlenecks, key=lambda x: x['median_waiting_time_minutes'], reverse=True)
        
        return bottlenecks[:10]  # Top 10 bottlenecks
    
    def save_visualizations(self) -> None:
        """Save process visualizations to figures directory."""
        logger.info("Saving process visualizations...")

        if self.dfg is not None:
            try:
                start_activities = pm4py.get_start_activities(self.event_log)
                end_activities = pm4py.get_end_activities(self.event_log)
                gviz = dfg_visualization.apply(
                    self.dfg,
                    start_activities=start_activities,
                    end_activities=end_activities,
                )
                dfg_visualization.save(
                    gviz, str(self.figures_dir / "sam_dfg.png")
                )
                logger.info("DFG visualization saved")
            except Exception as e:
                logger.error("Failed to save DFG visualization: %s", e)

        if self.petri_net is not None:
            try:
                gviz = pn_vis.apply(
                    self.petri_net, self.initial_marking, self.final_marking
                )
                pn_vis.save(gviz, str(self.figures_dir / "sam_petri_net.png"))
                logger.info("Petri net visualization saved")
            except Exception as e:
                logger.error("Failed to save Petri net visualization: %s", e)
    
    def save_reports(
        self,
        performance_results: Dict,
        conformance_results: Dict,
        bottlenecks: List[Dict],
        variants: List[Dict],
    ) -> None:
        """Save analysis reports to CSV files."""

        logger.info("Saving analysis reports...")

        if "activity_frequency" in performance_results:
            activity_df = pd.DataFrame(
                [
                    {"activity": act, "frequency": freq}
                    for act, freq in performance_results["activity_frequency"].items()
                ]
            )
            activity_df.to_csv(
                self.output_dir / "activity_frequency.csv", index=False
            )

        if variants:
            variant_df = pd.DataFrame(variants)
            variant_df.to_csv(self.output_dir / "process_variants.csv", index=False)

        if bottlenecks:
            bottleneck_df = pd.DataFrame(bottlenecks)
            bottleneck_df.to_csv(
                self.output_dir / "bottleneck_analysis.csv", index=False
            )

        if conformance_results:
            conformance_df = pd.DataFrame([conformance_results])
            conformance_df.to_csv(
                self.tables_dir / "sam_conformance.csv", index=False
            )
    
    def run_full_pipeline(
        self, sam_csv_path: Path, schema_path: Path
    ) -> Dict[str, Any]:
        """Run complete process mining pipeline."""

        logger.info("Starting complete process mining pipeline...")

        sam_df = self.load_and_validate_sam_data(sam_csv_path, schema_path)

        self.convert_to_event_log(sam_df)

        xes_path = self.export_xes()

        dfg, start_activities, end_activities = self.discover_dfg()
        net, initial_marking, final_marking = self.discover_petri_net()

        performance_results = self.analyze_performance()
        conformance_results = self.analyze_conformance()
        bottlenecks = self.identify_bottlenecks()

        self.save_visualizations()

        variants = performance_results.get("top_variants", [])
        self.save_reports(
            performance_results, conformance_results, bottlenecks, variants
        )
        
        # Compile final results
        results = {
            'data_summary': {
                'total_events': len(sam_df),
                'total_cases': sam_df['case_id'].nunique(),
                'unique_activities': sam_df['activity'].nunique(),
                'xes_export_path': str(xes_path)
            },
            'process_discovery': {
                'dfg_edges': len(dfg),
                'petri_net_places': len(net.places),
                'petri_net_transitions': len(net.transitions)
            },
            'performance': performance_results,
            'conformance': conformance_results,
            'bottlenecks': bottlenecks[:5],  # Top 5 bottlenecks
            'output_directory': str(self.output_dir)
        }
        
        logger.info("Process mining pipeline completed successfully!")
        return results


def main():
    """CLI interface for SAM process mining."""
    import argparse
    import json
    
    parser = argparse.ArgumentParser(
        description="SAM dataset process mining pipeline"
    )
    parser.add_argument(
        "--sam-csv",
        type=Path,
        default=Path("data/sam/events.csv"),
        help="Path to SAM CSV file",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("reports/process_mining"),
        help="Output directory for results",
    )
    parser.add_argument(
        "--schema",
        type=Path,
        default=Path("data/sam/schema.md"),
        help="Path to schema markdown file",
    )

    args = parser.parse_args()

    try:
        miner = SAMProcessMiner(args.out_dir)
        results = miner.run_full_pipeline(args.sam_csv, args.schema)

        with open(args.out_dir / "pipeline_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)

        print("\nProcess Mining Results Summary:")
        print("=" * 50)
        print(f"Total Events: {results['data_summary']['total_events']}")
        print(f"Total Cases: {results['data_summary']['total_cases']}")
        print(f"Unique Activities: {results['data_summary']['unique_activities']}")
        print(f"DFG Edges: {results['process_discovery']['dfg_edges']}")
        print(f"Petri Net Places: {results['process_discovery']['petri_net_places']}")
        print(f"Petri Net Transitions: {results['process_discovery']['petri_net_transitions']}")

        if "average_fitness" in results["conformance"]:
            print(
                f"Average Fitness: {results['conformance']['average_fitness']:.3f}"
            )

        print(f"\nResults saved to: {results['output_directory']}")

    except Exception as e:
        logger.error("Process mining pipeline failed: %s", e)
        raise


if __name__ == '__main__':
    setup_logging()
    main()
