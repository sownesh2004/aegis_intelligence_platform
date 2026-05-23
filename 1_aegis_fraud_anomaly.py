"""
AEGIS: Real-Time Financial Fraud Anomaly & Streaming Intelligence Engine
========================================================================
A state-of-the-art Python-based streaming analytics platform for financial
risk monitoring. This application simulates a live transaction feed (Credit Card,
Wire, ACH, Crypto), processes it through a real-time ETL pipeline, runs unsupervised
anomaly detection, and performs Explainable AI (XAI) feature attribution.

Built with high visual polish using the `rich` terminal interface framework.
"""

import sys
import time
import random
import math
from datetime import datetime
from typing import Dict, List, Tuple, Any

# Gracefully check and guide library installations
try:
    import numpy as np
except ImportError:
    print("\n[!] 'numpy' is required for the Aegis Analytics Engine.")
    print("    Please run: pip install numpy rich")
    sys.exit(1)

try:
    from rich.console import Console
    from rich.layout import Layout
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich.align import Align
    from rich.progress import BarColumn, Progress, TextColumn
except ImportError:
    print("\n[!] 'rich' is required for the high-fidelity terminal UI.")
    print("    Please run: pip install numpy rich")
    sys.exit(1)


# ==============================================================================
# 1. STREAM SIMULATOR
# ==============================================================================
class DataStreamSimulator:
    """Generates continuous streams of financial transactions with mixed channels and features."""
    
    CHANNELS = ["Credit Card", "Wire Transfer", "Crypto Wallet", "ACH Payment"]
    COUNTRIES = ["US", "GB", "DE", "SG", "CA", "KY", "NL", "IN"]
    
    def __init__(self, anomaly_rate: float = 0.08):
        self.anomaly_rate = anomaly_rate
        self.transaction_counter = 0
        
        # Baseline user patterns (Normal Behavior Profiles)
        self.baseline = {
            "Credit Card":  {"mean_amount": 45.0,   "std_amount": 30.0,  "mean_velocity": 1.2,  "mean_dist": 8.0},
            "Wire Transfer": {"mean_amount": 8500.0, "std_amount": 3500.0,"mean_velocity": 0.1,  "mean_dist": 150.0},
            "Crypto Wallet": {"mean_amount": 1200.0, "std_amount": 800.0, "mean_velocity": 2.5,  "mean_dist": 1200.0},
            "ACH Payment":   {"mean_amount": 450.0,  "std_amount": 200.0, "mean_velocity": 0.4,  "mean_dist": 3.0}
        }
        
    def generate_transaction(self) -> Dict[str, Any]:
        """Generates a single raw, unscaled transaction."""
        self.transaction_counter += 1
        channel = random.choices(self.CHANNELS, weights=[0.55, 0.15, 0.10, 0.20])[0]
        base = self.baseline[channel]
        
        is_anomaly = random.random() < self.anomaly_rate
        
        tx_id = f"TXN-{random.randint(100000, 999999)}"
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        
        if not is_anomaly:
            # Normal transactions follow Gaussian profiles
            amount = max(5.0, np.random.normal(base["mean_amount"], base["std_amount"]))
            velocity = max(1, int(np.random.poisson(base["mean_velocity"]) + 1))
            dist_drift = max(0.1, np.random.normal(base["mean_dist"], base["mean_dist"] * 0.3))
            country_risk = random.choices([1, 2, 3], weights=[0.85, 0.12, 0.03])[0]
            merchant_trust = random.choices([4, 5], weights=[0.30, 0.70])[0]
        else:
            # Anomalous transactions represent extreme multi-dimensional shifts
            anomaly_type = random.choice(["velocity_spike", "amount_burst", "location_jump", "high_risk_country"])
            
            if anomaly_type == "amount_burst":
                amount = base["mean_amount"] * random.uniform(8.0, 15.0)
                velocity = max(1, int(np.random.poisson(base["mean_velocity"]) + 1))
                dist_drift = max(0.1, np.random.normal(base["mean_dist"], base["mean_dist"] * 0.3))
                country_risk = random.choices([1, 2, 3], weights=[0.80, 0.15, 0.05])[0]
                merchant_trust = random.choices([2, 3, 4], weights=[0.4, 0.4, 0.2])[0]
                
            elif anomaly_type == "velocity_spike":
                amount = max(5.0, np.random.normal(base["mean_amount"], base["std_amount"]))
                velocity = int(base["mean_velocity"] * random.uniform(5.0, 10.0))
                dist_drift = max(0.1, np.random.normal(base["mean_dist"], base["mean_dist"] * 0.3))
                country_risk = random.choices([1, 2, 3], weights=[0.85, 0.12, 0.03])[0]
                merchant_trust = random.choices([3, 4, 5], weights=[0.2, 0.4, 0.4])[0]
                
            elif anomaly_type == "location_jump":
                amount = max(5.0, np.random.normal(base["mean_amount"], base["std_amount"]))
                velocity = max(1, int(np.random.poisson(base["mean_velocity"]) + 1))
                dist_drift = base["mean_dist"] * random.uniform(25.0, 50.0)
                country_risk = random.choices([2, 3], weights=[0.5, 0.5])[0]
                merchant_trust = random.choices([1, 2, 3], weights=[0.3, 0.5, 0.2])[0]
                
            else:  # high_risk_country
                amount = max(5.0, np.random.normal(base["mean_amount"], base["std_amount"]))
                velocity = max(1, int(np.random.poisson(base["mean_velocity"]) + 1))
                dist_drift = max(0.1, np.random.normal(base["mean_dist"], base["mean_dist"] * 0.3))
                country_risk = 5  # Critical risk country
                merchant_trust = 1  # Untrusted vendor
                
        # Clean rounding
        amount = round(amount, 2)
        dist_drift = round(dist_drift, 2)
        country = random.choice(self.COUNTRIES) if country_risk < 4 else random.choice(["RU", "KP", "IR", "SY"])
        
        return {
            "tx_id": tx_id,
            "timestamp": timestamp,
            "channel": channel,
            "amount": amount,
            "velocity": velocity,
            "distance_drift": dist_drift,
            "country_risk": country_risk,
            "merchant_trust": merchant_trust,
            "country": country,
            "ground_truth_anomaly": is_anomaly
        }


# ==============================================================================
# 2. STREAMING ETL PIPELINE
# ==============================================================================
class PipelineETL:
    """Performs real-time cleaning, imputation, transformation, and feature scaling."""
    
    def __init__(self):
        # Operational parameters derived from baseline historical training sets
        self.feature_means = {"amount": 800.0, "velocity": 1.5, "distance_drift": 180.0}
        self.feature_stds = {"amount": 1500.0, "velocity": 1.2, "distance_drift": 450.0}
        
    def transform(self, tx: Dict[str, Any]) -> Tuple[np.ndarray, List[str]]:
        """
        Processes a raw transaction dictionary, performing:
        1. Imputation (handling missing entries)
        2. Continuous Scaling (Z-Score Normalization)
        3. Categorical Encodings
        Returns scaled vector and labels list.
        """
        # 1. Extraction & Imputation
        amount = tx.get("amount", self.feature_means["amount"])
        velocity = tx.get("velocity", self.feature_means["velocity"])
        distance = tx.get("distance_drift", self.feature_drift_impute(tx))
        
        # 2. Continuous scaling (Z-Score Standardization: (X - mu) / sigma)
        scaled_amount = (amount - self.feature_means["amount"]) / self.feature_stds["amount"]
        scaled_velocity = (velocity - self.feature_means["velocity"]) / self.feature_stds["velocity"]
        scaled_distance = (distance - self.feature_stds["distance_drift"]) / self.feature_stds["distance_drift"]
        
        # 3. Scale risk categories ordinal features
        scaled_country = (tx.get("country_risk", 1) - 1.0) / 4.0  # Range: [0, 1]
        scaled_trust = (5.0 - tx.get("merchant_trust", 5)) / 4.0   # Reverse scale (high trust -> low score)
        
        features = np.array([scaled_amount, scaled_velocity, scaled_distance, scaled_country, scaled_trust])
        feature_labels = ["Amount", "Velocity", "Distance Drift", "Country Risk", "Merchant Risk"]
        
        return features, feature_labels

    def feature_drift_impute(self, tx: Dict[str, Any]) -> float:
        """Fallback dynamic imputation based on historical median channel distance."""
        channel = tx.get("channel", "Credit Card")
        return {"Credit Card": 8.0, "Wire Transfer": 150.0, "Crypto Wallet": 1200.0, "ACH Payment": 3.0}.get(channel, 50.0)


# ==============================================================================
# 3. UNSUPERVISED ANOMALY MODEL (Reconstruction Deviance Autoencoder Analogue)
# ==============================================================================
class AnomalyEngine:
    """
    Implements an unsupervised mathematical distance/reconstruction model.
    Models normal transactional space and computes the multi-dimensional error.
    """
    
    def __init__(self, threshold: float = 2.1):
        self.threshold = threshold
        
        # Ideal 'normal' centroid inside standard scaled feature space
        # Unsupervised behavior profile centers around standard standardizations
        self.normal_profile_centroid = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        
        # Feature covariance matrix scale modifiers
        self.weights = np.array([1.5, 1.8, 1.2, 1.4, 1.3])
        
    def evaluate(self, scaled_features: np.ndarray) -> Tuple[float, bool]:
        """
        Calculates reconstruction deviance (Mahalanobis-inspired weighted distance)
        to identify features diverging significantly from expectation.
        """
        deviance_vector = scaled_features - self.normal_profile_centroid
        weighted_deviance = deviance_vector * self.weights
        
        # Euclidean Reconstruction Error
        reconstruction_error = np.sqrt(np.sum(weighted_deviance ** 2))
        
        is_anomaly = reconstruction_error > self.threshold
        return float(reconstruction_error), bool(is_anomaly)


# ==============================================================================
# 4. EXPLAINABLE AI (XAI) ENGINE
# ==============================================================================
class ExplainableAI:
    """Calculates spatial-reconstruction squared deviations to isolate anomalous drivers."""
    
    @staticmethod
    def get_contributions(scaled_features: np.ndarray, labels: List[str], weights: np.ndarray) -> List[Tuple[str, float]]:
        """
        Breaks down the reconstruction error to assign percentage contribution scores
        to each feature. This gives diagnostic insight to risk analysts.
        """
        squared_deviations = (scaled_features ** 2) * (weights ** 2)
        total_dev = np.sum(squared_deviations)
        
        if total_dev == 0:
            return [(lbl, 20.0) for lbl in labels]
            
        contributions = []
        for i, lbl in enumerate(labels):
            percentage = (squared_deviations[i] / total_dev) * 100.0
            contributions.append((lbl, round(percentage, 1)))
            
        # Sort by importance (highest contribution first)
        return sorted(contributions, key=lambda x: x[1], reverse=True)


# ==============================================================================
# 5. DYNAMIC INTERACTIVE TERMINAL DASHBOARD
# ==============================================================================
class DashboardApp:
    """Core UI Application utilizing standard terminal frames and rich telemetry tables."""
    
    def __init__(self):
        self.simulator = DataStreamSimulator()
        self.pipeline = PipelineETL()
        self.model = AnomalyEngine()
        self.console = Console()
        
        # Live Stats variables
        self.total_processed = 0
        self.total_volume = 0.0
        self.anomalies_flagged = 0
        self.value_at_risk = 0.0
        self.start_time = time.time()
        
        # UI history feeds
        self.recent_tx_buffer: List[Dict[str, Any]] = []
        self.pipeline_logs: List[str] = [
            "[dim]SYSTEM: ETL Ingestion Pipeline Ready.[/dim]",
            "[dim]SYSTEM: Autoencoder Neural Space Initialized.[/dim]",
            "[dim]SYSTEM: Waiting for transaction stream ingestion...[/dim]"
        ]
        self.active_diagnostic: Dict[str, Any] = {}
        
    def add_pipeline_log(self, text: str):
        """Maintains clean running log sequence."""
        self.pipeline_logs.append(text)
        if len(self.pipeline_logs) > 6:
            self.pipeline_logs.pop(0)
            
    def update_metrics(self, tx: Dict[str, Any], is_flagged: bool):
        """Integrates mathematical summary changes from incoming live records."""
        self.total_processed += 1
        self.total_volume += tx["amount"]
        
        if is_flagged:
            self.anomalies_flagged += 1
            self.value_at_risk += tx["amount"]
            
    def generate_header(self) -> Panel:
        """Constructs glowing cyberpunk banner."""
        uptime = round(time.time() - self.start_time, 1)
        header_text = Text()
        header_text.append("🛡️  AEGIS RISK SYSTEM ", style="bold purple")
        header_text.append("|", style="dim cyan")
        header_text.append(" REAL-TIME TRANSACTION STREAMING INTELLIGENCE ENGINE ", style="bold white")
        header_text.append("|", style="dim cyan")
        header_text.append(f" ACTIVE UPTIME: {uptime}s", style="bold green")
        
        return Panel(Align.center(header_text), border_style="purple", box=Panel.box.HEAVY)
        
    def generate_kpi_panel(self) -> Panel:
        """Renders odometer-like stats counters and dynamic calculations."""
        tps = round(self.total_processed / max(1, time.time() - self.start_time), 1)
        anomaly_rate = round((self.anomalies_flagged / max(1, self.total_processed)) * 100, 1)
        
        table = Table.grid(padding=(0, 1))
        table.add_column("Metric", style="bold cyan")
        table.add_column("Value", style="bold white justify-right")
        
        table.add_row("Processed Txns", f"{self.total_processed:,}")
        table.add_row("Throughput", f"{tps} TPS")
        table.add_row("Total Inspected Vol", f"${self.total_volume:,.2f}")
        table.add_row("Anomalies Flagged", f"[bold red]{self.anomalies_flagged}[/]")
        table.add_row("Anomaly Detection Rate", f"[bold yellow]{anomaly_rate}%[/]")
        table.add_row("Estimated Value-at-Risk", f"[bold red]${self.value_at_risk:,.2f}[/]")
        
        return Panel(table, title="[bold purple]📊 TELEMETRY GAUGES[/bold purple]", border_style="cyan")
        
    def generate_stream_panel(self) -> Panel:
        """Displays scrolling multi-channel transactions with colored highlighting."""
        table = Table(box=Panel.box.SIMPLE, expand=True)
        table.add_column("Txn ID", style="dim")
        table.add_column("Time", style="dim")
        table.add_column("Channel", style="bold")
        table.add_column("Amount", justify="right")
        table.add_column("Country", justify="center")
        table.add_column("Velocity", justify="center")
        table.add_column("Risk Score", justify="right")
        
        for record in reversed(self.recent_tx_buffer[-8:]):
            is_anomaly = record["pred_anomaly"]
            
            # Formatting styles
            color = "red" if is_anomaly else "green"
            row_style = "bold red" if is_anomaly else ""
            
            table.add_row(
                record["tx_id"],
                record["timestamp"],
                record["channel"],
                f"${record['amount']:.2f}",
                record["country"],
                str(record["velocity"]),
                f"[{color}]{record['risk_score']:.2f}[/]",
                style=row_style
            )
            
        return Panel(table, title="[bold purple]📡 STREAM TICKER (LIVE INSPECTION)[/bold purple]", border_style="purple")

    def generate_xai_panel(self) -> Panel:
        """Explains active flagged anomalies with contribution bars."""
        if not self.active_diagnostic:
            return Panel(
                Align.center(Text("\n[dim]No critical anomaly currently flagged.\nListening to stream...[/dim]")),
                title="[bold red]🧠 DIAGNOSTIC COMPLIANCE (EXPLAINABLE AI)[/bold red]",
                border_style="red"
            )
            
        diag = self.active_diagnostic
        diag_text = Text()
        diag_text.append(f"CRITICAL RISK ALERT: {diag['tx_id']} flagged in {diag['channel']}\n", style="bold red")
        diag_text.append(f"Reconstruction Deviation: {diag['risk_score']:.2f} (Threshold: {diag['thresh']})\n\n", style="yellow")
        diag_text.append("Mathematical Contribution breakdown:\n", style="bold white")
        
        panel_layout = Table.grid(padding=(0, 2))
        panel_layout.add_column("Feature")
        panel_layout.add_column("Impact", width=25)
        
        progress = Progress(
            TextColumn("{task.description}"),
            BarColumn(bar_width=15, complete_style="red", finished_style="red"),
            TextColumn("[bold red]{task.percentage:>3.1f}%[/bold red]")
        )
        
        for feature, pct in diag["contributions"]:
            progress.add_task(f"[bold dim]{feature:15}[/]", total=100, completed=pct)
            
        return Panel(
            Align.center(progress),
            title="[bold red]🧠 DIAGNOSTIC COMPLIANCE (EXPLAINABLE AI)[/bold red]",
            border_style="red"
        )
        
    def generate_logs_panel(self) -> Panel:
        """Shows pipeline trace details."""
        log_text = Text()
        for log in self.pipeline_logs:
            log_text.append(log + "\n")
        return Panel(log_text, title="[bold purple]⚙️ ETL SANBOX LOGS[/bold purple]", border_style="cyan")

    def run(self):
        """Runs the main live application thread."""
        layout = Layout()
        layout.split(
            Layout(name="header", size=4),
            Layout(name="body", ratio=1)
        )
        
        # Split body into columns
        layout["body"].split_row(
            Layout(name="left", ratio=1),
            Layout(name="right", ratio=2)
        )
        
        # Split left panel vertically
        layout["left"].split_column(
            Layout(name="kpis", ratio=1),
            Layout(name="logs", ratio=1)
        )
        
        # Split right panel vertically
        layout["right"].split_column(
            Layout(name="stream", ratio=2),
            Layout(name="xai", ratio=1)
        )
        
        with Live(layout, refresh_per_second=8) as live:
            while True:
                # 1. Sim Ingestion
                tx = self.simulator.generate_transaction()
                
                # 2. ETL Sandbox transform
                features, labels = self.pipeline.transform(tx)
                self.add_pipeline_log(
                    f"[green]ETL Ingested[/green] [dim]{tx['tx_id']}[/dim] -> scaled vector standard scaling completed."
                )
                
                # 3. ML Prediction Anomaly Scoring
                risk_score, is_flagged = self.model.evaluate(features)
                tx["risk_score"] = risk_score
                tx["pred_anomaly"] = is_flagged
                
                # 4. Integrate stats tracking
                self.update_metrics(tx, is_flagged)
                self.recent_tx_buffer.append(tx)
                
                # 5. Diagnostic explainability calculations
                if is_flagged:
                    contributions = ExplainableAI.get_contributions(features, labels, self.model.weights)
                    self.active_diagnostic = {
                        "tx_id": tx["tx_id"],
                        "channel": tx["channel"],
                        "risk_score": risk_score,
                        "thresh": self.model.threshold,
                        "contributions": contributions
                    }
                    self.add_pipeline_log(
                        f"[bold red]ALERT Anomaly detected![/bold red] Flagged {tx['tx_id']} with score {risk_score:.2f}."
                    )
                
                # Update Layout parts
                layout["header"].update(self.generate_header())
                layout["kpis"].update(self.generate_kpi_panel())
                layout["logs"].update(self.generate_logs_panel())
                layout["stream"].update(self.generate_stream_panel())
                layout["xai"].update(self.generate_xai_panel())
                
                # Dynamic stream speed simulation
                sleep_delay = random.uniform(0.3, 0.8)
                if tx["ground_truth_anomaly"]:
                    sleep_delay = random.uniform(0.1, 0.3)  # Speeds up during anomaly bursts to simulate congestion
                    
                time.sleep(sleep_delay)


if __name__ == "__main__":
    app = DashboardApp()
    try:
        app.run()
    except KeyboardInterrupt:
        print("\n\n[🛡️] Aegis risk scoring pipeline safely shut down. System out.")
