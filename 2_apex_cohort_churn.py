"""
APEX: SaaS Customer Cohort Intelligence & Churn/LTV Predictor
=============================================================
An advanced Python-based predictive analytics platform for subscription finance.
This application tracks subscriber sign-ups, computes rolling Monthly Recurring Revenue (MRR),
generates dynamic cohort matrix heatmaps, and applies statistical Survival Analysis 
(Kaplan-Meier survival curves and Cox Proportional Hazards hazard ratio scaling) 
to forecast Customer Lifetime Value (LTV) and pinpoint customer attrition windows.

Built with high visual polish using the `rich` terminal UI framework.
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
    print("\n[!] 'numpy' is required for the Apex Analytics Engine.")
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
except ImportError:
    print("\n[!] 'rich' is required for the high-fidelity terminal UI.")
    print("    Please run: pip install numpy rich")
    sys.exit(1)


# ==============================================================================
# 1. MATHEMATICAL SURVIVAL MODELS (Pure NumPy / Math Implementation)
# ==============================================================================
class SurvivalAnalytics:
    """
    Implements core survival statistics mathematically on-the-fly.
    Includes Kaplan-Meier survival curves and Cox Proportional Hazard adjustments.
    """
    
    @staticmethod
    def calculate_kaplan_meier(timeline: np.ndarray, event_times: List[float], censor_times: List[float]) -> np.ndarray:
        """
        Computes Kaplan-Meier survival curve S(t) = Product(1 - d_i / n_i)
        Where:
          d_i = number of deaths (churns) at time t_i
          n_i = number of subjects at risk just before t_i
        """
        all_times = sorted(list(set(event_times + censor_times)))
        survival_probs = []
        current_prob = 1.0
        
        # Calculate survival probabilities for each distinct time point
        for t in timeline:
            # Subjects at risk (duration >= t)
            at_risk = sum(1 for d in event_times if d >= t) + sum(1 for c in censor_times if c >= t)
            # Deaths at this exact interval
            deaths = sum(1 for d in event_times if d == t)
            
            if at_risk > 0:
                current_prob *= (1.0 - (deaths / at_risk))
            
            survival_probs.append(current_prob)
            
        return np.array(survival_probs)

    @staticmethod
    def calculate_hazard_multiplier(support_tickets: int, pricing_tier: float, contract_months: int) -> float:
        """
        Implements a Cox Proportional Hazards Model linear predictor: h(t) = h_0(t) * exp(beta * X)
        Where features are:
          - X_1: Support ticket velocity (Beta_1 = +0.22 -> increases churn risk)
          - X_2: Pricing tier ($1=Basic, $2=Growth, $3=Enterprise; Beta_2 = -0.15 -> enterprise churns less)
          - X_3: Contract Term (Beta_3 = -0.45 -> annual contracts lock in users)
        """
        # Hazard coefficients (Betas)
        beta_tickets = 0.25
        beta_price = -0.12
        beta_contract = -0.50
        
        # Linear combinations
        linear_predictor = (beta_tickets * support_tickets) + (beta_price * pricing_tier) + (beta_contract * contract_months)
        
        # Hazard ratio multiplier (exp(LP))
        return math.exp(linear_predictor)


# ==============================================================================
# 2. SUBSCRIBER STREAM SIMULATOR
# ==============================================================================
class SubscriberSimulator:
    """Generates continuous streams of subscriber lifetime events and updates cohort structures."""
    
    TIERS = ["Basic", "Growth", "Enterprise"]
    TIER_PRICES = {"Basic": 49.0, "Growth": 149.0, "Enterprise": 499.0}
    
    def __init__(self):
        self.subscriber_counter = 0
        self.active_subscribers: Dict[str, Dict[str, Any]] = {}
        self.churned_subscribers: List[Dict[str, Any]] = []
        
        # Initialize cohort matrix variables
        # Row: Cohort Signup Month (Jan - May), Col: Retention Months (M0 - M5)
        self.cohorts = ["Jan", "Feb", "Mar", "Apr", "May"]
        self.cohort_sizes = {c: 0 for c in self.cohorts}
        # Matrix tracks percentage retention
        self.cohort_retention = {
            c: [100.0, 100.0, 100.0, 100.0, 100.0, 100.0] for c in self.cohorts
        }
        
        # Seed initial baseline populations to make the analytics look full and realistic
        self._seed_historical_cohorts()
        
    def _seed_historical_cohorts(self):
        """Seeds realistic historical customer distributions for the cohort matrix."""
        # Seeding Jan cohort (5 months ago)
        self.cohort_sizes["Jan"] = 120
        self.cohort_retention["Jan"] = [100.0, 92.5, 87.0, 81.2, 76.5, 73.0]
        
        # Feb cohort (4 months ago)
        self.cohort_sizes["Feb"] = 145
        self.cohort_retention["Feb"] = [100.0, 94.2, 89.5, 84.0, 79.2, 0.0]  # M5 not reached yet
        
        # Mar cohort (3 months ago)
        self.cohort_sizes["Mar"] = 160
        self.cohort_retention["Mar"] = [100.0, 95.0, 91.2, 87.5, 0.0, 0.0]
        
        # Apr cohort (2 months ago)
        self.cohort_sizes["Apr"] = 185
        self.cohort_retention["Apr"] = [100.0, 96.8, 93.0, 0.0, 0.0, 0.0]
        
        # May cohort (Current Month)
        self.cohort_sizes["May"] = 90
        self.cohort_retention["May"] = [100.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        
        # Seed some active subscribers
        for i in range(150):
            sub_id = f"SUB-{random.randint(10000, 99999)}"
            tier = random.choices(self.TIERS, weights=[0.60, 0.30, 0.10])[0]
            signup_cohort = random.choice(self.cohorts)
            self.active_subscribers[sub_id] = {
                "sub_id": sub_id,
                "tier": tier,
                "mrr": self.TIER_PRICES[tier],
                "cohort": signup_cohort,
                "duration": random.randint(1, 5),
                "tickets": random.choices([0, 1, 2, 3], weights=[0.65, 0.20, 0.10, 0.05])[0],
                "contract_type": random.choice([1, 12])  # 1 month or 12 month lock
            }

    def generate_next_event(self) -> Tuple[str, Dict[str, Any]]:
        """
        Simulates active live subscription activities:
        1. New sign-up (+MRR, grows current cohort size)
        2. Customer support ticket (increases Hazard ratio)
        3. Subscription upgrade/downgrade (alters LTV projections)
        4. Customer Churn (ends stream tracking, adds to survival curves)
        """
        event_type = random.choices(
            ["signup", "ticket", "upgrade", "churn_check"], 
            weights=[0.35, 0.40, 0.10, 0.15]
        )[0]
        
        if event_type == "signup" or not self.active_subscribers:
            # 1. NEW SIGN-UP
            self.subscriber_counter += 1
            sub_id = f"SUB-{random.randint(50000, 99999)}"
            tier = random.choices(self.TIERS, weights=[0.55, 0.35, 0.10])[0]
            cohort = "May"  # Current active cohort
            mrr = self.TIER_PRICES[tier]
            contract = random.choice([1, 12])
            
            self.active_subscribers[sub_id] = {
                "sub_id": sub_id,
                "tier": tier,
                "mrr": mrr,
                "cohort": cohort,
                "duration": 0,
                "tickets": 0,
                "contract_type": contract
            }
            
            # Update cohort sizes
            self.cohort_sizes["May"] += 1
            
            desc = f"[bold green]SIGN-UP[/bold green]: Customer [bold]{sub_id}[/bold] subscribed to [bold cyan]{tier} Tier[/bold cyan] (+${mrr}/mo)"
            return "signup", {"sub_id": sub_id, "desc": desc, "mrr": mrr}
            
        elif event_type == "ticket":
            # 2. SUPPORT TICKET CREATION
            sub_id = random.choice(list(self.active_subscribers.keys()))
            sub = self.active_subscribers[sub_id]
            sub["tickets"] += 1
            
            # Calculate updated Cox Hazard Multiplier
            hazard_mult = SurvivalAnalytics.calculate_hazard_multiplier(
                sub["tickets"], 
                self.TIERS.index(sub["tier"]) + 1, 
                sub["contract_type"]
            )
            
            desc = f"[bold yellow]SUPPORT TICKET[/bold yellow]: [bold]{sub_id}[/bold] raised complaint. Total: {sub['tickets']} | [bold red]Hazard Ratio: {hazard_mult:.2f}x[/bold red]"
            return "ticket", {"sub_id": sub_id, "desc": desc, "hazard": hazard_mult}
            
        elif event_type == "upgrade":
            # 3. SERVICE TIER UPGRADE
            sub_id = random.choice(list(self.active_subscribers.keys()))
            sub = self.active_subscribers[sub_id]
            
            if sub["tier"] != "Enterprise":
                old_tier = sub["tier"]
                new_tier = "Enterprise" if old_tier == "Growth" else "Growth"
                sub["tier"] = new_tier
                old_mrr = sub["mrr"]
                new_mrr = self.TIER_PRICES[new_tier]
                sub["mrr"] = new_mrr
                mrr_diff = new_mrr - old_mrr
                
                desc = f"[bold blue]UPGRADE[/bold blue]: [bold]{sub_id}[/bold] migrated [dim]{old_tier}[/dim] ➡️ [bold green]{new_tier}[/bold green] (+${mrr_diff:.0f} MRR)"
                return "upgrade", {"sub_id": sub_id, "desc": desc, "mrr_diff": mrr_diff}
                
            desc = f"[bold dim]ENGAGEMENT[/bold dim]: [bold]{sub_id}[/bold] active in dashboard."
            return "active", {"sub_id": sub_id, "desc": desc}
            
        else:
            # 4. CHURN RISK ASSESSMENT
            sub_id = random.choice(list(self.active_subscribers.keys()))
            sub = self.active_subscribers[sub_id]
            
            # Compute real-time hazard multiplier to see if they churn
            hazard_mult = SurvivalAnalytics.calculate_hazard_multiplier(
                sub["tickets"], 
                self.TIERS.index(sub["tier"]) + 1, 
                sub["contract_type"]
            )
            
            # Base probability of monthly attrition adjusted by hazard multiplier
            base_churn_prob = 0.05
            adjusted_prob = min(0.95, base_churn_prob * hazard_mult)
            
            if random.random() < adjusted_prob:
                # Trigger Customer Churn
                churned_sub = self.active_subscribers.pop(sub_id)
                self.churned_subscribers.append(churned_sub)
                
                # Deduct from cohort matrix retention (simulation representation)
                cohort = churned_sub["cohort"]
                dur = min(5, churned_sub["duration"])
                # Adjust cohort percentage downwards
                for m in range(dur + 1, 6):
                    self.cohort_retention[cohort][m] = max(40.0, self.cohort_retention[cohort][m] - random.uniform(0.5, 2.0))
                
                ltv_generated = churned_sub["mrr"] * max(1, churned_sub["duration"])
                desc = f"[bold red]CHURN ALERT[/bold red]: [bold]{sub_id}[/bold] cancelled subscription ({churned_sub['tier']} Tier) after {churned_sub['duration']} mos. LTV: ${ltv_generated:.2f}"
                return "churn", {"sub_id": sub_id, "desc": desc, "mrr_loss": churned_sub["mrr"], "ltv": ltv_generated}
            
            # Customer survived the billing interval, increment active duration
            sub["duration"] += 1
            desc = f"[dim]RENEWAL[/dim]: [bold]{sub_id}[/bold] billed for month {sub['duration']} successfully."
            return "renewal", {"sub_id": sub_id, "desc": desc}


# ==============================================================================
# 3. DYNAMIC INTERACTIVE TERMINAL DASHBOARD
# ==============================================================================
class ChurnDashboardApp:
    """Renders advanced SaaS cohort and survival dashboard in standard terminal."""
    
    def __init__(self):
        self.simulator = SubscriberSimulator()
        self.console = Console()
        self.start_time = time.time()
        
        # Historical running aggregates
        self.total_mrr_loss = 0.0
        self.total_ltv_recovered = 0.0
        self.stream_events: List[str] = [
            "[dim]SYSTEM: SaaS Cohort Ledger Engaged.[/dim]",
            "[dim]SYSTEM: Survival Math Models Loaded (Kaplan-Meier, Cox Proportional Hazards).[/dim]",
            "[dim]SYSTEM: Listening to live customer lifecycle events...[/dim]"
        ]
        
    def add_event_log(self, log_str: str):
        self.stream_events.append(log_str)
        if len(self.stream_events) > 5:
            self.stream_events.pop(0)

    def generate_header(self) -> Panel:
        """Constructs glowing SaaS dashboard header."""
        uptime = round(time.time() - self.start_time, 1)
        header_text = Text()
        header_text.append("📈  APEX COHORT INTELLIGENCE ", style="bold green")
        header_text.append("|", style="dim white")
        header_text.append(" PREDICTIVE CHURN & CUSTOMER LIFETIME VALUE ENGINE ", style="bold white")
        header_text.append("|", style="dim white")
        header_text.append(f" ACTIVE UPTIME: {uptime}s", style="bold yellow")
        return Panel(Align.center(header_text), border_style="green", box=Panel.box.HEAVY)
        
    def generate_kpis_panel(self) -> Panel:
        """Computes SaaS metric indicators dynamically from the simulated population."""
        active_cnt = len(self.simulator.active_subscribers)
        current_mrr = sum(sub["mrr"] for sub in self.simulator.active_subscribers.values())
        
        # Average LTV computation
        all_durations = [sub["duration"] for sub in self.simulator.active_subscribers.values()]
        avg_lifespan = np.mean(all_durations) if all_durations else 3.2
        avg_ltv = (current_mrr / max(1, active_cnt)) * avg_lifespan if active_cnt else 0.0
        
        # Gross Churn calculations
        total_historical = active_cnt + len(self.simulator.churned_subscribers)
        churn_rate = (len(self.simulator.churned_subscribers) / max(1, total_historical)) * 100.0
        
        table = Table.grid(padding=(0, 1))
        table.add_column("SaaS Metric", style="bold cyan")
        table.add_column("Value", style="bold white justify-right")
        
        table.add_row("Active Subscribers", f"{active_cnt:,}")
        table.add_row("Monthly Recurring Revenue (MRR)", f"${current_mrr:,.2f}")
        table.add_row("Annual Run Rate (ARR)", f"${current_mrr * 12:,.2f}")
        table.add_row("Average Active Lifespan", f"{avg_lifespan:.1f} months")
        table.add_row("Projected Customer LTV", f"[bold green]${avg_ltv:,.2f}[/bold green]")
        table.add_row("Gross Customer Churn Rate", f"[bold red]{churn_rate:.1f}%[/bold red]")
        
        return Panel(table, title="[bold green]📊 CORE FINANCIAL KPIs[/bold green]", border_style="cyan")
        
    def generate_cohort_matrix_panel(self) -> Panel:
        """Renders the cohort matrix heatmap table with color codes."""
        table = Table(box=Panel.box.SIMPLE, expand=True)
        table.add_column("Cohort Month", style="bold cyan")
        table.add_column("Size", justify="right", style="dim")
        table.add_column("Month 0", justify="center")
        table.add_column("Month 1", justify="center")
        table.add_column("Month 2", justify="center")
        table.add_column("Month 3", justify="center")
        table.add_column("Month 4", justify="center")
        table.add_column("Month 5", justify="center")
        
        for cohort in self.simulator.cohorts:
            size = self.simulator.cohort_sizes[cohort]
            row_items = [cohort, f"{size} users"]
            
            for m in range(6):
                val = self.simulator.cohort_retention[cohort][m]
                if val == 0.0:
                    row_items.append("[dim]-[/dim]")
                else:
                    # Color encode the retention metric (heatmap)
                    if val >= 90.0:
                        color = "green"
                    elif val >= 80.0:
                        color = "yellow"
                    else:
                        color = "red"
                    row_items.append(f"[{color}]{val:.1f}%[/{color}]")
                    
            table.add_row(*row_items)
            
        return Panel(table, title="[bold green]📊 CUSTOMER COHORT RETENTION MATRIX (HEATMAP)[/bold green]", border_style="green")
        
    def generate_survival_curve_panel(self) -> Panel:
        """
        Plots an ASCII Kaplan-Meier Survival Probability curve over a 12-month interval.
        This provides technical recruiters visual proof of statistical capabilities.
        """
        # Collect survival statistics from simulator populations
        event_times = [float(sub["duration"]) for sub in self.simulator.churned_subscribers]
        censor_times = [float(sub["duration"]) for sub in self.simulator.active_subscribers.values()]
        
        timeline = np.arange(0, 13)
        survival_curve = SurvivalAnalytics.calculate_kaplan_meier(timeline, event_times, censor_times)
        
        # Build ASCII visualization table
        table = Table.grid(padding=(0, 2))
        table.add_column("Time (mo)", style="dim justify-right")
        table.add_column("Survival Probability S(t)", width=35)
        table.add_column("P(Survival)", style="bold green justify-right")
        
        for t, prob in zip(timeline[::2], survival_curve[::2]):
            bar_width = int(prob * 20.0)
            bar_visual = "█" * bar_width + "░" * (20 - bar_width)
            
            color = "green" if prob > 0.80 else "yellow" if prob > 0.60 else "red"
            
            table.add_row(
                f"Month {t:02d}",
                f"[{color}]{bar_visual}[/{color}]",
                f"[{color}]{prob * 100.0:.1f}%[/{color}]"
            )
            
        return Panel(
            Align.center(table),
            title="[bold green]🔬 STATISTICAL SURVIVAL ANALYSIS (KAPLAN-MEIER CURVE)[/bold green]",
            border_style="green"
        )
        
    def generate_events_panel(self) -> Panel:
        """Displays raw events stream."""
        events_text = Text()
        for ev in self.stream_events:
            events_text.append(ev + "\n")
        return Panel(events_text, title="[bold green]🔔 REAL-TIME ENGAGEMENT / EVENT LOGS[/bold green]", border_style="cyan")
        
    def run(self):
        """Runs the main visualization thread."""
        layout = Layout()
        layout.split(
            Layout(name="header", size=4),
            Layout(name="body", ratio=1)
        )
        
        # Split body into columns
        layout["body"].split_row(
            Layout(name="left", ratio=1),
            Layout(name="right", ratio=1)
        )
        
        # Split left panel vertically
        layout["left"].split_column(
            Layout(name="kpis", ratio=1),
            Layout(name="events", ratio=1)
        )
        
        # Split right panel vertically
        layout["right"].split_column(
            Layout(name="cohort", ratio=1),
            Layout(name="survival", ratio=1)
        )
        
        with Live(layout, refresh_per_second=8) as live:
            while True:
                # 1. Sim Next Lifecycle Event
                ev_type, payload = self.simulator.generate_next_event()
                
                # 2. Update status log trace
                self.add_event_log(payload["desc"])
                
                # 3. Capture specific MRR adjustments
                if ev_type == "churn":
                    self.total_mrr_loss += payload["mrr_loss"]
                elif ev_type == "signup":
                    self.total_ltv_recovered += payload["mrr"]
                
                # Update Layout elements
                layout["header"].update(self.generate_header())
                layout["kpis"].update(self.generate_kpis_panel())
                layout["events"].update(self.generate_events_panel())
                layout["cohort"].update(self.generate_cohort_matrix_panel())
                layout["survival"].update(self.generate_survival_curve_panel())
                
                # Regular event pacing sleep
                time.sleep(random.uniform(0.5, 1.2))


if __name__ == "__main__":
    app = ChurnDashboardApp()
    try:
        app.run()
    except KeyboardInterrupt:
        print("\n\n[📈] Apex cohort prediction ledger safely finalized. System out.")
