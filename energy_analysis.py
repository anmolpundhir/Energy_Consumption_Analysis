"""Energy Consumption Trend Analysis - AEP Hourly Data
Generates charts and a PDF report covering peak demand, seasonal trends,
consumption patterns, and distribution recommendations.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
import os

OUT_DIR = "Charts"
os.makedirs(OUT_DIR, exist_ok=True)

# ---- Load & clean ----
df = pd.read_csv(r"C:\Users\ASUS\OneDrive\Desktop\Energy_Consumption_Analysis\AEP_hourly.csv")
df["Datetime"] = pd.to_datetime(df["Datetime"])
df = df.drop_duplicates(subset="Datetime").sort_values("Datetime").reset_index(drop=True)
df = df.rename(columns={"AEP_MW": "MW"})

df["year"] = df["Datetime"].dt.year
df["month"] = df["Datetime"].dt.month
df["hour"] = df["Datetime"].dt.hour
df["dow"] = df["Datetime"].dt.dayofweek  # 0=Mon
df["date"] = df["Datetime"].dt.date
df["is_weekend"] = df["dow"] >= 5

season_map = {12: "Winter", 1: "Winter", 2: "Winter",
              3: "Spring", 4: "Spring", 5: "Spring",
              6: "Summer", 7: "Summer", 8: "Summer",
              9: "Fall", 10: "Fall", 11: "Fall"}
df["season"] = df["month"].map(season_map)

# ---- Key stats ----
stats = {
    "rows": len(df),
    "start": df["Datetime"].min(),
    "end": df["Datetime"].max(),
    "years": df["year"].nunique(),
    "mean": df["MW"].mean(),
    "median": df["MW"].median(),
    "std": df["MW"].std(),
    "min": df["MW"].min(),
    "max": df["MW"].max(),
    "p95": df["MW"].quantile(0.95),
    "p99": df["MW"].quantile(0.99),
}
peak_row = df.loc[df["MW"].idxmax()]
min_row = df.loc[df["MW"].idxmin()]
stats["peak_dt"] = peak_row["Datetime"]
stats["min_dt"] = min_row["Datetime"]

# Daily totals & peaks
daily = df.groupby("date").agg(daily_mwh=("MW", "sum"),
                                daily_peak=("MW", "max"),
                                daily_avg=("MW", "mean")).reset_index()
daily["date"] = pd.to_datetime(daily["date"])

# Top 10 peak days
top10_peaks = daily.nlargest(10, "daily_peak")[["date", "daily_peak"]]

# Seasonal aggregates
season_stats = df.groupby("season")["MW"].agg(["mean", "max", "min", "std"]).reindex(
    ["Winter", "Spring", "Summer", "Fall"])

# Hour-of-day x season heatmap data
hour_season = df.groupby(["season", "hour"])["MW"].mean().unstack(0).reindex(columns=["Winter", "Spring", "Summer", "Fall"])

# Hour-of-day weekday vs weekend
hour_wk = df.groupby(["is_weekend", "hour"])["MW"].mean().unstack(0)
hour_wk.columns = ["Weekday", "Weekend"]

# Monthly avg over years
monthly_year = df.groupby(["year", "month"])["MW"].mean().unstack(0)

# Annual trend
annual = df.groupby("year")["MW"].agg(["mean", "max"]).reset_index()

# Day-of-week
dow_avg = df.groupby("dow")["MW"].mean()
dow_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

# ---- Style ----
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.edgecolor": "#333333",
    "axes.labelcolor": "#333333",
    "xtick.color": "#555555",
    "ytick.color": "#555555",
    "axes.titleweight": "bold",
    "axes.titlesize": 13,
    "axes.labelsize": 10,
    "figure.facecolor": "white",
})
PRIMARY = "#1f4e79"
ACCENT = "#e07b00"
SEASON_COLORS = {"Winter": "#2b6cb0", "Spring": "#38a169", "Summer": "#dd6b20", "Fall": "#9c4221"}

def save(fig, name):
    path = f"{OUT_DIR}/{name}.png"
    fig.savefig(path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path

# Chart 1: Daily peak demand over time
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(daily["date"], daily["daily_peak"], color=PRIMARY, linewidth=0.6, alpha=0.8)
ax.axhline(stats["p95"], color=ACCENT, linestyle="--", linewidth=1, label=f"95th pct ({stats['p95']:,.0f} MW)")
ax.set_title("Daily Peak Demand Over Time")
ax.set_ylabel("MW")
ax.legend(loc="upper right", frameon=False)
ax.grid(alpha=0.25)
save(fig, "01_daily_peak")

# Chart 2: Seasonal distribution (boxplot)
fig, ax = plt.subplots(figsize=(8, 4))
data = [df.loc[df["season"] == s, "MW"].values for s in ["Winter", "Spring", "Summer", "Fall"]]
bp = ax.boxplot(data, labels=["Winter", "Spring", "Summer", "Fall"], patch_artist=True,
                showfliers=False, medianprops=dict(color="black"))
for patch, s in zip(bp["boxes"], ["Winter", "Spring", "Summer", "Fall"]):
    patch.set_facecolor(SEASON_COLORS[s]); patch.set_alpha(0.75)
ax.set_title("Hourly Demand Distribution by Season")
ax.set_ylabel("MW")
ax.grid(axis="y", alpha=0.25)
save(fig, "02_season_box")

# Chart 3: Hour-of-day curves by season
fig, ax = plt.subplots(figsize=(9, 4.2))
for s in ["Winter", "Spring", "Summer", "Fall"]:
    ax.plot(hour_season.index, hour_season[s], label=s, color=SEASON_COLORS[s], linewidth=2)
ax.set_xticks(range(0, 24, 2))
ax.set_title("Average Daily Load Curve by Season")
ax.set_xlabel("Hour of Day")
ax.set_ylabel("Average MW")
ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.18))
ax.grid(alpha=0.25)
save(fig, "03_hour_season")

# Chart 4: Weekday vs weekend
fig, ax = plt.subplots(figsize=(9, 3.8))
ax.plot(hour_wk.index, hour_wk["Weekday"], color=PRIMARY, linewidth=2.2, label="Weekday")
ax.plot(hour_wk.index, hour_wk["Weekend"], color=ACCENT, linewidth=2.2, label="Weekend")
ax.set_xticks(range(0, 24, 2))
ax.set_title("Average Load Curve: Weekday vs Weekend")
ax.set_xlabel("Hour of Day"); ax.set_ylabel("Average MW")
ax.legend(frameon=False); ax.grid(alpha=0.25)
save(fig, "04_weekday_weekend")

# Chart 5: Monthly heatmap year x month
fig, ax = plt.subplots(figsize=(10, 4.5))
hm = monthly_year.T  # months x years
im = ax.imshow(hm.values, aspect="auto", cmap="YlOrRd")
ax.set_yticks(range(12)); ax.set_yticklabels(["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"])
ax.set_xticks(range(len(hm.columns))); ax.set_xticklabels(hm.columns, rotation=45)
ax.set_title("Average Monthly Demand Heatmap (MW)")
cbar = plt.colorbar(im, ax=ax); cbar.set_label("Avg MW")
save(fig, "05_heatmap_year_month")

# Chart 6: Annual trend
fig, ax = plt.subplots(figsize=(9, 3.8))
ax.bar(annual["year"], annual["mean"], color=PRIMARY, alpha=0.85, label="Avg MW")
ax2 = ax.twinx()
ax2.plot(annual["year"], annual["max"], color=ACCENT, marker="o", linewidth=2, label="Peak MW")
ax.set_title("Annual Average vs Peak Demand")
ax.set_ylabel("Average MW", color=PRIMARY); ax2.set_ylabel("Peak MW", color=ACCENT)
ax.set_xticks(annual["year"]); ax.tick_params(axis="x", rotation=45)
ax.grid(axis="y", alpha=0.25)
save(fig, "06_annual_trend")

# Chart 7: Day-of-week
fig, ax = plt.subplots(figsize=(7, 3.5))
colors = [PRIMARY if i < 5 else ACCENT for i in range(7)]
ax.bar(dow_labels, dow_avg.values, color=colors)
ax.set_title("Average Demand by Day of Week")
ax.set_ylabel("Average MW")
ax.grid(axis="y", alpha=0.25)
save(fig, "07_dow")

# Chart 8: Load duration curve
fig, ax = plt.subplots(figsize=(9, 4))
sorted_mw = np.sort(df["MW"].values)[::-1]
pct = np.arange(1, len(sorted_mw) + 1) / len(sorted_mw) * 100
ax.plot(pct, sorted_mw, color=PRIMARY, linewidth=1.5)
ax.fill_between(pct, sorted_mw, alpha=0.2, color=PRIMARY)
ax.axhline(stats["p95"], color=ACCENT, linestyle="--", linewidth=1, label=f"95th pct ({stats['p95']:,.0f} MW)")
ax.set_title("Load Duration Curve")
ax.set_xlabel("% of Hours Exceeded"); ax.set_ylabel("MW")
ax.legend(frameon=False); ax.grid(alpha=0.25)
save(fig, "08_load_duration")

# ---- Build PDF ----
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image,
                                Table, TableStyle, PageBreak)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY

PDF_PATH = "Report/Energy_Consumption_Trend_Analysis.pdf"
os.makedirs("reports", exist_ok=True)

doc = SimpleDocTemplate(PDF_PATH, pagesize=letter,
                        leftMargin=0.7*inch, rightMargin=0.7*inch,
                        topMargin=0.7*inch, bottomMargin=0.7*inch,
                        title="Energy Consumption Trend Analysis")

styles = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=styles["Heading1"], fontSize=22, textColor=colors.HexColor("#1f4e79"),
                    spaceAfter=4, alignment=TA_LEFT)
SUB = ParagraphStyle("SUB", parent=styles["Normal"], fontSize=10, textColor=colors.HexColor("#666666"), spaceAfter=14)
H2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=14, textColor=colors.HexColor("#1f4e79"),
                    spaceBefore=12, spaceAfter=6)
H3 = ParagraphStyle("H3", parent=styles["Heading3"], fontSize=11, textColor=colors.HexColor("#e07b00"),
                    spaceBefore=8, spaceAfter=4)
BODY = ParagraphStyle("BODY", parent=styles["Normal"], fontSize=10, leading=14, alignment=TA_JUSTIFY, spaceAfter=6)
CAP = ParagraphStyle("CAP", parent=styles["Normal"], fontSize=8.5, textColor=colors.HexColor("#666666"),
                     alignment=TA_CENTER, spaceAfter=10)
BULLET = ParagraphStyle("BULLET", parent=BODY, leftIndent=14, bulletIndent=2, spaceAfter=3)

story = []
story.append(Paragraph("Analyzed by Anmol Singh", H3))
story.append(Paragraph("Energy Consumption Trend Analysis", H1))
story.append(Paragraph(f"AEP Hourly Demand &middot; {stats['start']:%b %Y} – {stats['end']:%b %Y} &middot; {stats['rows']:,} hourly observations", SUB))

# Executive summary
story.append(Paragraph("Executive Summary", H2))
exec_text = (
    f"This report analyzes <b>{stats['rows']:,} hours</b> of electricity demand from American Electric Power (AEP) "
    f"spanning <b>{stats['years']} years</b> ({stats['start']:%Y} to {stats['end']:%Y}). The objective is to identify "
    f"peak demand patterns, seasonal trends, and consumption rhythms that inform energy distribution planning. "
    f"Average hourly load is <b>{stats['mean']:,.0f} MW</b> with an absolute peak of <b>{stats['max']:,.0f} MW</b> "
    f"(reached on {stats['peak_dt']:%b %d, %Y at %H:00}). The 95th percentile of demand is "
    f"<b>{stats['p95']:,.0f} MW</b>, meaning the grid must reliably serve loads above this threshold roughly "
    f"<b>438 hours per year</b>."
)
story.append(Paragraph(exec_text, BODY))

# KPI table
kpi_data = [
    ["Metric", "Value"],
    ["Observations", f"{stats['rows']:,} hours"],
    ["Date range", f"{stats['start']:%Y-%m-%d} → {stats['end']:%Y-%m-%d}"],
    ["Mean demand", f"{stats['mean']:,.0f} MW"],
    ["Median demand", f"{stats['median']:,.0f} MW"],
    ["Std deviation", f"{stats['std']:,.0f} MW"],
    ["Minimum", f"{stats['min']:,.0f} MW  ({stats['min_dt']:%Y-%m-%d %H:00})"],
    ["Maximum (peak)", f"{stats['max']:,.0f} MW  ({stats['peak_dt']:%Y-%m-%d %H:00})"],
    ["95th percentile", f"{stats['p95']:,.0f} MW"],
    ["99th percentile", f"{stats['p99']:,.0f} MW"],
]
t = Table(kpi_data, colWidths=[2.2*inch, 4.0*inch])
t.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#1f4e79")),
    ("TEXTCOLOR", (0,0), (-1,0), colors.white),
    ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
    ("FONTSIZE", (0,0), (-1,-1), 9.5),
    ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#f4f6fa")]),
    ("GRID", (0,0), (-1,-1), 0.4, colors.HexColor("#cccccc")),
    ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
    ("LEFTPADDING", (0,0), (-1,-1), 8),
    ("TOPPADDING", (0,0), (-1,-1), 5),
    ("BOTTOMPADDING", (0,0), (-1,-1), 5),
]))
story.append(t)
story.append(Spacer(1, 12))

# Section 1: Peak demand
story.append(PageBreak())
story.append(Paragraph("1. Peak Demand", H2))
story.append(Paragraph(
    f"Peak demand is the highest level of electricity drawn from the grid in any single hour. For AEP, the "
    f"all-time peak in this dataset reached <b>{stats['max']:,.0f} MW</b> on {stats['peak_dt']:%B %d, %Y} at "
    f"{stats['peak_dt']:%H:00}. This single value drives capacity planning: generators, transmission lines, and "
    f"reserves must be sized for this worst case, not the average.", BODY))

story.append(Image("Charts/01_daily_peak.png", width=6.8*inch, height=2.7*inch))
story.append(Paragraph("Figure 1. Daily peak demand. The dashed line marks the 95th percentile of hourly demand.", CAP))

story.append(Image("Charts/08_load_duration.png", width=6.8*inch, height=3*inch))
story.append(Paragraph("Figure 2. Load duration curve — sorted hourly demand showing how often each level is exceeded.", CAP))

story.append(Paragraph("Top 10 peak days", H3))
top_rows = [["#", "Date", "Peak MW"]]
for i, row in enumerate(top10_peaks.itertuples(index=False), 1):
    top_rows.append([str(i), f"{row.date:%Y-%m-%d}", f"{row.daily_peak:,.0f}"])
t = Table(top_rows, colWidths=[0.5*inch, 1.8*inch, 1.4*inch])
t.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#e07b00")),
    ("TEXTCOLOR", (0,0), (-1,0), colors.white),
    ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
    ("FONTSIZE", (0,0), (-1,-1), 9.5),
    ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#fff5e6")]),
    ("GRID", (0,0), (-1,-1), 0.4, colors.HexColor("#cccccc")),
    ("ALIGN", (2,1), (2,-1), "RIGHT"),
    ("LEFTPADDING", (0,0), (-1,-1), 6), ("RIGHTPADDING", (0,0), (-1,-1), 6),
]))
story.append(t)

# Section 2: Seasonal trend
story.append(PageBreak())
story.append(Paragraph("2. Seasonal Trend", H2))
season_summary_rows = [["Season", "Avg MW", "Max MW", "Min MW", "Std MW"]]
for s in ["Winter", "Spring", "Summer", "Fall"]:
    r = season_stats.loc[s]
    season_summary_rows.append([s, f"{r['mean']:,.0f}", f"{r['max']:,.0f}", f"{r['min']:,.0f}", f"{r['std']:,.0f}"])
t = Table(season_summary_rows, colWidths=[1.2*inch, 1.1*inch, 1.1*inch, 1.1*inch, 1.1*inch])
t.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#1f4e79")),
    ("TEXTCOLOR", (0,0), (-1,0), colors.white),
    ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
    ("FONTSIZE", (0,0), (-1,-1), 9.5),
    ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#f4f6fa")]),
    ("GRID", (0,0), (-1,-1), 0.4, colors.HexColor("#cccccc")),
    ("ALIGN", (1,1), (-1,-1), "RIGHT"),
]))
story.append(t)
story.append(Spacer(1, 8))

# Determine highest season
top_season = season_stats["mean"].idxmax()
low_season = season_stats["mean"].idxmin()
peak_season_pct = (season_stats.loc[top_season, "mean"] / season_stats.loc[low_season, "mean"] - 1) * 100
story.append(Paragraph(
    f"<b>{top_season}</b> shows the highest average load ({season_stats.loc[top_season, 'mean']:,.0f} MW), "
    f"about <b>{peak_season_pct:.0f}% above</b> the lowest-demand season (<b>{low_season}</b>, "
    f"{season_stats.loc[low_season, 'mean']:,.0f} MW). The boxplot below confirms that summer also has the "
    f"widest spread, driven by air-conditioning load on hot days. Winter is the secondary peak season due to "
    f"heating, while spring and fall are the natural shoulder seasons — ideal windows for planned maintenance.",
    BODY))

story.append(Image("Charts/02_season_box.png", width=6.5*inch, height=3.2*inch))
story.append(Paragraph("Figure 3. Hourly demand distribution by season (outliers hidden).", CAP))

story.append(Image("Charts/05_heatmap_year_month.png", width=6.8*inch, height=3.1*inch))
story.append(Paragraph("Figure 4. Average monthly demand across all years — confirms summer (Jul/Aug) and winter (Jan/Feb) bi-modal peaks.", CAP))

# Section 3: Consumption pattern
story.append(PageBreak())
story.append(Paragraph("3. Consumption Pattern", H2))
peak_hour = hour_season.mean(axis=1).idxmax()
trough_hour = hour_season.mean(axis=1).idxmin()
peak_dow = dow_labels[dow_avg.idxmax()]
low_dow = dow_labels[dow_avg.idxmin()]
weekday_avg = df.loc[~df["is_weekend"], "MW"].mean()
weekend_avg = df.loc[df["is_weekend"], "MW"].mean()
wk_diff_pct = (weekday_avg / weekend_avg - 1) * 100

story.append(Paragraph(
    f"Daily demand follows a predictable rhythm. Load bottoms out around <b>{trough_hour:02d}:00</b> overnight "
    f"and climbs to a peak near <b>{peak_hour:02d}:00</b>. Weekdays average "
    f"<b>{weekday_avg:,.0f} MW</b>, about <b>{wk_diff_pct:.1f}% higher</b> than weekends "
    f"({weekend_avg:,.0f} MW), reflecting commercial and industrial activity. "
    f"<b>{peak_dow}</b> is the highest-demand day of the week on average; <b>{low_dow}</b> is the lowest.",
    BODY))

story.append(Image("Charts/03_hour_season.png", width=6.8*inch, height=3.1*inch))
story.append(Paragraph("Figure 5. Average load curve by season. Summer shows a strong afternoon peak; winter has a dual morning/evening peak.", CAP))

story.append(Image("Charts/04_weekday_weekend.png", width=6.8*inch, height=2.8*inch))
story.append(Paragraph("Figure 6. Weekday vs weekend hourly load curves.", CAP))

story.append(Image("Charts/07_dow.png", width=5.5*inch, height=2.8*inch))
story.append(Paragraph("Figure 7. Average demand by day of week.", CAP))

story.append(Image("Charts/06_annual_trend.png", width=6.8*inch, height=2.9*inch))
story.append(Paragraph("Figure 8. Annual average and peak demand. Useful for spotting long-term growth or efficiency gains.", CAP))

# Section 4: Recommendations
story.append(PageBreak())
story.append(Paragraph("4. Recommendations for Energy Distribution", H2))
story.append(Paragraph(
    "The patterns above translate directly into operational and planning actions:", BODY))

recs = [
    ("Size capacity to the 99th percentile, not the average.",
     f"Average load is {stats['mean']:,.0f} MW but the 99th percentile is {stats['p99']:,.0f} MW — a "
     f"{(stats['p99']/stats['mean']-1)*100:.0f}% headroom that firm capacity, reserves, and transmission must cover."),
    (f"Pre-position resources for the {peak_hour:02d}:00 hour.",
     "Stage peaking units, dispatchable demand response, and battery storage to discharge into the late-afternoon "
     "ramp. Stagger industrial start-ups to flatten the morning rise."),
    (f"Treat {top_season.lower()} as the critical reliability season.",
     f"Schedule generator outages and major maintenance in {low_season.lower()} and the shoulder months "
     "(spring/fall) when average load is materially lower and weather risk is reduced."),
    ("Use weekend troughs for grid work and storage charging.",
     f"Weekend demand runs {abs(wk_diff_pct):.1f}% below weekdays. This window is ideal for planned switching, "
     "line work, and refilling pumped-hydro or battery storage at low marginal cost."),
    ("Deploy targeted demand-response on the top ~50 peak hours.",
     "The load duration curve shows a steep tail: a small number of hours per year dominate capacity needs. "
     "Voluntary curtailment, time-of-use pricing, and smart-thermostat events on these hours defer expensive "
     "capacity additions."),
    ("Forecast with weather-driven models.",
     "Both summer (cooling) and winter (heating) peaks are temperature-driven. Coupling short-term load forecasts "
     "with hourly weather forecasts will improve unit-commitment accuracy and reduce reserve over-procurement."),
]
for title, body in recs:
    story.append(Paragraph(f"&bull; <b>{title}</b> {body}", BULLET))

story.append(Spacer(1, 14))
story.append(Paragraph("Methodology", H3))
story.append(Paragraph(
    "Data: AEP_hourly.csv from the Kaggle 'Hourly Energy Consumption' dataset (PJM regional load). "
    "Cleaning: deduplicated on timestamp, sorted chronologically, no imputation applied. "
    "Seasons follow meteorological convention (Dec–Feb winter, Mar–May spring, Jun–Aug summer, Sep–Nov fall). "
    "All demand values are in megawatts (MW) and represent hourly average load.", BODY))

doc.build(story)
print(f"PDF saved: {PDF_PATH}")
print(f"Size: {os.path.getsize(PDF_PATH)/1024:.1f} KB")
