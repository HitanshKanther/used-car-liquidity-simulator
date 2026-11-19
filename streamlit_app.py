import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Used-Car Pricing & Market Positioning Simulator",
    layout="wide",
)


@st.cache_resource
def load_assets():
    bundle = joblib.load("models/pricing_bundle.joblib")
    df = pd.read_csv("data/processed/cleaned_used_cars.csv")
    return bundle, df


bundle, clean_df = load_assets()

LUXURY_BRANDS = {
    "BMW",
    "Mercedes-Benz",
    "Audi",
    "Jaguar",
    "Land Rover",
    "Volvo",
    "Porsche",
    "Mini",
    "Lexus",
    "Jeep",
}
NUM_COLS = [
    "Age",
    "km_driven",
    "log_km",
    "km_per_year",
    "engine_l",
    "is_top_trim",
    "has_condition_note",
    "is_7seater",
    "is_luxury",
    "model_supply_train",
]
CAT_COLS = ["Brand", "model", "Transmission", "Owner", "FuelType"]

brands_list = sorted(clean_df["Brand"].unique().tolist())
models_by_brand = (
    clean_df.groupby("Brand")["model"].unique().apply(sorted).to_dict()
)

st.title("Used-Car Listing Price & Market Positioning Simulator")
st.caption(
    "Goa Institute of Management | PGDM (Big Data Analytics) — AMLBT Decision-Support Prototype"
)

st.sidebar.header("1. Vehicle Specifications")
default_brand_idx = (
    brands_list.index("Maruti Suzuki") if "Maruti Suzuki" in brands_list else 0
)
brand = st.sidebar.selectbox("Brand", brands_list, index=default_brand_idx)

avail_models = models_by_brand.get(brand, [])
default_model_idx = (
    avail_models.index("Baleno") if "Baleno" in avail_models else 0
)
model = st.sidebar.selectbox("Model", avail_models, index=default_model_idx)

year = st.sidebar.slider("Manufacturing Year", 2005, 2024, 2020)
km_driven = st.sidebar.number_input(
    "Odometer Reading (km)", min_value=500, max_value=350000, value=18100, step=1000
)
transmission = st.sidebar.radio("Transmission", ["Manual", "Automatic"])
fuel_type = st.sidebar.selectbox("Fuel Type", ["Petrol", "Diesel", "Hybrid/CNG"])
owner = st.sidebar.radio("Owner Count", ["first", "second"])
notes = st.sidebar.text_input(
    "Variant / Condition Notes", value="1.2 Zeta Mint Condition"
)

st.sidebar.header("2. Seller Strategy & Target Price")
strategy = st.sidebar.radio(
    "Seller Objective",
    [
        "Balanced (Market Median)",
        "Fast Sale (High Liquidity)",
        "Maximum Price (Upper Quartile)",
    ],
)
custom_ask = st.sidebar.number_input(
    "Your Planned Asking Price (INR, Optional)",
    min_value=0,
    max_value=15000000,
    value=650000,
    step=10000,
)

age = max(0, 2024 - int(year))
info = str(notes).lower()
eng = pd.Series([info]).str.extract(r"\b([0-5]\.[0-9])\b")[0].astype(float).iloc[0]
if pd.isna(eng):
    eng = bundle["eng_med"].get((brand, model), bundle["global_eng_med"])

row = pd.DataFrame([
    {
        "Age": age,
        "km_driven": float(km_driven),
        "log_km": np.log1p(float(km_driven)),
        "km_per_year": float(km_driven) / (age + 1.0),
        "engine_l": eng,
        "is_top_trim": int(
            bool(
                pd.Series([info])
                .str.contains(
                    r"\b(?:zxi|vxi|vx|zx|sx|alpha|zeta|titanium|sport|m sport|amg|luxury|prestige|highline|top)\b"
                )
                .iloc[0]
            )
        ),
        "has_condition_note": int(
            bool(
                pd.Series([info])
                .str.contains(
                    r"\b(?:mint|genuine|showroom|well maintained|original|scratchless|excellent)\b"
                )
                .iloc[0]
            )
        ),
        "is_7seater": int(
            bool(
                pd.Series([info])
                .str.contains(r"\b(?:7 seater|7-seater|8 seater|8-seater)\b")
                .iloc[0]
            )
        ),
        "is_luxury": int(brand in LUXURY_BRANDS),
        "model_supply_train": bundle["supply_map"].get((brand, model), 0),
        "Brand": brand,
        "model": model,
        "Transmission": transmission,
        "Owner": owner,
        "FuelType": fuel_type,
    }
])

X_vec = bundle["preprocessor"].transform(row[NUM_COLS + CAT_COLS])
p20 = float(np.expm1(bundle["quantile_models"][0.20].predict(X_vec))[0])
p50 = float(np.expm1(bundle["quantile_models"][0.50].predict(X_vec))[0])
p80 = float(np.expm1(bundle["quantile_models"][0.80].predict(X_vec))[0])

c1, c2, c3 = st.columns(3)
c1.metric("Fast Sale Strategy (q20)", f"₹{p20:,.0f}", "Undercuts ~80% of peers")
c2.metric("Balanced Benchmark (q50)", f"₹{p50:,.0f}", "Market Median Equilibrium")
c3.metric("Maximum Price Limit (q80)", f"₹{p80:,.0f}", "Upper-Quartile Margin")

if "Fast Sale" in strategy:
    st.success(
        f"**Recommended Price (Fast Sale): ₹{p20:,.0f}** — Positioned at the 20th conditional percentile to undercut ~80% of comparable {year} {brand} {model} listings."
    )
elif "Balanced" in strategy:
    st.info(
        f"**Recommended Price (Balanced): ₹{p50:,.0f}** — Aligned with the conditional market median (50th percentile) for fair value and steady turnover."
    )
else:
    st.warning(
        f"**Recommended Price (Maximum Price): ₹{p80:,.0f}** — Targets the 80th percentile upper margin; expect longer holding time unless condition stands out."
    )

if custom_ask and custom_ask > 0:
    diff = ((custom_ask - p50) / p50) * 100
    status = (
        "Aggressive / Fast Sale Zone"
        if custom_ask <= p20
        else (
            "Competitive Market Band"
            if custom_ask <= p80
            else "Above 80th Percentile (High Holding Risk)"
        )
    )
    st.write(
        f"**Your Planned Ask Price (₹{custom_ask:,.0f}):** `{diff:+.1f}%` vs. market median — **{status}**"
    )

pcts = [-15, -10, -5, 0, 5, 10, 15]
rows, prices_lakh = [], []
for pct in pcts:
    cand = p50 * (1 + pct / 100.0)
    prices_lakh.append(cand / 1e5)
    zone = (
        "Fast Sale Zone (≤ q20)"
        if cand <= p20
        else (
            "Balanced Equilibrium"
            if cand <= p50 * 1.03
            else (
                "Upper Margin Zone (≤ q80)"
                if cand <= p80
                else "Overpriced Risk (> q80)"
            )
        )
    )
    rows.append([f"{pct:+d}%", f"₹{cand:,.0f}", zone])

scen_df = pd.DataFrame(
    rows,
    columns=["Price Shift", "Simulated Asking Price", "Market Positioning Zone"],
)

st.subheader("What-If Asking Price Scenario Analysis")
left, right = st.columns([1.2, 1.0])

with left:
    st.dataframe(scen_df, use_container_width=True, hide_index=True)

with right:
    fig, ax = plt.subplots(figsize=(6.5, 3.8), dpi=130)
    colors = [
        "#16a34a" if c * 1e5 <= p20 else ("#2563eb" if c * 1e5 <= p80 else "#dc2626")
        for c in prices_lakh
    ]
    ax.bar([f"{p:+d}%" for p in pcts], prices_lakh, color=colors, width=0.55)
    ax.axhline(
        p20 / 1e5,
        color="#16a34a",
        linestyle="--",
        lw=1.3,
        label=f"Fast Sale q20 (₹{p20/1e5:.2f}L)",
    )
    ax.axhline(
        p50 / 1e5,
        color="#1d4ed8",
        linestyle="-",
        lw=1.5,
        label=f"Balanced q50 (₹{p50/1e5:.2f}L)",
    )
    ax.axhline(
        p80 / 1e5,
        color="#dc2626",
        linestyle="--",
        lw=1.3,
        label=f"Max Price q80 (₹{p80/1e5:.2f}L)",
    )
    if custom_ask and custom_ask > 0:
        ax.axhline(
            custom_ask / 1e5,
            color="#f59e0b",
            linestyle=":",
            lw=2.2,
            label=f"Your Ask (₹{custom_ask/1e5:.2f}L)",
        )
    ax.set_ylabel("Price (INR Lakhs)")
    ax.set_title(f"{year} {brand} {model} — Scenario Band", fontweight="bold")
    ax.legend(fontsize=8, loc="upper left")
    plt.tight_layout()
    st.pyplot(fig)