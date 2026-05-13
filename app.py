import math
from collections import Counter
from datetime import datetime
from io import BytesIO

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pyproj import Geod


st.set_page_config(
    page_title="My Flight Atlas",
    layout="wide"
)

st.title("My Flight Atlas")
st.caption("Flighty CSVから正距方位図法のフライトマップを生成します。")


OUTPUT_PREFIX = "my_flight_map_aeqd"

AIRPORT_CSV_URL = (
    "https://raw.githubusercontent.com/"
    "mborsetti/airportsdata/refs/heads/main/airportsdata/airports.csv"
)

GEOD = Geod(ellps="WGS84")


@st.cache_data
def load_airports():
    airports_df = pd.read_csv(AIRPORT_CSV_URL)
    airports_df = airports_df.dropna(subset=["iata", "lat", "lon"])

    airports = {}

    for _, row in airports_df.iterrows():
        iata = str(row["iata"]).strip().upper()

        if iata and iata != "NAN":
            label = str(row.get("city", ""))

            if label == "nan" or label == "":
                label = str(row.get("name", ""))

            airports[iata] = {
                "lon": float(row["lon"]),
                "lat": float(row["lat"]),
                "label": label,
                "name": str(row.get("name", "")),
                "city": str(row.get("city", "")),
                "country": str(row.get("country", "")),
            }

    return airports


AIRPORTS = load_airports()


def read_flights(uploaded_file):
    df = pd.read_csv(uploaded_file)

    flights = []

    for _, row in df.iterrows():
        canceled = str(row.get("Canceled", "false")).lower()

        if canceled == "true":
            continue

        dep = str(row.get("From", "")).strip().upper()
        arr = str(row.get("Diverted To", "")).strip().upper()

        if not arr or arr == "NAN":
            arr = str(row.get("To", "")).strip().upper()

        if not dep or not arr or dep == "NAN" or arr == "NAN":
            continue

        flight = row.to_dict()
        flight["_dep"] = dep
        flight["_arr"] = arr

        flights.append(flight)

    return flights


def route_points(lon1, lat1, lon2, lat2, n=90):
    pts = GEOD.npts(lon1, lat1, lon2, lat2, n)
    lons = [lon1] + [p[0] for p in pts] + [lon2]
    lats = [lat1] + [p[1] for p in pts] + [lat2]
    return lons, lats


def build_flight_data(flights):
    missing = sorted({
        code
        for fl in flights
        for code in (fl["_dep"], fl["_arr"])
        if code not in AIRPORTS
    })

    valid_flights = [
        fl for fl in flights
        if fl["_dep"] in AIRPORTS and fl["_arr"] in AIRPORTS
    ]

    route_counts = Counter((fl["_dep"], fl["_arr"]) for fl in valid_flights)
    airport_counts = Counter()
    airline_counts = Counter()

    years = []
    total_km = 0.0

    for fl in valid_flights:
        dep = fl["_dep"]
        arr = fl["_arr"]

        airport_counts[dep] += 1
        airport_counts[arr] += 1
        airline_counts[str(fl.get("Airline", "Unknown"))] += 1

        lon1 = AIRPORTS[dep]["lon"]
        lat1 = AIRPORTS[dep]["lat"]
        lon2 = AIRPORTS[arr]["lon"]
        lat2 = AIRPORTS[arr]["lat"]

        _, _, dist_m = GEOD.inv(lon1, lat1, lon2, lat2)
        total_km += dist_m / 1000

        try:
            years.append(datetime.strptime(str(fl["Date"]), "%Y-%m-%d").year)
        except Exception:
            pass

    return {
        "missing": missing,
        "flights": valid_flights,
        "route_counts": route_counts,
        "airport_counts": airport_counts,
        "airline_counts": airline_counts,
        "years": years,
        "total_km": total_km,
    }


def plot_flight_map(data, center="SDJ", figsize=(24, 24), range_m=14_500_000):
    flights = data["flights"]
    route_counts = data["route_counts"]
    airport_counts = data["airport_counts"]
    years = data["years"]
    total_km = data["total_km"]

    center_lon = AIRPORTS[center]["lon"]
    center_lat = AIRPORTS[center]["lat"]
    center_label = AIRPORTS[center]["label"]

    years_text = f"{min(years)}–{max(years)}" if years else ""

    proj = ccrs.AzimuthalEquidistant(
        central_longitude=center_lon,
        central_latitude=center_lat,
    )

    fig = plt.figure(figsize=figsize)
    ax = plt.axes(projection=proj)

    fig.patch.set_facecolor("white")
    ax.set_facecolor("#eef6fb")

    ax.set_xlim(-range_m, range_m)
    ax.set_ylim(-range_m, range_m)

    ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="#eef6fb", zorder=0)
    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#f4efe6", edgecolor="none", zorder=1)
    ax.add_feature(cfeature.LAKES.with_scale("50m"), facecolor="#eef6fb", edgecolor="#bbbbbb", linewidth=0.25, zorder=2)
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.35, edgecolor="#555555", zorder=2)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.25, edgecolor="#777777", zorder=2)

    ax.gridlines(
        draw_labels=False,
        linewidth=0.25,
        color="#aaaaaa",
        alpha=0.35,
        linestyle="--",
        zorder=2,
    )

    max_count = max(route_counts.values()) if route_counts else 1

    for (dep, arr), count in route_counts.items():
        lon1 = AIRPORTS[dep]["lon"]
        lat1 = AIRPORTS[dep]["lat"]
        lon2 = AIRPORTS[arr]["lon"]
        lat2 = AIRPORTS[arr]["lat"]

        lons, lats = route_points(lon1, lat1, lon2, lat2, n=90)

        linewidth = 0.35 + 3.0 * (count / max_count) ** 0.6
        alpha = min(0.18 + 0.55 * (count / max_count) ** 0.45, 0.82)

        ax.plot(
            lons,
            lats,
            transform=ccrs.Geodetic(),
            linewidth=linewidth,
            alpha=alpha,
            color="#c7372f",
            zorder=3,
        )

    max_airport_count = max(airport_counts.values()) if airport_counts else 1

    for code, count in airport_counts.items():
        lon = AIRPORTS[code]["lon"]
        lat = AIRPORTS[code]["lat"]

        size = 20 + 165 * (count / max_airport_count) ** 0.55

        ax.scatter(
            lon,
            lat,
            s=size,
            color="#1f2a44",
            edgecolor="white",
            linewidth=0.7,
            transform=ccrs.PlateCarree(),
            zorder=5,
        )

    label_codes = (
        {code for code, _ in airport_counts.most_common(35)}
        | {
            center,
            "FAI", "ANC", "TPA", "JFK",
            "EWR", "DEL", "DOH", "HEL",
            "TPE", "HKG", "SEA", "SFO",
            "LAX", "SYD", "CDG",
        }
    )

    for code in label_codes:
        if code not in airport_counts:
            continue

        lon = AIRPORTS[code]["lon"]
        lat = AIRPORTS[code]["lat"]

        ax.text(
            lon,
            lat,
            f" {code}",
            fontsize=12,
            weight="bold",
            color="#111111",
            transform=ccrs.PlateCarree(),
            zorder=6,
        )

    ax.scatter(
        center_lon,
        center_lat,
        s=620,
        marker="*",
        color="#ffd166",
        edgecolor="#111111",
        linewidth=1.3,
        transform=ccrs.PlateCarree(),
        zorder=8,
    )

    ax.text(
        center_lon,
        center_lat,
        f"  {center} / {center_label}",
        fontsize=20,
        weight="bold",
        color="#111111",
        transform=ccrs.PlateCarree(),
        zorder=9,
    )

    title = f"My Flight Map — Azimuthal Equidistant centered on {center}"

    subtitle = (
        f"{len(flights):,} flights  |  "
        f"{len(airport_counts):,} airports  |  "
        f"{len(route_counts):,} routes  |  "
        f"approx. {total_km:,.0f} km  |  "
        f"{years_text}"
    )

    ax.set_title(title + "\n" + subtitle, fontsize=28, pad=28, weight="bold")

    legend = [
        Line2D([0], [0], color="#c7372f", lw=2.5, label="Flight route; thicker = more flights"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#1f2a44", markeredgecolor="white", markersize=10, label="Airport; larger = more use"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="#ffd166", markeredgecolor="#111111", markersize=18, label="Map center"),
    ]

    ax.legend(handles=legend, loc="lower left", fontsize=14, frameon=True, framealpha=0.9)

    ax.text(
        0.99,
        0.01,
        "Data: Flighty CSV export",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        color="#555555",
    )

    plt.subplots_adjust(top=0.90)

    return fig


def fig_to_bytes(fig, filetype="png"):
    buffer = BytesIO()

    if filetype == "png":
        fig.savefig(buffer, format="png", dpi=300, bbox_inches="tight")
    else:
        fig.savefig(buffer, format=filetype, bbox_inches="tight")

    buffer.seek(0)
    return buffer


uploaded = st.file_uploader("Flighty CSVをアップロード", type=["csv"])

if uploaded is None:
    st.info("FlightyのCSVを書き出してアップロードしてください。")
    st.stop()

flights = read_flights(uploaded)
data = build_flight_data(flights)

if data["missing"]:
    st.warning("座標が見つからなかった空港: " + ", ".join(data["missing"]))

if not data["flights"]:
    st.error("有効なフライトがありません。CSV形式を確認してください。")
    st.stop()

available_airports = sorted(data["airport_counts"].keys())

default_center = "SDJ" if "SDJ" in available_airports else available_airports[0]

with st.sidebar:
    st.header("Settings")

    center_airport = st.selectbox(
        "Center Airport",
        available_airports,
        index=available_airports.index(default_center),
    )

    range_m = st.slider(
        "Map range",
        min_value=8_000_000,
        max_value=20_000_000,
        value=14_500_000,
        step=500_000,
    )

    st.caption(f"{len(AIRPORTS):,} airports loaded")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Flights", f"{len(data['flights']):,}")
col2.metric("Airports", f"{len(data['airport_counts']):,}")
col3.metric("Routes", f"{len(data['route_counts']):,}")
col4.metric("Distance", f"{data['total_km']:,.0f} km")

st.divider()

fig = plot_flight_map(
    data,
    center=center_airport,
    figsize=(24, 24),
    range_m=range_m,
)

st.pyplot(fig, use_container_width=True)

png_bytes = fig_to_bytes(fig, "png")
pdf_bytes = fig_to_bytes(fig, "pdf")
svg_bytes = fig_to_bytes(fig, "svg")

st.download_button(
    "Download PNG",
    data=png_bytes,
    file_name=f"{OUTPUT_PREFIX}_{center_airport}.png",
    mime="image/png",
)

st.download_button(
    "Download PDF",
    data=pdf_bytes,
    file_name=f"{OUTPUT_PREFIX}_{center_airport}.pdf",
    mime="application/pdf",
)

st.download_button(
    "Download SVG",
    data=svg_bytes,
    file_name=f"{OUTPUT_PREFIX}_{center_airport}.svg",
    mime="image/svg+xml",
)

left, right = st.columns(2)

with left:
    st.subheader("Top Airports")
    airport_rank = pd.DataFrame(
        data["airport_counts"].most_common(),
        columns=["Airport", "Count"],
    )
    st.dataframe(airport_rank, use_container_width=True)

with right:
    st.subheader("Top Routes")
    route_rank = pd.DataFrame(
        [
            {"Route": f"{a}-{b}", "Count": c}
            for (a, b), c in data["route_counts"].most_common()
        ]
    )
    st.dataframe(route_rank, use_container_width=True)
