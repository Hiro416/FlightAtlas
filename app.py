import math
from collections import Counter
from datetime import datetime
from io import BytesIO

import airportsdata
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pyproj import Geod


# =========================================================
# Streamlit config
# =========================================================

st.set_page_config(
    page_title="My Flight Atlas",
    layout="wide",
    page_icon="✈️",
)

st.markdown(
    """
    <style>
    :root {
        --atlas-ink: #172033;
        --atlas-muted: #64748b;
        --atlas-line: #d9e2ec;
        --atlas-panel: #ffffff;
        --atlas-surface: #f6f8fb;
        --atlas-accent: #c7372f;
        --atlas-blue: #1f5f8b;
    }

    .stApp {
        background:
            radial-gradient(circle at top left, rgba(31, 95, 139, 0.10), transparent 30rem),
            linear-gradient(180deg, #f8fbff 0%, #f4f7fb 46%, #ffffff 100%);
        color: var(--atlas-ink);
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1320px;
    }

    [data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid var(--atlas-line);
    }

    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.84);
        border: 1px solid var(--atlas-line);
        border-radius: 8px;
        padding: 1rem 1rem 0.85rem;
        box-shadow: 0 10px 28px rgba(23, 32, 51, 0.06);
    }

    div[data-testid="stMetric"] label {
        color: var(--atlas-muted);
        font-weight: 700;
    }

    .atlas-hero {
        border-bottom: 1px solid var(--atlas-line);
        margin-bottom: 1.25rem;
        padding: 0.75rem 0 1.4rem;
    }

    .atlas-kicker {
        color: var(--atlas-blue);
        font-size: 0.78rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }

    .atlas-title {
        color: var(--atlas-ink);
        font-size: clamp(2.2rem, 5vw, 4.6rem);
        font-weight: 850;
        letter-spacing: 0;
        line-height: 0.98;
        margin: 0.2rem 0 0.75rem;
    }

    .atlas-lead {
        color: #3d4b63;
        font-size: 1.05rem;
        line-height: 1.65;
        max-width: 760px;
        margin: 0;
    }

    .atlas-section-label {
        color: var(--atlas-muted);
        font-size: 0.82rem;
        font-weight: 800;
        letter-spacing: 0.07em;
        margin: 0.25rem 0 0.75rem;
        text-transform: uppercase;
    }

    .atlas-empty {
        background: rgba(255, 255, 255, 0.80);
        border: 1px dashed #b9c7d6;
        border-radius: 8px;
        padding: 1.25rem 1.35rem;
    }

    .atlas-empty strong {
        color: var(--atlas-ink);
        display: block;
        font-size: 1.05rem;
        margin-bottom: 0.35rem;
    }

    .atlas-empty span {
        color: var(--atlas-muted);
    }

    .stDownloadButton button,
    .stButton button {
        border-radius: 8px;
        font-weight: 800;
    }

    div[data-testid="stFileUploader"] > label,
    div[data-testid="stFileUploader"] > label p {
        color: var(--atlas-ink) !important;
        font-weight: 800;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <section class="atlas-hero">
      <div class="atlas-kicker">Flighty CSV mapper</div>
      <h1 class="atlas-title">My Flight Atlas</h1>
      <p class="atlas-lead">
        Flightyから書き出したCSVを読み込み、任意の空港を中心にした正距方位図法の地図へフライトログをプロットします。
        よく使う空港、重複ルート、総移動距離を一画面で確認できます。
      </p>
    </section>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# Support button
# =========================================================

def render_support_button():

    components.html(
        """
        <script
            type="text/javascript"
            src="https://cdnjs.buymeacoffee.com/1.0.0/button.prod.min.js"
            data-name="bmc-button"
            data-slug="hayahiro"
            data-color="#5F7FFF"
            data-emoji=""
            data-font="Lato"
            data-text="Buy me a coffee"
            data-outline-color="#000000"
            data-font-color="#ffffff"
            data-coffee-color="#FFDD00">
        </script>
        """,
        height=80,
    )


# =========================================================
# Constants
# =========================================================

OUTPUT_PREFIX = "my_flight_map_aeqd"

GEOD = Geod(ellps="WGS84")


# =========================================================
# Airport database
# =========================================================

@st.cache_data
def load_airports():

    raw_airports = airportsdata.load("IATA")

    airports = {}

    for iata, row in raw_airports.items():

        if not iata:
            continue

        airports[iata] = {
            "lon": float(row["lon"]),
            "lat": float(row["lat"]),
            "label": row.get("city", ""),
            "name": row.get("name", ""),
            "city": row.get("city", ""),
            "country": row.get("country", ""),
        }

    return airports


AIRPORTS = load_airports()


def airport_option_label(code):

    airport = AIRPORTS.get(code, {})

    city = airport.get("city") or airport.get("label") or ""
    country = airport.get("country") or ""

    detail = " / ".join(
        part for part in (city, country)
        if part
    )

    if detail:
        return f"{code} — {detail}"

    return code


# =========================================================
# Read Flighty CSV
# =========================================================

def read_flights(uploaded_file):

    df = pd.read_csv(uploaded_file)

    flights = []

    for _, row in df.iterrows():

        canceled = str(
            row.get("Canceled", "false")
        ).lower()

        if canceled == "true":
            continue

        dep = str(
            row.get("From", "")
        ).strip().upper()

        arr = str(
            row.get("Diverted To", "")
        ).strip().upper()

        if not arr or arr == "NAN":
            arr = str(
                row.get("To", "")
            ).strip().upper()

        if (
            not dep
            or not arr
            or dep == "NAN"
            or arr == "NAN"
        ):
            continue

        flight = row.to_dict()

        flight["_dep"] = dep
        flight["_arr"] = arr

        flights.append(flight)

    return flights


# =========================================================
# Route points
# =========================================================

def route_points(
    lon1,
    lat1,
    lon2,
    lat2,
    n=90
):

    pts = GEOD.npts(
        lon1,
        lat1,
        lon2,
        lat2,
        n
    )

    lons = [lon1] + [p[0] for p in pts] + [lon2]
    lats = [lat1] + [p[1] for p in pts] + [lat2]

    return lons, lats


# =========================================================
# Build data
# =========================================================

def build_flight_data(flights):

    missing = sorted({
        code
        for fl in flights
        for code in (fl["_dep"], fl["_arr"])
        if code not in AIRPORTS
    })

    valid_flights = [
        fl for fl in flights
        if fl["_dep"] in AIRPORTS
        and fl["_arr"] in AIRPORTS
    ]

    route_counts = Counter(
        (fl["_dep"], fl["_arr"])
        for fl in valid_flights
    )

    airport_counts = Counter()

    airline_counts = Counter()

    years = []

    total_km = 0.0

    for fl in valid_flights:

        dep = fl["_dep"]
        arr = fl["_arr"]

        airport_counts[dep] += 1
        airport_counts[arr] += 1

        airline_counts[
            str(fl.get("Airline", "Unknown"))
        ] += 1

        lon1 = AIRPORTS[dep]["lon"]
        lat1 = AIRPORTS[dep]["lat"]

        lon2 = AIRPORTS[arr]["lon"]
        lat2 = AIRPORTS[arr]["lat"]

        _, _, dist_m = GEOD.inv(
            lon1,
            lat1,
            lon2,
            lat2
        )

        total_km += dist_m / 1000

        try:

            years.append(
                datetime.strptime(
                    str(fl["Date"]),
                    "%Y-%m-%d"
                ).year
            )

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


# =========================================================
# Plot
# =========================================================

def plot_flight_map(
    data,
    center="SDJ",
    figsize=(24, 24),
    range_m=14_500_000,
):

    flights = data["flights"]
    route_counts = data["route_counts"]
    airport_counts = data["airport_counts"]

    years = data["years"]
    total_km = data["total_km"]

    center_lon = AIRPORTS[center]["lon"]
    center_lat = AIRPORTS[center]["lat"]
    center_label = AIRPORTS[center]["label"]

    years_text = (
        f"{min(years)}–{max(years)}"
        if years
        else ""
    )

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

    ax.add_feature(
        cfeature.OCEAN.with_scale("50m"),
        facecolor="#eef6fb",
        zorder=0
    )

    ax.add_feature(
        cfeature.LAND.with_scale("50m"),
        facecolor="#f4efe6",
        edgecolor="none",
        zorder=1
    )

    ax.add_feature(
        cfeature.LAKES.with_scale("50m"),
        facecolor="#eef6fb",
        edgecolor="#bbbbbb",
        linewidth=0.25,
        zorder=2
    )

    ax.add_feature(
        cfeature.COASTLINE.with_scale("50m"),
        linewidth=0.35,
        edgecolor="#555555",
        zorder=2
    )

    ax.add_feature(
        cfeature.BORDERS.with_scale("50m"),
        linewidth=0.25,
        edgecolor="#777777",
        zorder=2
    )

    ax.gridlines(
        draw_labels=False,
        linewidth=0.25,
        color="#aaaaaa",
        alpha=0.35,
        linestyle="--",
        zorder=2
    )

    max_count = (
        max(route_counts.values())
        if route_counts
        else 1
    )

    for (dep, arr), count in route_counts.items():

        lon1 = AIRPORTS[dep]["lon"]
        lat1 = AIRPORTS[dep]["lat"]

        lon2 = AIRPORTS[arr]["lon"]
        lat2 = AIRPORTS[arr]["lat"]

        lons, lats = route_points(
            lon1,
            lat1,
            lon2,
            lat2,
            n=90
        )

        linewidth = (
            0.35
            + 3.0 * (count / max_count) ** 0.6
        )

        alpha = min(
            0.18
            + 0.55 * (count / max_count) ** 0.45,
            0.82
        )

        ax.plot(
            lons,
            lats,
            transform=ccrs.Geodetic(),
            linewidth=linewidth,
            alpha=alpha,
            color="#c7372f",
            zorder=3
        )

    max_airport_count = (
        max(airport_counts.values())
        if airport_counts
        else 1
    )

    for code, count in airport_counts.items():

        lon = AIRPORTS[code]["lon"]
        lat = AIRPORTS[code]["lat"]

        size = (
            20
            + 165
            * (count / max_airport_count) ** 0.55
        )

        ax.scatter(
            lon,
            lat,
            s=size,
            color="#1f2a44",
            edgecolor="white",
            linewidth=0.7,
            transform=ccrs.PlateCarree(),
            zorder=5
        )

    label_codes = (
        {
            code
            for code, _
            in airport_counts.most_common(35)
        }
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

    title = (
        f"My Flight Map — "
        f"Azimuthal Equidistant centered on {center}"
    )

    subtitle = (
        f"{len(flights):,} flights  |  "
        f"{len(airport_counts):,} airports  |  "
        f"{len(route_counts):,} routes  |  "
        f"approx. {total_km:,.0f} km  |  "
        f"{years_text}"
    )

    ax.set_title(
        title + "\n" + subtitle,
        fontsize=28,
        pad=28,
        weight="bold"
    )

    legend = [
        Line2D(
            [0], [0],
            color="#c7372f",
            lw=2.5,
            label="Flight route; thicker = more flights"
        ),
        Line2D(
            [0], [0],
            marker="o",
            color="w",
            markerfacecolor="#1f2a44",
            markeredgecolor="white",
            markersize=10,
            label="Airport; larger = more use"
        ),
        Line2D(
            [0], [0],
            marker="*",
            color="w",
            markerfacecolor="#ffd166",
            markeredgecolor="#111111",
            markersize=18,
            label="Map center"
        ),
    ]

    ax.legend(
        handles=legend,
        loc="lower left",
        fontsize=14,
        frameon=True,
        framealpha=0.9
    )

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


# =========================================================
# Figure export
# =========================================================

def fig_to_bytes(
    fig,
    filetype="png"
):

    buffer = BytesIO()

    if filetype == "png":

        fig.savefig(
            buffer,
            format="png",
            dpi=300,
            bbox_inches="tight"
        )

    else:

        fig.savefig(
            buffer,
            format=filetype,
            bbox_inches="tight"
        )

    buffer.seek(0)

    return buffer


# =========================================================
# Upload
# =========================================================

uploaded = st.file_uploader(
    "Flighty CSVをアップロード",
    type=["csv"],
    help="FlightyのエクスポートCSVをそのまま指定してください。",
)

if uploaded is None:

    st.markdown(
        """
        <div class="atlas-empty">
          <strong>CSVをアップロードすると、地図と集計がここに表示されます。</strong>
          <span>Flightyのフライトログを書き出し、この欄へCSVファイルを追加してください。</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.stop()


# =========================================================
# Build
# =========================================================

flights = read_flights(uploaded)
with st.spinner("フライトログを解析しています..."):
    data = build_flight_data(flights)

if data["missing"]:

    st.warning(
        "座標が見つからなかった空港: "
        + ", ".join(data["missing"])
    )

if not data["flights"]:

    st.error(
        "有効なフライトがありません。"
    )

    st.stop()


# =========================================================
# Sidebar
# =========================================================

available_airports = sorted(
    data["airport_counts"].keys()
)

default_center = (
    "SDJ"
    if "SDJ" in available_airports
    else available_airports[0]
)

with st.sidebar:

    st.header("Map settings")

    center_airport = st.selectbox(
        "Center airport",
        available_airports,
        index=available_airports.index(
            default_center
        ),
        format_func=airport_option_label,
    )

    range_km = st.slider(
        "Map radius",
        min_value=8_000,
        max_value=20_000,
        value=14_500,
        step=500,
        format="%d km",
    )

    range_m = range_km * 1000

    st.caption(
        f"{len(AIRPORTS):,} airports loaded from the local database."
    )

    st.divider()

    st.subheader("Export")

    st.caption(
        "地図の生成後にPNG、PDF、SVGを書き出せます。"
    )

    st.divider()

    st.subheader("Support")

    render_support_button()


# =========================================================
# Metrics
# =========================================================

years_text = (
    f"{min(data['years'])}–{max(data['years'])}"
    if data["years"]
    else "N/A"
)

st.markdown(
    '<div class="atlas-section-label">Overview</div>',
    unsafe_allow_html=True,
)

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric(
    "Flights",
    f"{len(data['flights']):,}"
)

col2.metric(
    "Airports",
    f"{len(data['airport_counts']):,}"
)

col3.metric(
    "Routes",
    f"{len(data['route_counts']):,}"
)

col4.metric(
    "Distance",
    f"{data['total_km']:,.0f} km"
)

col5.metric(
    "Years",
    years_text
)


# =========================================================
# Plot
# =========================================================

map_tab, data_tab = st.tabs(
    ["Map", "Rankings"]
)

with map_tab:

    st.markdown(
        '<div class="atlas-section-label">Azimuthal equidistant map</div>',
        unsafe_allow_html=True,
    )

    with st.spinner("地図を描画しています..."):
        fig = plot_flight_map(
            data,
            center=center_airport,
            figsize=(24, 24),
            range_m=range_m,
        )

    st.pyplot(
        fig,
        use_container_width=True
    )


# =========================================================
# Downloads
# =========================================================

    png_bytes = fig_to_bytes(fig, "png")
    pdf_bytes = fig_to_bytes(fig, "pdf")
    svg_bytes = fig_to_bytes(fig, "svg")

    dl1, dl2, dl3 = st.columns(3)

    dl1.download_button(
        "Download PNG",
        data=png_bytes,
        file_name=f"{OUTPUT_PREFIX}_{center_airport}.png",
        mime="image/png",
        use_container_width=True,
    )

    dl2.download_button(
        "Download PDF",
        data=pdf_bytes,
        file_name=f"{OUTPUT_PREFIX}_{center_airport}.pdf",
        mime="application/pdf",
        use_container_width=True,
    )

    dl3.download_button(
        "Download SVG",
        data=svg_bytes,
        file_name=f"{OUTPUT_PREFIX}_{center_airport}.svg",
        mime="image/svg+xml",
        use_container_width=True,
    )

with data_tab:

    left, right = st.columns(2)

    airport_rank = pd.DataFrame(
        [
            {
                "Airport": code,
                "City": AIRPORTS[code].get("city", ""),
                "Country": AIRPORTS[code].get("country", ""),
                "Flights": count,
            }
            for code, count
            in data["airport_counts"].most_common()
        ]
    )

    route_rank = pd.DataFrame(
        [
            {
                "Route": f"{a}-{b}",
                "From": airport_option_label(a),
                "To": airport_option_label(b),
                "Flights": c
            }
            for (a, b), c
            in data["route_counts"].most_common()
        ]
    )

    airline_rank = pd.DataFrame(
        [
            {
                "Airline": airline,
                "Flights": count,
            }
            for airline, count
            in data["airline_counts"].most_common()
        ]
    )

    with left:

        st.subheader("Top Airports")

        st.dataframe(
            airport_rank,
            use_container_width=True,
            hide_index=True,
        )

    with right:

        st.subheader("Top Routes")

        st.dataframe(
            route_rank,
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("Top Airlines")

    st.dataframe(
        airline_rank,
        use_container_width=True,
        hide_index=True,
    )
