const EARTH_RADIUS_KM = 6371.0088;
const OUTPUT_PREFIX = "my_flight_map_aeqd_static";
const SVG_NS = "http://www.w3.org/2000/svg";

const els = {
  file: document.querySelector("#csv-file"),
  sample: document.querySelector("#load-sample"),
  note: document.querySelector("#file-note"),
  empty: document.querySelector("#empty-state"),
  app: document.querySelector("#app"),
  center: document.querySelector("#center-airport"),
  radius: document.querySelector("#map-radius"),
  radiusValue: document.querySelector("#map-radius-value"),
  missing: document.querySelector("#missing-airports"),
  svg: document.querySelector("#flight-map"),
  metrics: {
    flights: document.querySelector("#metric-flights"),
    airports: document.querySelector("#metric-airports"),
    routes: document.querySelector("#metric-routes"),
    distance: document.querySelector("#metric-distance"),
    years: document.querySelector("#metric-years"),
  },
  airportRank: document.querySelector("#airport-rank"),
  routeRank: document.querySelector("#route-rank"),
  airlineRank: document.querySelector("#airline-rank"),
  downloadSvg: document.querySelector("#download-svg"),
  downloadPng: document.querySelector("#download-png"),
};

let airports = {};
let worldLand = [];
let state = null;

init();

async function init() {
  const [airportData, landData] = await Promise.all([
    fetch("static/airports.json").then((res) => res.json()),
    fetch("static/world-land.json").then((res) => res.json()),
  ]);
  airports = airportData;
  worldLand = landData.polygons ?? [];
  els.file.addEventListener("change", handleFile);
  els.sample.addEventListener("click", handleSample);
  els.center.addEventListener("change", renderAll);
  els.radius.addEventListener("input", renderAll);
  els.downloadSvg.addEventListener("click", downloadSvg);
  els.downloadPng.addEventListener("click", downloadPng);
}

async function handleFile(event) {
  const file = event.target.files[0];
  if (!file) return;

  const text = await file.text();
  loadCsvText(text);
}

async function handleSample() {
  const text = await fetch("static/sample-flighty.csv").then((res) => res.text());
  loadCsvText(text);
}

function loadCsvText(text) {
  const rows = parseCsv(text);
  const flights = readFlights(rows);
  const data = buildFlightData(flights);

  if (!data.flights.length) {
    els.note.textContent = "有効なフライトが見つかりませんでした。";
    return;
  }

  state = data;
  els.empty.hidden = true;
  els.app.hidden = false;
  populateCenters(data);
  renderAll();
}

function parseCsv(text) {
  const rows = [];
  let row = [];
  let field = "";
  let inQuotes = false;

  for (let i = 0; i < text.length; i += 1) {
    const char = text[i];
    const next = text[i + 1];

    if (char === '"' && inQuotes && next === '"') {
      field += '"';
      i += 1;
    } else if (char === '"') {
      inQuotes = !inQuotes;
    } else if (char === "," && !inQuotes) {
      row.push(field);
      field = "";
    } else if ((char === "\n" || char === "\r") && !inQuotes) {
      if (char === "\r" && next === "\n") i += 1;
      row.push(field);
      if (row.some((value) => value.length)) rows.push(row);
      row = [];
      field = "";
    } else {
      field += char;
    }
  }

  row.push(field);
  if (row.some((value) => value.length)) rows.push(row);

  const headers = rows.shift()?.map((header) => header.trim()) ?? [];
  return rows.map((values) => Object.fromEntries(headers.map((header, index) => [header, values[index] ?? ""])));
}

function readFlights(rows) {
  return rows.flatMap((row) => {
    if (String(row.Canceled ?? "false").toLowerCase() === "true") return [];

    const dep = cleanCode(row.From);
    const diverted = cleanCode(row["Diverted To"]);
    const arr = diverted || cleanCode(row.To);

    if (!dep || !arr) return [];
    return [{ ...row, _dep: dep, _arr: arr }];
  });
}

function cleanCode(value) {
  const code = String(value ?? "").trim().toUpperCase();
  return code && code !== "NAN" ? code : "";
}

function buildFlightData(flights) {
  const missing = [...new Set(flights.flatMap((flight) => [flight._dep, flight._arr]).filter((code) => !airports[code]))].sort();
  const validFlights = flights.filter((flight) => airports[flight._dep] && airports[flight._arr]);
  const routeCounts = new Map();
  const airportCounts = new Map();
  const airlineCounts = new Map();
  const years = [];
  let totalKm = 0;

  for (const flight of validFlights) {
    increment(airportCounts, flight._dep);
    increment(airportCounts, flight._arr);
    increment(airlineCounts, String(flight.Airline || "Unknown"));
    increment(routeCounts, `${flight._dep}|${flight._arr}`);

    const dep = airports[flight._dep];
    const arr = airports[flight._arr];
    totalKm += distanceKm(dep.lon, dep.lat, arr.lon, arr.lat);

    const year = Number.parseInt(String(flight.Date ?? "").slice(0, 4), 10);
    if (Number.isFinite(year)) years.push(year);
  }

  return { missing, flights: validFlights, routeCounts, airportCounts, airlineCounts, years, totalKm };
}

function increment(map, key) {
  map.set(key, (map.get(key) ?? 0) + 1);
}

function populateCenters(data) {
  const codes = [...data.airportCounts.keys()].sort();
  const defaultCode = codes.includes("SDJ") ? "SDJ" : codes[0];
  els.center.replaceChildren(
    ...codes.map((code) => {
      const option = document.createElement("option");
      option.value = code;
      option.textContent = airportOptionLabel(code);
      option.selected = code === defaultCode;
      return option;
    }),
  );
}

function renderAll() {
  if (!state) return;
  els.radiusValue.textContent = `${number(els.radius.value)} km`;
  renderMetrics();
  renderWarnings();
  renderRankings();
  renderMap();
}

function renderMetrics() {
  const { flights, airportCounts, routeCounts, years, totalKm } = state;
  els.metrics.flights.textContent = number(flights.length);
  els.metrics.airports.textContent = number(airportCounts.size);
  els.metrics.routes.textContent = number(routeCounts.size);
  els.metrics.distance.textContent = `${number(Math.round(totalKm))} km`;
  els.metrics.years.textContent = years.length ? `${Math.min(...years)}-${Math.max(...years)}` : "N/A";
}

function renderWarnings() {
  if (!state.missing.length) {
    els.missing.hidden = true;
    return;
  }

  els.missing.hidden = false;
  els.missing.textContent = `座標が見つからなかった空港: ${state.missing.join(", ")}`;
}

function renderRankings() {
  renderAirportRank();
  renderRouteRank();
  renderAirlineRank();
}

function renderAirportRank() {
  const rows = sortedEntries(state.airportCounts).map(([code, count]) => {
    const airport = airports[code];
    return [code, airport.city || "", airport.country || "", number(count)];
  });
  renderRows(els.airportRank, rows);
}

function renderRouteRank() {
  const rows = sortedEntries(state.routeCounts).map(([route, count]) => {
    const [dep, arr] = route.split("|");
    return [`${dep}-${arr}`, airportOptionLabel(dep), airportOptionLabel(arr), number(count)];
  });
  renderRows(els.routeRank, rows);
}

function renderAirlineRank() {
  const rows = sortedEntries(state.airlineCounts).map(([airline, count]) => [airline, number(count)]);
  renderRows(els.airlineRank, rows);
}

function renderRows(tbody, rows) {
  tbody.replaceChildren(
    ...rows.map((cells) => {
      const tr = document.createElement("tr");
      tr.replaceChildren(...cells.map((cell) => {
        const td = document.createElement("td");
        td.textContent = cell;
        return td;
      }));
      return tr;
    }),
  );
}

function renderMap() {
  const size = 1200;
  const pad = 76;
  const radiusPx = size / 2 - pad;
  const centerCode = els.center.value;
  const center = airports[centerCode];
  const rangeKm = Number(els.radius.value);
  const rangeRad = rangeKm / EARTH_RADIUS_KM;
  const project = makeAzimuthalProjector(center.lon, center.lat, size / 2, size / 2, radiusPx / rangeRad);

  els.svg.setAttribute("viewBox", `0 0 ${size} ${size}`);
  els.svg.replaceChildren();

  append("rect", { x: 0, y: 0, width: size, height: size, fill: "#ffffff" });
  const defs = append("defs");
  const clip = document.createElementNS(SVG_NS, "clipPath");
  clip.setAttribute("id", "map-clip");
  const clipCircle = document.createElementNS(SVG_NS, "circle");
  clipCircle.setAttribute("cx", size / 2);
  clipCircle.setAttribute("cy", size / 2);
  clipCircle.setAttribute("r", radiusPx);
  clip.appendChild(clipCircle);
  defs.appendChild(clip);

  append("circle", { cx: size / 2, cy: size / 2, r: radiusPx, fill: "#eef6fb", stroke: "#9fb4c8", "stroke-width": 1.2 });
  drawWorldLand(project, rangeRad);
  drawGraticule(project, rangeRad);
  drawRoutes(project, rangeRad);
  drawAirports(project, rangeRad);
  drawLabels(project, rangeRad, centerCode);
  drawTitle(size, centerCode, center);
}

function drawWorldLand(project, rangeRad) {
  const group = append("g", {
    "clip-path": "url(#map-clip)",
    fill: "#f4efe6",
    stroke: "#d0c7b7",
    "stroke-width": 0.8,
  });

  for (const polygon of worldLand) {
    const path = projectedPath(polygon, project, Infinity, true);
    if (path) appendPath(group, path);
  }
}

function drawGraticule(project, rangeRad) {
  const group = append("g", { fill: "none", stroke: "#9aa9b7", "stroke-width": 0.75, "stroke-dasharray": "5 8", opacity: 0.42 });

  for (let lat = -60; lat <= 60; lat += 30) {
    const points = [];
    for (let lon = -180; lon <= 180; lon += 4) points.push([lon, lat]);
    appendPath(group, projectedPath(points, project, rangeRad));
  }

  for (let lon = -180; lon < 180; lon += 30) {
    const points = [];
    for (let lat = -88; lat <= 88; lat += 4) points.push([lon, lat]);
    appendPath(group, projectedPath(points, project, rangeRad));
  }
}

function drawRoutes(project, rangeRad) {
  const maxCount = Math.max(1, ...state.routeCounts.values());
  const group = append("g", { fill: "none", stroke: "#c7372f", "stroke-linecap": "round" });

  for (const [route, count] of state.routeCounts) {
    const [depCode, arrCode] = route.split("|");
    const dep = airports[depCode];
    const arr = airports[arrCode];
    const points = greatCirclePoints(dep.lon, dep.lat, arr.lon, arr.lat, 90);
    const path = projectedPath(points, project, rangeRad);
    if (!path) continue;

    appendPath(group, path, {
      "stroke-width": 0.35 + 3.0 * (count / maxCount) ** 0.6,
      opacity: Math.min(0.18 + 0.55 * (count / maxCount) ** 0.45, 0.82),
    });
  }
}

function drawAirports(project, rangeRad) {
  const maxCount = Math.max(1, ...state.airportCounts.values());

  for (const [code, count] of state.airportCounts) {
    const airport = airports[code];
    const point = project(airport.lon, airport.lat);
    if (!point || point.c > rangeRad) continue;

    append("circle", {
      cx: point.x,
      cy: point.y,
      r: 3 + 9 * (count / maxCount) ** 0.55,
      fill: "#1f2a44",
      stroke: "#ffffff",
      "stroke-width": 1.4,
    });
  }
}

function drawLabels(project, rangeRad, centerCode) {
  const topCodes = sortedEntries(state.airportCounts).slice(0, 35).map(([code]) => code);
  const fixedCodes = ["FAI", "ANC", "TPA", "JFK", "EWR", "DEL", "DOH", "HEL", "TPE", "HKG", "SEA", "SFO", "LAX", "SYD", "CDG"];
  const codes = new Set([centerCode, ...topCodes, ...fixedCodes]);

  for (const code of codes) {
    if (!state.airportCounts.has(code)) continue;
    const airport = airports[code];
    const point = project(airport.lon, airport.lat);
    if (!point || point.c > rangeRad) continue;

    append("text", {
      x: point.x + 7,
      y: point.y - 7,
      fill: "#111111",
      "font-size": code === centerCode ? 19 : 13,
      "font-weight": 800,
    }, `${code}${code === centerCode ? ` / ${airport.label || airport.city || ""}` : ""}`);
  }

  const center = airports[centerCode];
  const point = project(center.lon, center.lat);
  append("path", {
    d: starPath(point.x, point.y, 17, 7, 5),
    fill: "#ffd166",
    stroke: "#111111",
    "stroke-width": 1.4,
  });
}

function drawTitle(size, centerCode) {
  const years = state.years.length ? `${Math.min(...state.years)}-${Math.max(...state.years)}` : "";
  append("text", {
    x: size / 2,
    y: 38,
    "text-anchor": "middle",
    fill: "#172033",
    "font-size": 30,
    "font-weight": 850,
  }, "My Flight Map");
  append("text", {
    x: size / 2,
    y: 68,
    "text-anchor": "middle",
    fill: "#3d4b63",
    "font-size": 16,
    "font-weight": 700,
  }, `Azimuthal equidistant centered on ${centerCode}`);
  append("text", {
    x: size / 2,
    y: 92,
    "text-anchor": "middle",
    fill: "#3d4b63",
    "font-size": 15,
    "font-weight": 700,
  }, `${number(state.flights.length)} flights | ${number(state.airportCounts.size)} airports | ${number(state.routeCounts.size)} routes | approx. ${number(Math.round(state.totalKm))} km | ${years}`);
  append("text", {
    x: size - 22,
    y: size - 20,
    "text-anchor": "end",
    fill: "#555555",
    "font-size": 12,
  }, "Data: Flighty CSV export");
}

function makeAzimuthalProjector(centerLon, centerLat, originX, originY, scale) {
  const lon0 = radians(centerLon);
  const lat0 = radians(centerLat);
  const sinLat0 = Math.sin(lat0);
  const cosLat0 = Math.cos(lat0);

  return (lon, lat) => {
    const lambda = radians(lon);
    const phi = radians(lat);
    const dlon = normalizeRadians(lambda - lon0);
    const sinPhi = Math.sin(phi);
    const cosPhi = Math.cos(phi);
    const cosC = clamp(sinLat0 * sinPhi + cosLat0 * cosPhi * Math.cos(dlon), -1, 1);
    const c = Math.acos(cosC);
    const k = Math.abs(c) < 1e-9 ? 1 : c / Math.sin(c);

    return {
      x: originX + scale * k * cosPhi * Math.sin(dlon),
      y: originY - scale * k * (cosLat0 * sinPhi - sinLat0 * cosPhi * Math.cos(dlon)),
      c,
    };
  };
}

function projectedPath(points, project, rangeRad, closePath = false) {
  const commands = [];
  let drawing = false;

  for (const [lon, lat] of points) {
    const point = project(lon, lat);
    if (!point || !Number.isFinite(point.x) || !Number.isFinite(point.y) || point.c > rangeRad) {
      drawing = false;
      continue;
    }
    commands.push(`${drawing ? "L" : "M"} ${round(point.x)} ${round(point.y)}`);
    drawing = true;
  }

  if (commands.length <= 1) return "";
  return `${commands.join(" ")}${closePath ? " Z" : ""}`;
}

function greatCirclePoints(lon1, lat1, lon2, lat2, steps) {
  const start = toCartesian(lon1, lat1);
  const end = toCartesian(lon2, lat2);
  const omega = Math.acos(clamp(dot(start, end), -1, 1));
  const sinOmega = Math.sin(omega);
  const points = [];

  for (let i = 0; i <= steps + 1; i += 1) {
    const t = i / (steps + 1);
    let v;
    if (Math.abs(sinOmega) < 1e-9) {
      v = start;
    } else {
      const a = Math.sin((1 - t) * omega) / sinOmega;
      const b = Math.sin(t * omega) / sinOmega;
      v = [a * start[0] + b * end[0], a * start[1] + b * end[1], a * start[2] + b * end[2]];
    }
    points.push(fromCartesian(v));
  }

  return points;
}

function distanceKm(lon1, lat1, lon2, lat2) {
  const phi1 = radians(lat1);
  const phi2 = radians(lat2);
  const dphi = radians(lat2 - lat1);
  const dlambda = radians(lon2 - lon1);
  const a = Math.sin(dphi / 2) ** 2 + Math.cos(phi1) * Math.cos(phi2) * Math.sin(dlambda / 2) ** 2;
  return 2 * EARTH_RADIUS_KM * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
}

function toCartesian(lon, lat) {
  const lambda = radians(lon);
  const phi = radians(lat);
  return [Math.cos(phi) * Math.cos(lambda), Math.cos(phi) * Math.sin(lambda), Math.sin(phi)];
}

function fromCartesian([x, y, z]) {
  const hyp = Math.hypot(x, y);
  return [degrees(Math.atan2(y, x)), degrees(Math.atan2(z, hyp))];
}

function dot(a, b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function append(name, attrs = {}, text = "") {
  const node = document.createElementNS(SVG_NS, name);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
  if (text) node.textContent = text;
  els.svg.appendChild(node);
  return node;
}

function appendPath(parent, d, attrs = {}) {
  if (!d) return null;
  const node = document.createElementNS(SVG_NS, "path");
  node.setAttribute("d", d);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
  parent.appendChild(node);
  return node;
}

function starPath(cx, cy, outer, inner, points) {
  const commands = [];
  for (let i = 0; i < points * 2; i += 1) {
    const radius = i % 2 === 0 ? outer : inner;
    const angle = -Math.PI / 2 + (i * Math.PI) / points;
    commands.push(`${i === 0 ? "M" : "L"} ${round(cx + Math.cos(angle) * radius)} ${round(cy + Math.sin(angle) * radius)}`);
  }
  return `${commands.join(" ")} Z`;
}

function downloadSvg() {
  const svgText = serializeSvg();
  downloadBlob(new Blob([svgText], { type: "image/svg+xml" }), `${OUTPUT_PREFIX}_${els.center.value}.svg`);
}

function downloadPng() {
  const svgText = serializeSvg();
  const img = new Image();
  const url = URL.createObjectURL(new Blob([svgText], { type: "image/svg+xml" }));

  img.onload = () => {
    const canvas = document.createElement("canvas");
    canvas.width = 2400;
    canvas.height = 2400;
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    URL.revokeObjectURL(url);
    canvas.toBlob((blob) => {
      if (blob) downloadBlob(blob, `${OUTPUT_PREFIX}_${els.center.value}.png`);
    }, "image/png");
  };

  img.src = url;
}

function serializeSvg() {
  const clone = els.svg.cloneNode(true);
  clone.setAttribute("xmlns", SVG_NS);
  return `<?xml version="1.0" encoding="UTF-8"?>\n${new XMLSerializer().serializeToString(clone)}`;
}

function downloadBlob(blob, filename) {
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = filename;
  link.click();
  URL.revokeObjectURL(link.href);
}

function airportOptionLabel(code) {
  const airport = airports[code] ?? {};
  const detail = [airport.city || airport.label || "", airport.country || ""].filter(Boolean).join(" / ");
  return detail ? `${code} - ${detail}` : code;
}

function sortedEntries(map) {
  return [...map.entries()].sort((a, b) => b[1] - a[1] || String(a[0]).localeCompare(String(b[0])));
}

function number(value) {
  return new Intl.NumberFormat("en-US").format(value);
}

function radians(value) {
  return (value * Math.PI) / 180;
}

function degrees(value) {
  return (value * 180) / Math.PI;
}

function normalizeRadians(value) {
  return Math.atan2(Math.sin(value), Math.cos(value));
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function round(value) {
  return Math.round(value * 100) / 100;
}
