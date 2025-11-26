# app.py
import os
import re
import math
import json
import time
from datetime import datetime, timedelta

import requests
import numpy as np
import pandas as pd
import streamlit as st
from google.oauth2.service_account import Credentials
import streamlit as st
import gspread
from google.oauth2.service_account import Credentials

# --------------------------- Google Sheets config ---------------------------
SERVICE_ACCOUNT_FILE = "service_account.json"
GSHEET_SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]

@st.cache_resource
def get_gsheet_client():
    creds = Credentials.from_service_account_info(
        st.secrets["gcp_service_account"],
        scopes=GSHEET_SCOPES,
    )
    return gspread.authorize(creds)


def append_log_row(sheet_id: str, boonie: str, x: int, y: int, arrival_dt: datetime):
    """Append a log row (boonie, x, y, arrival_time) to the 'log' sheet."""
    client = get_gsheet_client()
    sh = client.open_by_key(sheet_id)
    ws = sh.worksheet("log")
    ws.append_row(
        [boonie, x, y, arrival_dt.strftime("%Y-%m-%d %H:%M:%S")],
        value_input_option="USER_ENTERED",
    )


def get_active_targets(sheet_id: str, boonie: str):
    """
    Read the 'log' sheet and return a set of (x, y) pairs for this boonie
    where the arrival_time is still in the future (arrival > now).
    """
    try:
        client = get_gsheet_client()
        sh = client.open_by_key(sheet_id)
        ws = sh.worksheet("log")
        rows = ws.get_all_values()
    except Exception as e:
        # If log sheet doesn't exist or can't be read, just treat as no active targets
        return set()

    now = datetime.now()
    active = set()

    # Assume row 0 might be header; just try parsing from row 1 onwards
    for row in rows[1:]:
        if len(row) < 4:
            continue
        b, sx, sy, sat = row[0], row[1], row[2], row[3]
        if b != boonie:
            continue
        try:
            x = int(sx)
            y = int(sy)
            arrival = datetime.strptime(sat, "%Y-%m-%d %H:%M:%S")
        except Exception:
            continue

        if arrival > now:
            active.add((x, y))

    return active

# --------------------------- Constants & data ---------------------------
unit_mapping = {
    'u31': ('rat', 160),
    'u32': ('spider', 160),
    'u33': ('serpent', 160),
    'u34': ('bat', 160),
    'u35': ('wild boar', 320),
    'u36': ('wolf', 320),
    'u37': ('bear', 480),
    'u38': ('croco', 480),
    'u39': ('tiger', 480),
    'u40': ('elephant', 800)
}
xp_per_unit = [1, 1, 1, 1, 2, 2, 3, 3, 3, 5]
spawn_rates = [5,6,7,8,9,10,11,12,13,14]
animal_inf_def = [25,35,40,66,70,80,140,380,170,440]
animal_cav_def = [20,40,60,50,33,70,200,240,250,520]

tk_atk = 150
club_atk = 40
tk_cost = 1525
club_cost = 250

DEFAULT_SERVER = 'ts6.x1.europe.travian.com'

BOONIE_COORDS = {'C01': (-7, -90), 'C02': (133, -162), 'C03': (198, 47)}
BOONIE_VILIDS = {'C01': 42069, 'C02': 68319, 'C03': 72888}

DEFAULT_SHEET_ID = "1Ev6h8-rTbCHoK2vNEyYpBSVs4AvJ43XzPQ3CuC6sTSw"
DEFAULT_SHEET_NAME = "cookie"

# --------------------------- Core functions ---------------------------
def get_cost(hp_lost, nclubs, ntk):
    TK_lost = np.round(ntk * hp_lost, 0)
    clubs_lost = np.round(nclubs * hp_lost, 0)
    return TK_lost * tk_cost + clubs_lost * club_cost


def get_res_gained(animals, hp_lost):
    actual_points = 0
    for i, count in enumerate(animals):
        units_killed = round(count * (1 - hp_lost))
        points_per_unit = unit_mapping[f'u{31 + i}'][1]
        actual_points += units_killed * points_per_unit
    return actual_points


def get_calculation_result(animals, nclub, ntk):
    inf_atk = club_atk * nclub
    cav_atk = tk_atk * ntk
    inf_def = np.sum([a * d for a, d in zip(animals, animal_inf_def)])
    cav_def = np.sum([a * d for a, d in zip(animals, animal_cav_def)])
    attacker_points = inf_atk + cav_atk
    nature_points = (inf_def * (inf_atk / attacker_points)) + (cav_def * (cav_atk / attacker_points)) + 10
    atk_result = 100 * (nature_points / attacker_points) ** 1.5
    hp_lost = atk_result / (atk_result + 100)
    return hp_lost


def get_best_team(animals, max_clubs=500, max_ntk=20):
    nclubs_range = np.arange(1, max_clubs + 1)
    ntk_range = np.arange(1, max_ntk + 1)

    X, Y = np.meshgrid(nclubs_range, ntk_range)
    costs = np.zeros_like(X, dtype=float)
    gains = np.zeros_like(X, dtype=float)
    profit_ratio = np.zeros_like(X, dtype=float)

    for i in range(len(ntk_range)):
        for j in range(len(nclubs_range)):
            nclubs = X[i, j]
            ntk = Y[i, j]
            hp_lost = get_calculation_result(animals, nclubs, ntk)
            cost = get_cost(hp_lost, nclubs, ntk)
            gain = get_res_gained(animals, hp_lost)
            team_cost = 250 * nclubs + 1525 * ntk

            costs[i, j] = cost
            gains[i, j] = gain
            if team_cost > 0:
                profit_ratio[i, j] = (gain - cost) / team_cost
            else:
                profit_ratio[i, j] = -1e9

    flat_indices = np.argpartition(profit_ratio.ravel(), -10)[-10:]
    top10_indices = flat_indices[np.argsort(profit_ratio.ravel()[flat_indices])][::-1]

    best_points = []
    for idx in top10_indices:
        i, j = np.unravel_index(idx, profit_ratio.shape)
        if profit_ratio[i, j] > 0:
            nclubs_best = X[i, j]
            ntk_best = Y[i, j]
            ratio_best = profit_ratio[i, j]
            team_cost = 250 * nclubs_best + 1525 * ntk_best
            gain = gains[i, j]
            cost = costs[i, j]
            best_points.append((nclubs_best, ntk_best, ratio_best, team_cost, gain, cost))

    if not best_points:
        return None

    final_best = min(best_points, key=lambda x: x[3])
    return final_best


def calculate_distance(x1, y1, x2, y2):
    deltax = x2 - x1
    if deltax > 350:
        deltax = 401 - deltax
    deltay = y2 - y1
    if deltay > 350:
        deltay = 401 - deltay
    return round(math.sqrt(deltax ** 2 + deltay ** 2), 2)


def extract_data(entry, x_vil, y_vil):
    x_match = re.search(r'"x":\s*(-?\d+)', entry)
    y_match = re.search(r'"y":\s*(-?\d+)', entry)
    if not (x_match and y_match):
        return None

    x_coord = int(x_match.group(1))
    y_coord = int(y_match.group(1))
    distance = calculate_distance(x_coord, y_coord, x_vil, y_vil)

    units = re.findall(r'class="unit u(3[1-9]|40)"><\/i><span class="value ">(\d+)<\/span>', entry)
    unit_counts = [0] * 10
    total_points = 0

    for unit_id, count in units:
        index = int(unit_id) - 31
        count = int(count)
        unit_counts[index] = count
        points_per_unit = unit_mapping[f'u{unit_id}'][1]
        total_points += count * points_per_unit

    return {"x": x_coord, "y": y_coord, "distance": distance, "total_points": total_points, "unit_counts": unit_counts}


def post_request(server, x, y, cookie, fullmap=False):
    url = f"https://{server}/api/v1/map/position"
    headers = {
        "authority": f"{server}",
        "method": "POST",
        "path": "/api/v1/map/position",
        "scheme": "https",
        "accept": "application/json, text/javascript, */*; q=0.01",
        "accept-encoding": "gzip, deflate, br, zstd",
        "accept-language": "en-US,en;q=0.9",
        "content-type": "application/json; charset=UTF-8",
        "cookie": cookie,
        "origin": f"https://{server}",
        "referer": f"https://{server}/karte.php",
        "sec-ch-ua": '"Google Chrome";v="131", "Chromium";v="131", "Not_A Brand";v="24"',
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": '"Windows"',
        "sec-fetch-dest": "empty",
        "sec-fetch-mode": "cors",
        "sec-fetch-site": "same-origin",
        "user-agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
        ),
        "x-requested-with": "XMLHttpRequest",
        "x-version": "52.8",
    }

    if fullmap:
        texts = ""
        for xi in range(x - 90, x + 90, 30):
            for yi in range(y - 90, y + 90, 30):
                payload = {"data": {"x": xi, "y": yi, "zoomLevel": 3, "ignorePositions": []}}
                r = requests.post(url, headers=headers, json=payload, timeout=25)
                r.raise_for_status()
                texts += r.text
                time.sleep(0.15)
        return texts

    payload = {"data": {"x": x, "y": y, "zoomLevel": 3, "ignorePositions": []}}
    r = requests.post(url, headers=headers, json=payload, timeout=25)
    r.raise_for_status()
    return r.text


def process_map_data(server, x_vil, y_vil, cookie, fullmap):
    data = post_request(server, x_vil, y_vil, cookie, fullmap)
    if "Unauth" in data:
        raise RuntimeError("Cookie invalid/expired (found 'Unauth' in response).")

    cleaned = data.replace("\\", "")
    chunks = cleaned.split('"position":{')
    filtered = [c for c in chunks if "animal" in c]
    filtered = list(set(filtered))
    processed = [extract_data(entry, x_vil, y_vil) for entry in filtered]
    processed = [p for p in processed if p is not None]
    return processed


def get_nodeid(x, y):
    row_offset = (200 - y) * 401
    col_offset = x + 200
    nodeid = row_offset + col_offset + 1
    return nodeid


def build_url(x, y, boonie, nclub, ntk):
    starting_part = r"https://ts6.x1.europe.travian.com/build.php?"
    village_id = BOONIE_VILIDS[boonie]
    troop_part = f"tt=2&troop%5Bt1%5D={nclub}&troop%5Bt3%5D={ntk}&"
    node_id = get_nodeid(x, y)
    return (
        f"{starting_part}"
        f"newdid={village_id}&gid=16&"
        f"{troop_part}"
        f"targetMapId={node_id}&eventType=4&"
    )


def adapt_unit_counts(unit_counts, distance, speed):
    last_non_zero_index = len(unit_counts) - 1
    for i in range(len(unit_counts) - 1, -1, -1):
        if unit_counts[i] != 0:
            last_non_zero_index = i
            break
    spawn_rate = spawn_rates[last_non_zero_index]

    if distance < 20:
        runtime = distance / speed
    else:
        runtime = (20 / speed) + ((distance - 20) / (speed * 4.2))
    minutes = int(runtime * 60)

    units_spawned = round(minutes / spawn_rate)
    unit_counts[last_non_zero_index] = unit_counts[last_non_zero_index] + units_spawned
    return unit_counts


def read_cookie_from_sheet(sheet_id: str, sheet_name: str) -> str:
    csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"
    df = pd.read_csv(csv_url, header=None)
    return str(df.values[0][0])


# --------------------------- Streamlit UI ---------------------------
st.set_page_config(page_title="Boonie Oasis Planner", layout="wide")
st.title("Boonie Oasis Planner")

with st.expander("Inputs", expanded=True):
    c1, c2, c3 = st.columns(3)
    boonie = c1.selectbox("Select boonie", options=list(BOONIE_COORDS.keys()), index=0)
    maxdist = c2.number_input("Max distance", value=20.0, min_value=1.0, step=1.0)
    maxclubs = c3.number_input("Max clubs", value=250, min_value=1, step=1)

    c4, c5, c6 = st.columns(3)
    speed = c4.number_input("Hero speed (for runtime & arrival)", value=7, min_value=1, step=1)
    server = c5.text_input("Server for map fetch", value=DEFAULT_SERVER)
    fullmap = c6.checkbox("Scan 180×180 (fullmap window)", value=False)

    st.markdown("**Cookie** source")
    c7, c8 = st.columns(2)
    cookie_mode = c7.radio(
        "",
        options=["Google Sheet", "Paste manually"],
        index=0,
        horizontal=True,
        label_visibility="collapsed",
    )
    if cookie_mode == "Google Sheet":
        sheet_id = c7.text_input("Google Sheet ID", value=DEFAULT_SHEET_ID)
        sheet_name = c8.text_input("Sheet name", value=DEFAULT_SHEET_NAME)
        cookie = None
    else:
        cookie = c8.text_area("Paste cookie", value="", height=80)
        sheet_id = DEFAULT_SHEET_ID
        sheet_name = DEFAULT_SHEET_NAME

run_btn = st.button("Analyze oases")

st.markdown("---")
results_container = st.container()

# --------------------------- Analyze & store in session ---------------------------
if run_btn:
    try:
        os.environ["NO_PROXY"] = "*"
        os.environ["no_proxy"] = "*"

        x_vil, y_vil = BOONIE_COORDS[boonie]
        if cookie is None:
            cookie = read_cookie_from_sheet(sheet_id, sheet_name)

        # NEW: load active targets from log & skip those
        active_targets = get_active_targets(sheet_id, boonie)

        entries = process_map_data(server, x_vil, y_vil, cookie, fullmap)

        if not entries:
            st.session_state["valid_entries"] = []
            st.warning("No oasis with animals found.")
        else:
            valid_entries = []
            progress = st.progress(0.0, text="Evaluating best teams...")
            total = len(entries)
            for idx, entry in enumerate(entries, 1):
                progress.progress(idx / total, text=f"Evaluating {idx}/{total}")

                # Skip if this oasis is already in log with future arrival
                if (entry['x'], entry['y']) in active_targets:
                    continue

                if entry['distance'] < maxdist:
                    animals = adapt_unit_counts(entry['unit_counts'], entry['distance'], speed)
                    best = get_best_team(animals, max_clubs=maxclubs, max_ntk=20)
                    if best is None:
                        continue

                    nc, nt, ratio, tc, g, cst = best
                    hp_lost_frac = get_calculation_result(animals, nc, nt)
                    hp_lost_pct = np.round(hp_lost_frac * 100, 0)

                    total_res = g - cst
                    resource_score = round(((g - cst) * speed) / entry['distance'], 0) / 2

                    if hp_lost_pct < 60 and nc < maxclubs:
                        valid_entries.append({
                            "x": entry['x'],
                            "y": entry['y'],
                            "distance": entry['distance'],
                            "nc": int(nc),
                            "nt": int(nt),
                            "GS/u": resource_score,
                            "%_lost": int(hp_lost_pct),
                            "GS": int(total_res),
                        })

            progress.empty()
            valid_entries.sort(key=lambda x: x["GS/u"], reverse=True)
            st.session_state["valid_entries"] = valid_entries
            st.session_state["meta"] = {
                "boonie": boonie,
                "speed": float(speed),
                "sheet_id": sheet_id,
            }

    except requests.exceptions.ProxyError as e:
        st.error("Proxy blocked the request. Run locally or on a host with unrestricted internet.")
        st.exception(e)
    except requests.HTTPError as e:
        st.error("HTTP error from server.")
        st.exception(e)
    except Exception as e:
        st.error("Unexpected error during analysis.")
        st.exception(e)

# --------------------------- Render from session_state ---------------------------
valid_entries = st.session_state.get("valid_entries", [])
meta = st.session_state.get("meta", {})

if valid_entries:
    boonie_for_results = meta.get("boonie", boonie)
    speed_for_results = float(meta.get("speed", speed))
    sheet_id_for_results = meta.get("sheet_id", sheet_id)

    results_container.subheader("Results")

    for idx, e in enumerate(valid_entries):
        col_btn, col_text = results_container.columns([1, 9])

        sent_state_key = f"sent_state_{boonie_for_results}_{e['x']}_{e['y']}"
        sent_button_key = f"sent_btn_{boonie_for_results}_{e['x']}_{e['y']}_{idx}"

        already_sent = st.session_state.get(sent_state_key, False)

        sent_clicked = col_btn.button(
            "Sent!",
            key=sent_button_key,
            disabled=already_sent,
        )

        # Determine display state (we want immediate “✅ SENT” feedback)
        sent_flag = already_sent or sent_clicked

        url = build_url(e['x'], e['y'], boonie_for_results, e['nc'], e['nt'])
        prefix = "✅ SENT - " if sent_flag else ""
        line = (
            f"{prefix}"
            f"x: {e['x']}, y: {e['y']}, team:{e['nc']}|{e['nt']} "
            f"dist.: {round(e['distance'],1)},\t GS/u: {e['GS/u']},\t "
            f"%_lost: {e['%_lost']},\t GS: {e['GS']} "
            f"[link]({url})"
        )
        col_text.markdown(line)

        if sent_clicked and not already_sent:
            # Mark as sent in session state
            st.session_state[sent_state_key] = True

            # Compute arrival time: now + distance / speed (hours)
            travel_hours = e["distance"] / float(speed_for_results)
            arrival_time = datetime.now() + timedelta(hours=travel_hours)

            # Log to Google Sheet ('log' tab)
            try:
                append_log_row(sheet_id_for_results, boonie_for_results, e["x"], e["y"], arrival_time)
                st.success(
                    f"Logged to sheet: {boonie_for_results}, {e['x']}, {e['y']}, "
                    f"arrival {arrival_time.strftime('%Y-%m-%d %H:%M:%S')}"
                )
            except Exception as log_err:
                st.error(f"Failed to log to Google Sheet: {log_err}")

elif run_btn is False:
    # First load with no results and no run yet
    pass

