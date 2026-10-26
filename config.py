"""
Shared constants, team mappings, and classification logic.
All other scripts import from here.
"""

import os

# --- Paths ---
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "output")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Year range ---
YEARS = range(2022, 2027)  # 2022-2026
HISTORY_YEARS = range(2018, 2027)  # 2018-2026 for debut tracking

# --- Team name standardization ---
# FanGraphs team batting/pitching/fielding uses short abbreviations.
# Standings from Baseball Reference uses full names.
# This maps everything to a canonical 3-letter code.

# Baseball Reference full name -> canonical abbreviation
BR_NAME_TO_ABBR = {
    "Arizona Diamondbacks": "ARI",
    "Atlanta Braves": "ATL",
    "Baltimore Orioles": "BAL",
    "Boston Red Sox": "BOS",
    "Chicago Cubs": "CHC",
    "Chicago White Sox": "CHW",
    "Cincinnati Reds": "CIN",
    "Cleveland Guardians": "CLE",
    "Cleveland Indians": "CLE",
    "Colorado Rockies": "COL",
    "Detroit Tigers": "DET",
    "Houston Astros": "HOU",
    "Kansas City Royals": "KCR",
    "Los Angeles Angels": "LAA",
    "Los Angeles Dodgers": "LAD",
    "Miami Marlins": "MIA",
    "Milwaukee Brewers": "MIL",
    "Minnesota Twins": "MIN",
    "New York Mets": "NYM",
    "New York Yankees": "NYY",
    "Oakland Athletics": "OAK",
    "Philadelphia Phillies": "PHI",
    "Pittsburgh Pirates": "PIT",
    "San Diego Padres": "SDP",
    "San Francisco Giants": "SFG",
    "Seattle Mariners": "SEA",
    "St. Louis Cardinals": "STL",
    "Tampa Bay Rays": "TBR",
    "Texas Rangers": "TEX",
    "Toronto Blue Jays": "TOR",
    "Washington Nationals": "WSN",
    "Sacramento Athletics": "OAK",
}

# FanGraphs sometimes uses slightly different abbreviations
FG_ABBR_MAP = {
    "ARI": "ARI", "ATL": "ATL", "BAL": "BAL", "BOS": "BOS",
    "CHC": "CHC", "CHW": "CHW", "CIN": "CIN", "CLE": "CLE",
    "COL": "COL", "DET": "DET", "HOU": "HOU", "KCR": "KCR",
    "LAA": "LAA", "LAD": "LAD", "MIA": "MIA", "MIL": "MIL",
    "MIN": "MIN", "NYM": "NYM", "NYY": "NYY", "OAK": "OAK",
    "PHI": "PHI", "PIT": "PIT", "SDP": "SDP", "SFG": "SFG",
    "SEA": "SEA", "STL": "STL", "TBR": "TBR", "TEX": "TEX",
    "TOR": "TOR", "WSN": "WSN",
    # Common FanGraphs variants
    "WSH": "WSN", "KCA": "KCR", "TBA": "TBR", "SFN": "SFG",
    "SDN": "SDP", "NYA": "NYY", "NYN": "NYM", "CHA": "CHW",
    "CHN": "CHC", "SLN": "STL", "ANA": "LAA", "FLA": "MIA",
    "MON": "WSN", "ATH": "OAK",
}

# Statcast 3-letter codes (used by Baseball Savant)
STATCAST_ABBR_MAP = {
    "ARI": "ARI", "ATL": "ATL", "BAL": "BAL", "BOS": "BOS",
    "CHC": "CHC", "CWS": "CHW", "CIN": "CIN", "CLE": "CLE",
    "COL": "COL", "DET": "DET", "HOU": "HOU", "KC": "KCR",
    "LAA": "LAA", "LAD": "LAD", "MIA": "MIA", "MIL": "MIL",
    "MIN": "MIN", "NYM": "NYM", "NYY": "NYY", "OAK": "OAK",
    "PHI": "PHI", "PIT": "PIT", "SD": "SDP", "SF": "SFG",
    "SEA": "SEA", "STL": "STL", "TB": "TBR", "TEX": "TEX",
    "TOR": "TOR", "WSH": "WSN", "ATH": "OAK",
}

ALL_TEAMS = sorted(set(BR_NAME_TO_ABBR.values()))  # 30 canonical codes


# BRef uses short city names. Disambiguate using league (AL/NL).
CITY_LEAGUE_MAP = {
    ("New York", "AL"): "NYY",
    ("New York", "NL"): "NYM",
    ("Los Angeles", "AL"): "LAA",
    ("Los Angeles", "NL"): "LAD",
    ("Chicago", "AL"): "CHW",
    ("Chicago", "NL"): "CHC",
}

# BRef short city names (unambiguous)
BR_SHORT_TO_ABBR = {
    "Arizona": "ARI", "Atlanta": "ATL", "Baltimore": "BAL", "Boston": "BOS",
    "Cincinnati": "CIN", "Cleveland": "CLE", "Colorado": "COL", "Detroit": "DET",
    "Houston": "HOU", "Kansas City": "KCR", "Miami": "MIA", "Milwaukee": "MIL",
    "Minnesota": "MIN", "Oakland": "OAK", "Athletics": "OAK", "Sacramento": "OAK",
    "Philadelphia": "PHI", "Pittsburgh": "PIT", "San Diego": "SDP",
    "San Francisco": "SFG", "Seattle": "SEA", "St. Louis": "STL",
    "Tampa Bay": "TBR", "Texas": "TEX", "Toronto": "TOR", "Washington": "WSN",
}


def standardize_team(name, league=None):
    """
    Convert any team name/abbreviation to canonical 3-letter code.
    For ambiguous city names (New York, Los Angeles, Chicago),
    pass league="AL" or "NL" to disambiguate.
    """
    if not isinstance(name, str):
        return None
    name = name.strip()
    # Try direct abbreviation maps first
    if name in FG_ABBR_MAP:
        return FG_ABBR_MAP[name]
    if name in STATCAST_ABBR_MAP:
        return STATCAST_ABBR_MAP[name]
    # Try full name
    if name in BR_NAME_TO_ABBR:
        return BR_NAME_TO_ABBR[name]
    # Try BRef short city name (unambiguous)
    if name in BR_SHORT_TO_ABBR:
        return BR_SHORT_TO_ABBR[name]
    # Ambiguous city: use league to disambiguate
    if league:
        lg = "AL" if "AL" in str(league) else "NL" if "NL" in str(league) else None
        if lg and (name, lg) in CITY_LEAGUE_MAP:
            return CITY_LEAGUE_MAP[(name, lg)]
    # Default for ambiguous cities without league info
    if name == "New York":
        return "NYY"  # default
    if name == "Los Angeles":
        return "LAD"  # default
    if name == "Chicago":
        return "CHC"  # default
    # Fuzzy: check if the name contains a known full name
    for full, abbr in BR_NAME_TO_ABBR.items():
        if full in name or name in full:
            return abbr
    return name  # Return as-is if no match


# --- Playoff teams by year (12 teams per year starting 2022) ---
PLAYOFF_TEAMS = {
    2022: ["HOU", "NYY", "CLE", "TBR", "SEA", "TOR",
           "LAD", "ATL", "STL", "NYM", "SDP", "PHI"],
    2023: ["TEX", "HOU", "BAL", "MIN", "TBR", "TOR",
           "ARI", "ATL", "LAD", "MIL", "PHI", "MIA"],
    2024: ["NYY", "CLE", "HOU", "BAL", "KCR", "DET",
           "LAD", "PHI", "MIL", "SDP", "ATL", "NYM"],
    2025: ["TOR", "SEA", "CLE", "NYY", "BOS", "DET",
           "MIL", "PHI", "LAD", "CHC", "SDP", "CIN"],
    # 2026 -- World Series: MIL vs TBR (ongoing Oct 2026)
    2026: ["TBR", "CLE", "HOU", "NYY", "BOS", "CHW",
           "MIL", "LAD", "ATL", "SDP", "CHC", "PHI"],
}

# Division winners by year
DIVISION_WINNERS = {
    2022: ["HOU", "NYY", "CLE", "LAD", "ATL", "STL"],
    2023: ["BAL", "MIN", "HOU", "ATL", "MIL", "LAD"],
    2024: ["NYY", "CLE", "HOU", "LAD", "PHI", "MIL"],
    2025: ["TOR", "CLE", "SEA", "PHI", "MIL", "LAD"],
    2026: ["TBR", "CLE", "HOU", "ATL", "MIL", "LAD"],
}


def classify_team(team, year, payroll_rank_pct, wins):
    """
    Classify a team-season into a comparison group.

    Args:
        team: canonical 3-letter code
        year: season year
        payroll_rank_pct: 0-1, where 0 = lowest payroll, 1 = highest
        wins: season wins

    Returns:
        str: one of 'efficient_winner', 'big_budget_winner',
             'cheap_loser', 'expensive_loser', 'middle_winner', 'middle_loser'
    """
    made_playoffs = team in PLAYOFF_TEAMS.get(year, [])
    # Safety: teams below .463 W% (75 wins) are never "winners"
    # regardless of playoff tag (catches data entry errors)
    is_good = (made_playoffs and wins >= 75) or wins >= 85

    if payroll_rank_pct <= 0.333:
        return "efficient_winner" if is_good else "cheap_loser"
    elif payroll_rank_pct >= 0.667:
        return "big_budget_winner" if is_good else "expensive_loser"
    else:
        return "middle_winner" if is_good else "middle_loser"


# --- Payroll data (millions USD, opening day payroll) ---
# Sources: Spotrac, Cot's Baseball Contracts, USA Today salary database
# These are approximate opening day active roster payrolls.
PAYROLL_DATA = {
    2022: {
        "LAD": 267, "NYM": 255, "NYY": 249, "SDP": 206, "PHI": 204,
        "BOS": 197, "CHC": 180, "STL": 177, "LAA": 173, "HOU": 172,
        "ATL": 170, "CHW": 168, "SFG": 166, "TEX": 164, "TOR": 160,
        "MIN": 138, "SEA": 133, "MIL": 131, "WSN": 127, "CIN": 118,
        "COL": 117, "DET": 107, "KCR": 103, "TBR": 86, "MIA": 83,
        "ARI": 74, "CLE": 68, "PIT": 63, "OAK": 48, "BAL": 45,
    },
    2023: {
        "NYM": 341, "NYY": 278, "LAD": 232, "SDP": 222, "PHI": 213,
        "TEX": 210, "STL": 196, "BOS": 189, "LAA": 183, "CHW": 174,
        "ATL": 171, "HOU": 166, "SFG": 163, "CHC": 162, "TOR": 157,
        "MIN": 145, "SEA": 140, "CIN": 130, "COL": 119, "MIL": 117,
        "DET": 112, "WSN": 106, "KCR": 95, "ARI": 92, "TBR": 82,
        "MIA": 78, "CLE": 74, "PIT": 67, "OAK": 58, "BAL": 54,
    },
    2024: {
        "NYM": 306, "LAD": 285, "NYY": 280, "PHI": 248, "TEX": 233,
        "SDP": 223, "ATL": 210, "BOS": 208, "SFG": 195, "HOU": 190,
        "STL": 181, "CHC": 175, "LAA": 170, "TOR": 168, "SEA": 158,
        "MIN": 148, "CHW": 140, "CIN": 123, "DET": 118, "MIL": 114,
        "COL": 109, "WSN": 105, "ARI": 101, "KCR": 94, "MIA": 86,
        "TBR": 80, "CLE": 79, "PIT": 71, "OAK": 59, "BAL": 55,
    },
    2025: {
        "NYM": 316, "LAD": 310, "NYY": 290, "PHI": 261, "BOS": 228,
        "SDP": 218, "ATL": 215, "TEX": 205, "SFG": 198, "HOU": 195,
        "STL": 178, "CHC": 172, "TOR": 162, "SEA": 155, "MIN": 149,
        "LAA": 145, "DET": 133, "CIN": 128, "MIL": 118, "CHW": 98,
        "COL": 95, "ARI": 92, "WSN": 88, "KCR": 85, "MIA": 78,
        "TBR": 75, "CLE": 72, "PIT": 68, "BAL": 65, "OAK": 55,
    },
    # 2026 payroll estimates (approximate, from preseason projections)
    2026: {
        "NYM": 325, "LAD": 315, "NYY": 295, "PHI": 265, "BOS": 235,
        "SDP": 220, "ATL": 218, "TEX": 200, "SFG": 195, "HOU": 188,
        "STL": 175, "CHC": 170, "TOR": 158, "SEA": 150, "DET": 142,
        "MIN": 140, "LAA": 130, "CIN": 125, "MIL": 115, "ARI": 100,
        "COL": 95, "WSN": 90, "KCR": 88, "CHW": 85, "MIA": 75,
        "TBR": 72, "BAL": 68, "CLE": 65, "PIT": 62, "OAK": 50,
    },
}

# Premium defensive positions (higher replacement level)
PREMIUM_POSITIONS = {"C", "SS", "CF", "2B"}
