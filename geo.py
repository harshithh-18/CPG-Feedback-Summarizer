"""India geo lookup — resolve a free-text location string to lat/lon + a
canonical display name, using a small bundled gazetteer of Indian cities and
states. No network, no external files: works fully offline so the India map on
the dashboard renders as soon as the dataset carries a ``location`` field.

The lookup is deliberately forgiving: records may say "Mumbai", "mumbai, MH",
"Bengaluru", "New Delhi", or a bare state like "Kerala" — all resolve. Anything
unrecognized is reported back so the UI can show how many points it couldn't
place instead of silently dropping them.
"""

from __future__ import annotations

import re
from typing import Any

# name -> (display_name, latitude, longitude). Keys are lowercase; aliases point
# at the same tuple. Coordinates are city centroids (state entries use the
# capital / geographic centroid) — precise enough for a scatter map.
_CITIES: dict[str, tuple[str, float, float]] = {
    "mumbai": ("Mumbai", 19.0760, 72.8777),
    "bombay": ("Mumbai", 19.0760, 72.8777),
    "delhi": ("Delhi", 28.6139, 77.2090),
    "new delhi": ("Delhi", 28.6139, 77.2090),
    "bengaluru": ("Bengaluru", 12.9716, 77.5946),
    "bangalore": ("Bengaluru", 12.9716, 77.5946),
    "hyderabad": ("Hyderabad", 17.3850, 78.4867),
    "chennai": ("Chennai", 13.0827, 80.2707),
    "madras": ("Chennai", 13.0827, 80.2707),
    "kolkata": ("Kolkata", 22.5726, 88.3639),
    "calcutta": ("Kolkata", 22.5726, 88.3639),
    "pune": ("Pune", 18.5204, 73.8567),
    "ahmedabad": ("Ahmedabad", 23.0225, 72.5714),
    "surat": ("Surat", 21.1702, 72.8311),
    "jaipur": ("Jaipur", 26.9124, 75.7873),
    "lucknow": ("Lucknow", 26.8467, 80.9462),
    "kanpur": ("Kanpur", 26.4499, 80.3319),
    "nagpur": ("Nagpur", 21.1458, 79.0882),
    "indore": ("Indore", 22.7196, 75.8577),
    "bhopal": ("Bhopal", 23.2599, 77.4126),
    "visakhapatnam": ("Visakhapatnam", 17.6868, 83.2185),
    "vizag": ("Visakhapatnam", 17.6868, 83.2185),
    "patna": ("Patna", 25.5941, 85.1376),
    "vadodara": ("Vadodara", 22.3072, 73.1812),
    "baroda": ("Vadodara", 22.3072, 73.1812),
    "ghaziabad": ("Ghaziabad", 28.6692, 77.4538),
    "ludhiana": ("Ludhiana", 30.9010, 75.8573),
    "agra": ("Agra", 27.1767, 78.0081),
    "nashik": ("Nashik", 19.9975, 73.7898),
    "faridabad": ("Faridabad", 28.4089, 77.3178),
    "meerut": ("Meerut", 28.9845, 77.7064),
    "rajkot": ("Rajkot", 22.3039, 70.8022),
    "varanasi": ("Varanasi", 25.3176, 82.9739),
    "srinagar": ("Srinagar", 34.0837, 74.7973),
    "amritsar": ("Amritsar", 31.6340, 74.8723),
    "kochi": ("Kochi", 9.9312, 76.2673),
    "cochin": ("Kochi", 9.9312, 76.2673),
    "thiruvananthapuram": ("Thiruvananthapuram", 8.5241, 76.9366),
    "trivandrum": ("Thiruvananthapuram", 8.5241, 76.9366),
    "coimbatore": ("Coimbatore", 11.0168, 76.9558),
    "madurai": ("Madurai", 9.9252, 78.1198),
    "guwahati": ("Guwahati", 26.1445, 91.7362),
    "chandigarh": ("Chandigarh", 30.7333, 76.7794),
    "gurgaon": ("Gurugram", 28.4595, 77.0266),
    "gurugram": ("Gurugram", 28.4595, 77.0266),
    "noida": ("Noida", 28.5355, 77.3910),
    "mysuru": ("Mysuru", 12.2958, 76.6394),
    "mysore": ("Mysuru", 12.2958, 76.6394),
    "bhubaneswar": ("Bhubaneswar", 20.2961, 85.8245),
    "dehradun": ("Dehradun", 30.3165, 78.0322),
    "raipur": ("Raipur", 21.2514, 81.6296),
    "ranchi": ("Ranchi", 23.3441, 85.3096),
    "jodhpur": ("Jodhpur", 26.2389, 73.0243),
    # State-level fallbacks (capital / centroid).
    "maharashtra": ("Maharashtra", 19.0760, 72.8777),
    "karnataka": ("Karnataka", 12.9716, 77.5946),
    "tamil nadu": ("Tamil Nadu", 13.0827, 80.2707),
    "telangana": ("Telangana", 17.3850, 78.4867),
    "gujarat": ("Gujarat", 23.0225, 72.5714),
    "rajasthan": ("Rajasthan", 26.9124, 75.7873),
    "uttar pradesh": ("Uttar Pradesh", 26.8467, 80.9462),
    "west bengal": ("West Bengal", 22.5726, 88.3639),
    "kerala": ("Kerala", 9.9312, 76.2673),
    "punjab": ("Punjab", 31.6340, 74.8723),
    "madhya pradesh": ("Madhya Pradesh", 23.2599, 77.4126),
    "bihar": ("Bihar", 25.5941, 85.1376),
    "andhra pradesh": ("Andhra Pradesh", 17.6868, 83.2185),
    "odisha": ("Odisha", 20.2961, 85.8245),
    "haryana": ("Haryana", 28.4595, 77.0266),
    "assam": ("Assam", 26.1445, 91.7362),
}

# Two-letter state codes some datasets append (e.g. "Pune, MH").
_STATE_CODES = {
    "mh", "ka", "tn", "tg", "ts", "gj", "rj", "up", "wb", "kl", "pb", "mp",
    "br", "ap", "od", "hr", "as", "dl", "ch", "uk", "jk",
}


def _canonical_key(raw: str) -> str:
    """Lowercase, strip a trailing state code / state name, collapse spaces."""
    s = raw.strip().lower()
    # Drop a trailing ", XX" state code or ", state" qualifier.
    parts = [p.strip() for p in re.split(r"[,/|]", s) if p.strip()]
    if parts:
        # Prefer the first part that resolves to a known place.
        for p in parts:
            if p in _CITIES:
                return p
        # Otherwise drop obvious state-code tails and retry the head.
        head = parts[0]
        if head in _CITIES:
            return head
        return head
    return s


def resolve_location(raw: Any) -> tuple[str, float, float] | None:
    """Resolve a free-text location to (display_name, lat, lon), or None."""
    if not raw or not isinstance(raw, str):
        return None
    key = _canonical_key(raw)
    hit = _CITIES.get(key)
    if hit:
        return hit
    # Last resort: token scan for any known place name inside the string.
    tokens = re.sub(r"[^a-z\s]", " ", raw.lower())
    for name, tup in _CITIES.items():
        if name in _STATE_CODES:
            continue
        if re.search(rf"\b{re.escape(name)}\b", tokens):
            return tup
    return None


def location_field(record: dict[str, Any]) -> str | None:
    """Pull a location string from whichever common key a record uses."""
    for key in ("location", "city", "state", "region", "place"):
        val = record.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return None
