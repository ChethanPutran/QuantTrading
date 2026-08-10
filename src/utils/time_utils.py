from datetime import datetime
from typing import Any

from dateutil import parser as date_parser
from dateutil.tz import gettz, tzutc


IST = gettz("Asia/Kolkata")
TZINFOS = {
    "IST": IST,
    "GMT": tzutc(),
    "UTC": tzutc(),
    "Z": tzutc(),
}


def normalize_to_ist(value: Any) -> datetime | None:
    if value is None:
        return None

    try:
        if isinstance(value, (int, float)):
            dt = datetime.fromtimestamp(value, tz=tzutc())
        elif isinstance(value, datetime):
            dt = value
        else:
            dt = date_parser.parse(str(value), tzinfos=TZINFOS)

        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=IST)

        return dt.astimezone(IST)
    except Exception:
        return None
