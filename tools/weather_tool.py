from pydantic import BaseModel
from typing import Literal


class Weather(BaseModel):
    temperature: float
    wind_speed_mph: float
    precipitation_mm: float


def get_current_weather(
    location: Literal["New York", "Dunfermline", "Tokyo"],
) -> Weather:
    """Get the weather from a particular location"""

    if location == "New York":
        return Weather(temperature=75.0, wind_speed_mph=5.0, precipitation_mm=0.0)
    elif location == "Dunfermline":
        return Weather(temperature=60.0, wind_speed_mph=10.0, precipitation_mm=2.0)
    elif location == "Tokyo":
        return Weather(temperature=80.0, wind_speed_mph=3.0, precipitation_mm=1.0)
    raise ValueError(f"Unsupported location: {location}")
