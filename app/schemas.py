"""Pydantic request/response schemas."""
from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    # Context features
    airport:           Optional[str]   = Field(None, description="ICAO airport code")
    runway:            Optional[str]   = Field(None, description="Runway identifier")
    typecode:          Optional[str]   = Field(None, description="Aircraft type code (ICAO)")
    icaoaircrafttype:  Optional[str]   = Field(None, description="ICAO aircraft type category")
    wtc:               Optional[str]   = Field(None, description="Wake turbulence category: L/M/H/J")
    has_intersection:  Optional[str]   = Field(None, description="Intersection approach flag")
    airport_country:   Optional[str]   = Field(None, description="Airport country code")
    airport_region:    Optional[str]   = Field(None, description="Airport region")
    operator_country:  Optional[str]   = Field(None, description="Operator country code")
    operator_region:   Optional[str]   = Field(None, description="Operator region")
    n_approaches:      Optional[float] = Field(None, description="Number of approaches")
    n_rwy_approached:  Optional[float] = Field(None, description="Number of runways approached")
    glide_slope_angle: Optional[float] = Field(None, description="Glide slope angle (degrees)")
    rwy_length:        Optional[float] = Field(None, description="Runway length (metres)")
    month:             Optional[float] = Field(None, description="Month (1-12)")
    day_of_week:       Optional[float] = Field(None, description="Day of week (0=Mon, 6=Sun)")
    hour_utc:          Optional[float] = Field(None, description="Hour UTC (0-23)")
    # METAR features
    wind_speed_knts:   Optional[float] = Field(None, description="Wind speed (knots)")
    wind_dir_deg:      Optional[float] = Field(None, description="Wind direction (degrees)")
    wind_gust_knts:    Optional[float] = Field(None, description="Wind gust speed (knots)")
    visibility_m:      Optional[float] = Field(None, description="Visibility (metres)")
    temperature_deg:   Optional[float] = Field(None, description="Temperature (°C)")
    press_sea_level_p: Optional[float] = Field(None, description="Sea-level pressure (hPa)")
    press_p:           Optional[float] = Field(None, description="Station pressure (hPa)")
    weather_intensity: Optional[str]   = Field(None, description="Weather intensity code")
    weather_precipitation: Optional[str] = Field(None, description="Precipitation type")
    weather_desc:      Optional[str]   = Field(None, description="Weather descriptor")
    weather_obscuration: Optional[str] = Field(None, description="Obscuration type")
    weather_other:     Optional[str]   = Field(None, description="Other weather code")

    model_config = {"json_schema_extra": {"example": {
        "airport": "EDDF", "runway": "25L", "wtc": "M",
        "wind_speed_knts": 12.0, "visibility_m": 8000.0,
        "temperature_deg": 15.0, "rwy_length": 4000.0,
        "hour_utc": 14.0, "month": 5.0, "day_of_week": 2.0,
    }}}


class PredictResponse(BaseModel):
    predicted_class:         int
    predicted_label:         str
    probability_go_around:   float
    probability_normal_landing: float
    threshold:               float
