"""Pydantic models describing football data key specifications."""
from __future__ import annotations

from pydantic import BaseModel, Field
from typing import List, Optional


class Section(BaseModel):
    """Groups of fields for a particular data section."""
    required: Optional[List[str]] = None
    optional: Optional[List[str]] = None
    required_any: Optional[List[str]] = None
    fields: Optional[List[str]] = None
    fields_any: Optional[List[str]] = None


class Aliases(BaseModel):
    """Aliases for common result columns."""
    FTHG: List[str]
    FTAG: List[str]
    FTR: List[str]
    HTHG: Optional[List[str]] = None
    HTAG: Optional[List[str]] = None
    HTR: Optional[List[str]] = None


class Constraints(BaseModel):
    """Constraints used when ingesting data."""
    min_odds: float
    max_odds: float
    max_overround_1x2: float
    timezone: str
    date_format: str


class Notes(BaseModel):
    """Additional notes describing the specification."""
    closing_suffix: str
    description: str


class FootballDataKeys(BaseModel):
    """Schema for the football_data_keys.yaml specification."""
    results: Section
    odds_1x2_pre: Section
    odds_1x2_close: Section
    ou_25_pre: Section
    ou_25_close: Section
    asian_handicap_pre: Section
    asian_handicap_close: Section
    aliases: Aliases
    constraints: Constraints
    notes: Notes

