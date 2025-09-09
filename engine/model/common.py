from __future__ import annotations

from dataclasses import dataclass
import pandas as pd


@dataclass
class ScoreGrid:
    """Container for score probability grids.

    The dataframe is expected in *long* format with the following columns:
    ``match_id``, ``date``, ``home``, ``away``, ``i``, ``j`` and ``p_ij``
    representing the probability of the home team scoring ``i`` goals and the
    away team scoring ``j`` goals.
    """

    df: pd.DataFrame
