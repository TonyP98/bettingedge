# Notes for Football Data

These notes summarize the structure of the CSV files published by football-data.co.uk. Each file contains match results and a variety of betting odds from different bookmakers. The key concepts are:

- *Results*: final scores, half-time scores and match metadata.
- *1X2 Odds*: home win, draw and away win odds from multiple bookmakers and their aggregated averages/maxima.
- *Over/Under 2.5* and *Asian Handicap* odds with available line information.
- Closing odds are identified by the suffix `C`.

The ingestion process normalises dates and times to the `Europe/Rome` timezone and validates that odds fall within sensible ranges.
