# Data

This project uses a single source dataset: total electrical load on the Belgian
transmission grid, published by **Elia Transmission Belgium SA** via the
[Elia Open Data Portal](https://opendata.elia.be/explore/dataset/ods001).

## The dataset

- **File expected here:** `ods001.csv`
- **Coverage:** January 2015 – April 2025
- **Resolution:** quarter-hourly (15-minute) measurements, ~395,000 rows
- **Target variable:** measured total grid load (MW)

The raw CSV is **not committed to this repository** — it is large and is
redistributed under the [Elia Open Data Licence](https://opendata.elia.be/),
so it should be obtained directly from the source.

## How to obtain it

1. Open the [Elia ods001 dataset page](https://opendata.elia.be/explore/dataset/ods001).
2. Use **Export → CSV** to download the full series.
3. Save the file as `ods001.csv`.

## Where to put it

- **Google Colab (recommended):** upload `ods001.csv` to the session. The
  notebook reads it from `/content/ods001.csv`.
- **Local run:** place `ods001.csv` in this `data/` directory and update the
  load path in the notebook's data-loading cell accordingly.

> The notebook resamples the raw quarter-hourly series to hourly resolution,
> imputes the small fraction of missing values by linear interpolation, and
> performs a strictly chronological train / validation / test split. None of
> these steps mutate the source file — the original CSV is read once and left
> untouched.
