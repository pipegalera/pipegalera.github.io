+++
title = "One week electricity forecast in California"
description = "CI/CD machine learning pipeline using Github Actions."
date = "2023-01-09"
weight = 1

[extra]
local_image = "/projects/weather_forecast_cover.png"
+++

# One week electricity forecast in California

![cover image](/projects/weather_forecast_cover.png)

_____________________________


This dashboard shows the latest data on electricity demand for the main 4 primary electric utility companies in California:

- Pacific Gas and Electric (PGAE)
- Southern California Edison (SCE)
- San Diego Gas and Electric (SDGE)
- Valley Electric Association (VEA)

[Live Dashboard](https://pipegalera.github.io/energy_forecasting/)

[Github repository](https://github.com/pipegalera/energy_forecasting)


The `data` (US hourly demand for electricity) comes from EIA API. The predictions are made with `XGBoost` trained via `Optuna` for hypertunning. I tracked experiments and select the best models via `MLflow`.
The visualization is made with `plotly` package.

The data, forecasting, and visualization is refreshed daily using a `Docker` image run via `Github Actions` and deployed in a `Github page`.
