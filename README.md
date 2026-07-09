# Reproducibility Package for "An Analytical Model for the Influence of Soil Moisture on Temperature Extremes in the Midlatitudes"

By: Adam Michael Bauer

Citation for journal article: Bauer, A. M., L. R. Vargas Zeppetello, C. Proistosescu. *An Analytical Model for the Influence of Soil Moisture on Temperature Extremes in the Midlatitudes*, *Journal of Climate*, 2025.

# General Package Overview
This set of codes reproduces the figures and analysis in our paper. Each code shouldn't take too long to run, unless there are simulations involved. Here's a summary table of the scripts to run and the figures they reproduce.

|**Figure Desired**| **Code to Run** | **Data Required** | **Notes** |
|------------------|-----------------|-------------------|-----------|
| Figure 1 | `01_obs.sh` | Observations, see `DATA_AVAIL.md` | - |
| Figure 2 | `02_reanalysis.sh` | Reanalysis, see `DATA_AVAIL.md` | - |
| Figure 3 | `03_param_test.sh` | None | - |
| Figure 4 | `04_time_series_short_sgp.sh` | Reanalysis, see `DATA_AVAIL.md` | - |
| Figure 5 | `05_phase_space_Td.sh` | Reanalysis and model simulations, see `DATA_AVAIL.md` | Requires running `00_model_sims.sh` |
| Figure 6 | `06_ansatz_verif.sh` | None | - |
| Figure S1 | `s01_obs_noszn.sh` | Observations, see `DATA_AVAIL.md` | - |
| Figure S2 | `s02_reanalysis_Tmax.sh` | Reanalysis, see `DATA_AVAIL.md` | - |
| Figure S3 | `s03_reanalysis_detrended.sh` | Reanalysis, see `DATA_AVAIL.md` | - |
| Figure S4 | `s04_lag_sgp.sh` | Reanalysis, see `DATA_AVAIL.md` | - |
| Figure S5 | `s05_lag_atl.sh` | Reanalysis, see `DATA_AVAIL.md` | - |
| Figure S6 | `s06_lag_dc.sh` | Reanalysis, see `DATA_AVAIL.md` | - |
| Figure S7 | `s07_time_series_long_sgp.sh` | Reanalysis, see `DATA_AVAIL.md` | - |
| Figure S8 | `s08_time_series_long_atl.sh` | Reanalysis, see `DATA_AVAIL.md` | - |
| Figure S9 | `s09_time_series_long_dc.sh` | Reanalysis, see `DATA_AVAIL.md` | - |
| Figure S10 | `s10_phase_space_F.sh` | Reanalysis and model simulations, see `DATA_AVAIL.md` | Requires running `00_model_sims.sh` | 

All codes are in the `codes` folder. You might need to run `chmod +x filename.sh` in order to run each bash script.

## Data File Pointers
This package uses reanalysis data, observational data, and generated model simulations to make the figures. **In order for this package to work, you must edit the file `codes/data_locs.py` so the code can find your data**. There is another file, `codes/figs_loc.py`, that tells the code where to save your figures. The default is just `codes/figs` in the head directory.

---

The hardware for the original author is a Linux distribution on the UIUC Keeling cluster. All codes were run using a 8-core, 64 GB ram setup with about 1 TB static storage available (though the analysis doesn't take anywhere close to this to complete).

Last edited: 7/25/2025
