# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: dolfinx-env
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Subduction Zone Global Suites
#

# %% [markdown]
# In the following section we present a full global suite of subduction zone thermal models as discussed in [van Keken & Wilson, PEPS, 2023 (III)](https://doi.org/10.1186/s40645-023-00589-5).  This suite, originally proposed by [Syracuse et al., PEPI, 2010](https://doi.org/10.1016/j.pepi.2010.02.004), represents a global compilation of two-dimensional subduction zone transects at the locations seen in Figure 4.1.1.
#
# ![Global Suite Locations](../01_introduction/images/syracuselocations.png)
#
# *Figure 4.1.1 Locations of global suite of subduction zones, after [Syracuse et al., PEPI, 2010](https://doi.org/10.1016/j.pepi.2010.02.004).*
#
# We present two versions of the global suite, the first [steady state](4.2a_steady_state.ipynb) and the second [time-dependent](4.3a_time_dependent.ipynb).  Both use a non-linear temperature and strain-rate dependent dislocation creep rheology.
