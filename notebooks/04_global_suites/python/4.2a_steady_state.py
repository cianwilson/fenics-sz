# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Subduction Zone Steady State Suite
#
# In the following pages we implement a global suite of subduction zone models assuming a steady state solution for temperature and a dislocation creep rheology.
#
# ```{admonition} Resolution
# In all cases the default resolution is low to allow for a quick runtime and smaller website size.  If sufficient computational resources are available set a lower `resscale` to get higher resolutions and results with sufficient accuracy.
# ```
#
# ![Global Suite Locations](../01_introduction/images/syracuselocations.png)
#
# *Figure 4.2.1 Locations of global suite of subduction zones, after [Syracuse et al., PEPI, 2010](https://doi.org/10.1016/j.pepi.2010.02.004).*
