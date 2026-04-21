# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Subduction Zone Time-Dependent Suite
#
# In the following pages we implement a global suite of subduction zone models using our time-dependent solution for temperature with a dislocation creep rheology for Stokes.
#
# ```{admonition} Resolution
# In all cases the default resolution is low to allow for a quick runtime and smaller website size.  If sufficient computational resources are available set a lower `resscale` to get higher resolutions and results with sufficient accuracy.
# ```
#
# ![Global Suite Locations](../01_introduction/images/syracuselocations.png)
#
# *Figure 4.3.1 Locations of global suite of subduction zones, after [Syracuse et al., PEPI, 2010](https://doi.org/10.1016/j.pepi.2010.02.004).*
