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
# # Create a map of the global suite locations

# %%
import sys, os
basedir = ''
if "__file__" in globals(): basedir = os.path.dirname(__file__)
import pathlib
output_folder = pathlib.Path(os.path.join(basedir, "output"))
output_folder.mkdir(exist_ok=True, parents=True)

# %%
import pygmt
import pandas as pd

# %%
locs = pd.read_csv("data/syracuselocations.csv", index_col="No")

# %%
fig = pygmt.Figure()
fig.basemap(region="d", projection="W-115/25c", frame=True)
fig.coast(shorelines="1/1p,black")
cmap = pygmt.makecpt(cmap="viridis", series=[locs.index.min(), locs.index.max()])
for c, i in enumerate([val for pair in zip(list(range(0, 28)), list(range(28, 56))) for val in pair]):
#for i, data in locs.iterrows():
    data = locs.iloc[i]
    label = "{} {}".format(repr(i+1).zfill(1), data["Name"])
    if c%2==0: label += "+N2"
    fig.plot(x=data["Lat"], y=data["Lon"], fill="+z", zvalue=i, style="c0.3c", pen="black", cmap=True, label=label)
    fig.text(x=data['Lat'], y=data['Lon'], text=repr(i+1).zfill(1), offset=["{}/{}".format(data['Offx'], data['Offy'])])
    #fig.text(x=data['Lat'], y=data['Lon'], text=repr(i+1).zfill(1), fill="+z", zvalue=i)#offset=["{}/{}".format(data['Offx'], data['Offy'])])
fig.legend(position="JMC+jMC+o12.5/0+w60")
fig.show()
fig.savefig(output_folder / "syracuselocations.png")
fig.savefig(output_folder / "syracuselocations.pdf")

# %%
