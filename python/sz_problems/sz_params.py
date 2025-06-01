#!/usr/bin/env python
# coding: utf-8

import os
basedir = ''
if "__file__" in globals(): basedir = os.path.dirname(__file__)
params_filename = os.path.join(basedir, os.path.pardir, os.path.pardir, "data", "default_params.json")


import json
with open(params_filename, "r") as fp:
    default_params = json.load(fp)


allsz_filename = os.path.join(basedir, os.path.pardir, os.path.pardir, "data", "all_sz.json")
with open(allsz_filename, "r") as fp:
    allsz_params = json.load(fp)




