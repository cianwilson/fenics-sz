#!/usr/bin/env python
# coding: utf-8

import os
basedir = ''
if "__file__" in globals(): basedir = os.path.dirname(__file__)
params_filename = os.path.join(basedir, os.path.pardir, os.path.pardir, "data", "default_params.json")


import json
with open(params_filename, "r") as fp:
    default_params = json.load(fp)


from mpi4py import MPI
if __name__ == "__main__":
    if MPI.COMM_WORLD.rank == 0:
        print("{:<35} {:<10}".format('Key','Value'))
        print("-"*45)
        for k, v in default_params.items():
            print("{:<35} {:<10}".format(k, v))


allsz_filename = os.path.join(basedir, os.path.pardir, os.path.pardir, "data", "all_sz.json")
with open(allsz_filename, "r") as fp:
    allsz_params = json.load(fp)


if __name__ == "__main__":
    if MPI.COMM_WORLD.rank == 0:
        print("{}".format('Name'))
        print("-"*30)
        for k in allsz_params.keys():
            print("{}".format(k,))


if __name__ == "__main__":
    names = ['01_Alaska_Peninsula', '19_N_Antilles']
    if MPI.COMM_WORLD.rank == 0:
        for name in names:
            print("{}:".format(name))
            print("{:<35} {:<10}".format('Key','Value'))
            print("-"*100)
            for k, v in allsz_params[name].items():
                if v is not None: print("{:<35} {}".format(k, v))
            print("="*100)


if __name__ == "__main__" and "__file__" not in globals():
    from ipylab import JupyterFrontEnd
    app = JupyterFrontEnd()
    app.commands.execute('docmanager:save')
    get_ipython().system('jupyter nbconvert --TagRemovePreprocessor.enabled=True --TagRemovePreprocessor.remove_cell_tags="[\'main\', \'ipy\']" --TemplateExporter.exclude_markdown=True --TemplateExporter.exclude_input_prompt=True --TemplateExporter.exclude_output_prompt=True --NbConvertApp.export_format=script --ClearOutputPreprocessor.enabled=True --FilesWriter.build_directory=../../python/sz_problems --NbConvertApp.output_base=sz_base 3.2a_sz_base.ipynb')




