__all__ = [
           'plot',
           'mesh',
           'ipp',
           'Myr_to_s',
           's_to_Myr',
           'nondim_to_K',
           'mmpyr_to_mps',
           'mps_to_mmpyr',
          ]

from . import plot
from . import mesh
from . import ipp

Myr_to_s = lambda a: a*1.e6*365.25*24*60*60
s_to_Myr = lambda a: a/1.e6/365.25/24/60/60
nondim_to_K = lambda T: T + 273.15
mmpyr_to_mps = lambda v: v*1.0e-3/365.25/24/60/60
mps_to_mmpyr = lambda v: v*1.0e3*365.25*24*60*60
