
from .Mean import *
from .Median import *
from .GeoMed import *
from .Krum import *
from .Bulyan import *
from .signguard import *
from .DnC import *
from .LUP import *
from .LUP_SAC import *
from .Clippedclustering import *
from .Centeredclipping import * 

def aggregator(rule):
    # gradient aggregation rule
    GAR = {'Mean':mean,
           'TrMean':trimmed_mean,
           'Median':median,
           'GeoMed':geomed,
           'Multi-Krum':multi_krum,
           'Bulyan':bulyan,
           'DnC':DnC,
           'SignGuard': signguard_multiclass,
           'SignGuard-Sim': signguard_multiclass_plus1,
           'SignGuard-Dist': signguard_multiclass_plus2,
           'LUP': LUP,
           'LUP_SAC': LUP_SAC,
           "clipcluster": Clippedclustering,
           "centerclip":Centeredclipping
    }

    return GAR[rule]