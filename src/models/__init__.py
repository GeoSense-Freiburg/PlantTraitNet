from .builder import build_model
from .depthmapencoder import DepthMapEncoder
#from .bioclipvisual import BioCLIP
from .dinov2 import Dinov2
# from .satCLIP import SatCLIP
# from .geoCLIP import GeoCLIP
from .climplicit import Climplicit
from .planttraitnet import PlantTraitNet

__all__ = ['PlantTraitNet', 'DepthMapEncoder', 'MultiTrait_Geo', 
           'Climplicit', 'Dinov2'
            #'GeoCLIP', #To enable GeoCLIP, uncomment this line and the import statement above
            #'SatCLIP', #To enable SatCLIP, uncomment this line and the import statement above
            #'BioCLIP', #To enable BioCLIP, uncomment this line and the import statement above
           ]