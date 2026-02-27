"""MineLab - Comprehensive mining and metallurgical engineering library.

Import all public functions via top-level namespace::

    import minelab as ml
    result = ml.npv(rate=0.10, cashflows=[-1000, 400, 400, 400])

Or import from specific submodules::

    from minelab.economics import npv
"""

# Build __all__ from all submodules
import minelab.data_management as _data_management
import minelab.drilling_blasting as _drilling_blasting
import minelab.economics as _economics
import minelab.environmental as _environmental
import minelab.equipment as _equipment
import minelab.geomechanics as _geomechanics
import minelab.geostatistics as _geostatistics
import minelab.hydrogeology as _hydrogeology
import minelab.mine_planning as _mine_planning
import minelab.mineral_processing as _mineral_processing
import minelab.production as _production
import minelab.resource_classification as _resource_classification
import minelab.surveying as _surveying
import minelab.underground_mining as _underground_mining
import minelab.utilities as _utilities
import minelab.ventilation as _ventilation
from minelab._version import __version__
from minelab.data_management import *  # noqa: F401, F403
from minelab.drilling_blasting import *  # noqa: F401, F403
from minelab.economics import *  # noqa: F401, F403
from minelab.environmental import *  # noqa: F401, F403
from minelab.equipment import *  # noqa: F401, F403
from minelab.geomechanics import *  # noqa: F401, F403
from minelab.geostatistics import *  # noqa: F401, F403
from minelab.hydrogeology import *  # noqa: F401, F403
from minelab.mine_planning import *  # noqa: F401, F403
from minelab.mineral_processing import *  # noqa: F401, F403
from minelab.production import *  # noqa: F401, F403
from minelab.resource_classification import *  # noqa: F401, F403
from minelab.surveying import *  # noqa: F401, F403
from minelab.underground_mining import *  # noqa: F401, F403
from minelab.utilities import *  # noqa: F401, F403
from minelab.ventilation import *  # noqa: F401, F403

__all__ = ["__version__"]
for _mod in [
    _data_management,
    _drilling_blasting,
    _economics,
    _environmental,
    _equipment,
    _geomechanics,
    _geostatistics,
    _hydrogeology,
    _mine_planning,
    _mineral_processing,
    _production,
    _resource_classification,
    _surveying,
    _underground_mining,
    _utilities,
    _ventilation,
]:
    __all__.extend(getattr(_mod, "__all__", []))
