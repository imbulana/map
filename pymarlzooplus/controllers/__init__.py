from .basic_controller import BasicMAC
from .map_controller import MAPMAC

REGISTRY = {
        "basic_mac": BasicMAC,
        "map_mac": MAPMAC,
}