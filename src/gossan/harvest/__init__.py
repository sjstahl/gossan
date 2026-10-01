"""Harvest: one connector per data source."""

from gossan.harvest.base import Connector
from gossan.harvest.bc.aris import ArisIndexConnector
from gossan.harvest.bc.minfile import MinfileConnector
from gossan.harvest.bc.mto_tenure import MtoTenureConnector
from gossan.harvest.bc.rgs import RgsConnector
from gossan.harvest.national.nrcan_geophysics import NrcanGeophysicsConnector
from gossan.harvest.satellite.aster import AsterConnector
from gossan.harvest.satellite.emit import EmitConnector
from gossan.harvest.satellite.enmap import EnmapConnector

CONNECTORS: dict[str, type[Connector]] = {
    c.name: c
    for c in (
        MinfileConnector,
        MtoTenureConnector,
        ArisIndexConnector,
        RgsConnector,
        NrcanGeophysicsConnector,
        EmitConnector,
        EnmapConnector,
        AsterConnector,
    )
}
