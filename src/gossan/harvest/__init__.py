"""Harvest: one connector per data source."""

from gossan.harvest.base import Connector
from gossan.harvest.bc.aris import ArisIndexConnector
from gossan.harvest.bc.minfile import MinfileConnector
from gossan.harvest.bc.mto_tenure import MtoTenureConnector

CONNECTORS: dict[str, type[Connector]] = {
    c.name: c for c in (MinfileConnector, MtoTenureConnector, ArisIndexConnector)
}
