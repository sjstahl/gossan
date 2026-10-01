"""Harvest: one connector per data source."""

from gossan.harvest.base import Connector
from gossan.harvest.bc.minfile import MinfileConnector

CONNECTORS: dict[str, type[Connector]] = {MinfileConnector.name: MinfileConnector}
