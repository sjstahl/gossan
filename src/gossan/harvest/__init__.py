"""Harvest: one connector per data source, grouped by jurisdiction.

Each connector declares a ``jurisdiction`` ("bc", "us", or "any" for sources that work anywhere,
such as satellite scenes). A plain ``gossan harvest`` runs the connectors whose ``default`` is
true and whose jurisdiction matches the project's; ``--source`` names connectors explicitly.
"""

from gossan.harvest.base import Connector
from gossan.harvest.bc_provincial.aris import ArisIndexConnector
from gossan.harvest.bc_provincial.minfile import MinfileConnector
from gossan.harvest.bc_provincial.mto_tenure import MtoTenureConnector
from gossan.harvest.bc_provincial.rgs import RgsConnector
from gossan.harvest.national.nrcan_geophysics import NrcanGeophysicsConnector
from gossan.harvest.satellite.aster import AsterConnector
from gossan.harvest.satellite.emit import EmitConnector
from gossan.harvest.satellite.enmap import EnmapConnector
from gossan.harvest.us_federal.blm_mlrs import BlmClaimsConnector
from gossan.harvest.us_federal.mrds import MrdsConnector

JURISDICTIONS = {"bc": "British Columbia (provincial)", "us": "United States (federal)"}

CONNECTORS: dict[str, type[Connector]] = {
    c.name: c
    for c in (
        MinfileConnector,
        MtoTenureConnector,
        ArisIndexConnector,
        RgsConnector,
        MrdsConnector,
        BlmClaimsConnector,
        NrcanGeophysicsConnector,
        EmitConnector,
        EnmapConnector,
        AsterConnector,
    )
}


def default_sources(jurisdiction: str) -> list[str]:
    """Connector names a plain ``gossan harvest`` runs for a project in ``jurisdiction``."""
    return [
        name
        for name, c in CONNECTORS.items()
        if c.default and getattr(c, "jurisdiction", "any") in (jurisdiction, "any")
    ]
