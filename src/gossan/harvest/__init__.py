"""Harvest: one connector per data source, grouped by jurisdiction.

Each connector declares a ``jurisdiction`` ("bc", "us", or "any" for sources that work anywhere,
such as satellite scenes). A plain ``gossan harvest`` runs the connectors whose ``default`` is
true and whose jurisdiction matches the project's; ``--source`` names connectors explicitly.
"""

from gossan.harvest.base import Connector
from gossan.harvest.bc_provincial.access import (
    RoadsConnector,
    TransmissionLinesConnector,
)
from gossan.harvest.bc_provincial.activity import (
    CrownGrantsConnector,
    HistoricNoticesConnector,
    MajorMineAreasConnector,
    PermittedMineAreasConnector,
)
from gossan.harvest.bc_provincial.aris import ArisIndexConnector
from gossan.harvest.bc_provincial.drilling_notices import DrillingNoticesConnector
from gossan.harvest.bc_provincial.geology import (
    BedrockGeologyConnector,
    FaultsConnector,
    MineralPotentialConnector,
    QuaternaryCoverConnector,
)
from gossan.harvest.bc_provincial.land_status import (
    ConservanciesConnector,
    IndianReservesConnector,
    NationalParksConnector,
    ParksProtectedConnector,
    ReserveSitesConnector,
    TreatyLandsConnector,
)
from gossan.harvest.bc_provincial.mineral_inventory import (
    MinfileInventoryConnector,
    MinfileProductionConnector,
)
from gossan.harvest.bc_provincial.minfile import MinfileConnector
from gossan.harvest.bc_provincial.mto_tenure import MtoTenureConnector
from gossan.harvest.bc_provincial.rgs import RgsConnector, RgsSurveyAreasConnector
from gossan.harvest.national.mrdem import MrdemConnector
from gossan.harvest.national.nrcan_geophysics import NrcanGeophysicsConnector
from gossan.harvest.satellite.aster import AsterConnector
from gossan.harvest.satellite.emit import EmitConnector
from gossan.harvest.satellite.enmap import EnmapConnector
from gossan.harvest.us_federal.blm_mlrs import BlmClaimsConnector
from gossan.harvest.us_federal.mrds import MrdsConnector
from gossan.harvest.us_state.ndep_notices import NdepNoticesConnector

JURISDICTIONS = {"bc": "British Columbia (provincial)", "us": "United States (federal)"}

CONNECTORS: dict[str, type[Connector]] = {
    c.name: c
    for c in (
        MinfileConnector,
        MtoTenureConnector,
        ArisIndexConnector,
        DrillingNoticesConnector,
        RgsConnector,
        # BC: geology and mineral inventory
        BedrockGeologyConnector,
        FaultsConnector,
        QuaternaryCoverConnector,
        MineralPotentialConnector,
        MinfileProductionConnector,
        MinfileInventoryConnector,
        RgsSurveyAreasConnector,
        # BC: past and present activity
        HistoricNoticesConnector,
        PermittedMineAreasConnector,
        MajorMineAreasConnector,
        CrownGrantsConnector,
        # BC: land status and access
        ParksProtectedConnector,
        ConservanciesConnector,
        NationalParksConnector,
        IndianReservesConnector,
        TreatyLandsConnector,
        ReserveSitesConnector,
        RoadsConnector,
        TransmissionLinesConnector,
        # Canada: terrain
        MrdemConnector,
        MrdsConnector,
        BlmClaimsConnector,
        NdepNoticesConnector,
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
