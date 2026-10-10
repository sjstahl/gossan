"""BC Notice of Work (NoW) applications and permits that include surface exploration drilling.

Access method: the DataBC public WFS, layer
``pub:WHSE_MINERAL_TENURE.HSP_NOTICE_OF_WORK_PA_POINT_SV`` ("Notice of Work (NoW) Spatial
Locations - Public"); see ``gossan.harvest.bc_provincial.wfs`` for the request details. The geometry
property is ``SHAPE``. One feature is one NoW (a Mines Act permit application or amendment) as a
point; the layer holds mineral, placer, coal, quarry and gravel work.

Only notices that disturb ground for drilling are kept: the request adds
``EXP_SURFACE_DRILLING_HA>0`` (hectares of disturbance for surface exploration drilling, a field
of the NoW application). There is no hole count, depth or schedule in the layer, so a notice
says "drilling is applied for or authorised here", not "a rig is turning now".

What matters for lead scoring, and is classified at scoring time against the as-of date (see
``gossan.leads.drilling``): ``NOW_APPLICATION_STATUS_DESC`` (Approved, Rejected, Withdrawn, or an
in-progress state such as Referred), ``PERMIT_STATUS_CODE_DESC`` (Open or Closed),
``AMENDMENT_AUTH_END_DATE`` (end of the authorisation), ``NOW_APPLICATION_SUBMITTED_DATE`` and
``TENURE_NUMBER`` (the tenures the work is on, as comma-separated numbers). Dates arrive as text
with a trailing ``Z`` (for example ``2026-09-14Z``).

Caveats from the data owner: administrative amendments are not in the point layer, and some
records have no coordinates (listed in a separate CSV on the catalogue record, not harvested here).
A permit can be authorised and never used.

Catalogue record: https://catalogue.data.gov.bc.ca/dataset/notice-of-work-now-spatial-locations-public

Terms of use: Open Government Licence - British Columbia, as stated on the catalogue record.
Reuse is permitted with attribution: "Contains information licensed under the Open Government
Licence - British Columbia." The layer includes permittee names; treat them as public record
but do not use them beyond the licence.
"""

from __future__ import annotations

from gossan.harvest.bc_provincial.wfs import OGL_BC, WfsConnector


class DrillingNoticesConnector(WfsConnector):
    """NoW points with surface drilling intersecting the AOI, written to ``drilling_notices``."""

    name = "drilling_notices"
    source = "BC Mines Act Notice of Work, Spatial Locations (DataBC WFS)"
    license = OGL_BC
    type_name = "pub:WHSE_MINERAL_TENURE.HSP_NOTICE_OF_WORK_PA_POINT_SV"
    geometry_column = "SHAPE"
    extra_cql = "EXP_SURFACE_DRILLING_HA>0"
