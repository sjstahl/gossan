"""Deprecated alias for ``gossan.harvest.bc_provincial``.

The BC connectors moved to ``bc_provincial`` when ``us_federal`` was added. Old import paths
such as ``gossan.harvest.bc.wfs`` keep working: each submodule here *is* the new module (the same
object), so patching or reading either path affects the same code. New code should import from
``gossan.harvest.bc_provincial``.
"""

import sys

from gossan.harvest.bc_provincial import aris, aris_pdf, minfile, mto_tenure, rgs, wfs

for _module in (aris, aris_pdf, minfile, mto_tenure, rgs, wfs):
    sys.modules[f"{__name__}.{_module.__name__.rsplit('.', 1)[1]}"] = _module
