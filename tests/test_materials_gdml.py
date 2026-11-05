import numpy as np

from mu2efastsim import materials as M

GDML = """<gdml><materials>
<isotope N="27" Z="13" name="Al270x1"><atom unit="g/mole" value="26.9815"/></isotope>
<element name="Al0x2"><fraction n="1" ref="Al270x1"/></element>
<isotope N="93" Z="41" name="Nb930x3"><atom unit="g/mole" value="92.9064"/></isotope>
<element name="Nb0x4"><fraction n="1" ref="Nb930x3"/></element>
<material name="G4_Al0x5" state="solid"><MEE unit="eV" value="166"/><D unit="g/cm3" value="2.699"/><fraction n="1" ref="Al0x2"/></material>
<material name="Mix0x6" state="solid"><MEE unit="eV" value="200"/><D unit="g/cm3" value="3.0"/>
 <fraction n="0.5" ref="G4_Al0x5"/><fraction n="0.5" ref="Nb0x4"/></material>
</materials><solids/><structure/></gdml>"""


def test_from_gdml_expands_fractions_and_keeps_known(tmp_path):
    f = tmp_path / "m.gdml"
    f.write_text(GDML)
    d = M.from_gdml(str(f))
    assert d["G4_Al"] is M.ALUMINIUM
    m = d["Mix"]
    assert m.rho == 3.0 and m.I_eV == 200.0 and np.isclose(m.fractions["Al"], 0.5) and np.isclose(m.fractions["Nb"], 0.5)
    assert M.ELEMENTS["Nb"][0] == 41
    assert len(m.density_params()) == 6
