"""Tests for the plugin registry that backs `Structure`/`Molecule` I/O dispatch."""

from __future__ import annotations

import warnings

import pytest

from pymatgen.core import Lattice, Molecule, Structure
from pymatgen.io.registry import (
    StructureFormat,
    filter_kwargs,
    get_molecule_format,
    get_structure_format,
    list_structure_formats,
    register_structure_format,
    unregister_structure_format,
)


@pytest.fixture
def simple_structure():
    return Structure(Lattice.cubic(3), ["Si", "Si"], [[0, 0, 0], [0.5, 0.5, 0.5]])


@pytest.fixture
def simple_molecule():
    return Molecule(["C", "O"], [[0, 0, 0], [1.1, 0, 0]])


class TestRegistration:
    def test_register_and_get(self):
        marker: list = []

        def _read(s, **_):
            marker.append(("read_str", s))
            return Structure(Lattice.cubic(3), ["Si"], [[0, 0, 0]])

        fmt = StructureFormat(name="testfmt-roundtrip", patterns=("*.testfmt",), read_str=_read)
        try:
            register_structure_format(fmt)
            assert get_structure_format(name="testfmt-roundtrip") is fmt
            assert get_structure_format(filename="x.testfmt") is fmt
        finally:
            unregister_structure_format("testfmt-roundtrip")

    def test_unregister(self):
        fmt = StructureFormat(name="testfmt-unreg", patterns=("*.unreg",))
        register_structure_format(fmt)
        assert any(f.name == "testfmt-unreg" for f in list_structure_formats())
        unregister_structure_format("testfmt-unreg")
        assert not any(f.name == "testfmt-unreg" for f in list_structure_formats())
        # Unregistering twice is a no-op.
        unregister_structure_format("testfmt-unreg")

    def test_register_case_insensitive_name(self):
        fmt = StructureFormat(name="TestUpperCase", patterns=())
        try:
            register_structure_format(fmt)
            # Lookup is case-insensitive on the name.
            assert get_structure_format(name="testuppercase") is fmt
            assert get_structure_format(name="TESTUPPERCASE") is fmt
        finally:
            unregister_structure_format("TestUpperCase")


class TestBuiltinDispatch:
    """Built-in formats should resolve via the lazy-import map (no manual registration)."""

    @pytest.mark.parametrize("fmt", ["cif", "poscar", "cssr", "xsf", "res", "pwmat", "json", "yaml"])
    def test_structure_fmt_lookup(self, fmt):
        handler = get_structure_format(name=fmt)
        assert handler.name == fmt

    @pytest.mark.parametrize(
        ("filename", "expected"),
        [
            ("foo.cif", "cif"),
            # Legacy behavior: *.cif* glob also matches .mcif (reading mcif uses
            # the same CIF parser; only writing distinguishes magnetic CIF).
            ("foo.mcif", "cif"),
            ("POSCAR", "poscar"),
            ("CONTCAR", "poscar"),
            ("foo.vasp", "poscar"),
            ("CHGCAR", "chgcar"),
            ("LOCPOT", "chgcar"),
            ("vasprun.xml", "vasprun"),
            ("foo.cssr", "cssr"),
            ("foo.json", "json"),
            ("foo.yaml", "yaml"),
            ("foo.yml", "yaml"),
            ("foo.xsf", "xsf"),
            ("foo.res", "res"),
            ("rndstr.in", "mcsqs"),
            ("CTRL.foo", "lmto"),
            ("foo.config", "pwmat"),
            ("foo.pwmat", "pwmat"),
        ],
    )
    def test_structure_filename_inference(self, filename, expected):
        handler = get_structure_format(filename=filename)
        assert handler.name == expected

    @pytest.mark.parametrize("fmt", ["xyz", "gaussian", "gjf", "g09", "json", "yaml"])
    def test_molecule_fmt_lookup(self, fmt):
        handler = get_molecule_format(name=fmt)
        assert handler.name == fmt

    @pytest.mark.parametrize(
        ("filename", "expected"),
        [
            ("foo.xyz", "xyz"),
            ("foo.gjf", "gaussian"),
            ("foo.g09", "gaussian"),
            ("foo.out", "gaussian-out"),
            ("foo.json", "json"),
            ("foo.yaml", "yaml"),
        ],
    )
    def test_molecule_filename_inference(self, filename, expected):
        handler = get_molecule_format(filename=filename)
        assert handler.name == expected


class TestErrorMessages:
    def test_unknown_fmt(self):
        with pytest.raises(ValueError, match="badformat"):
            get_structure_format(name="badformat")

    def test_no_args(self):
        with pytest.raises(ValueError, match="Either fmt or filename"):
            get_structure_format()

    def test_unknown_filename(self):
        with pytest.raises(ValueError, match="Unrecognized extension"):
            get_structure_format(filename="whatever.unknownext")

    def test_molecule_unknown_filename(self):
        with pytest.raises(ValueError, match="Cannot determine"):
            get_molecule_format(filename="whatever.unknownext")


class TestFilterKwargs:
    def test_filters_unsupported(self):
        # Signature has only a, b — `bogus` should be dropped with a warning.
        def f(a, b=1):
            pass

        with pytest.warns(UserWarning, match="bogus"):
            filtered = filter_kwargs(f, {"a": 1, "bogus": 2})
        assert filtered == {"a": 1}

    def test_passes_through_with_var_keyword(self):
        # `**kwargs` opt-out: nothing should be filtered or warned about.
        def f(**kwargs):
            pass

        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            result = filter_kwargs(f, {"a": 1, "b": 2})
        assert result == {"a": 1, "b": 2}


class TestRoundtripsViaRegistry:
    """End-to-end checks that the public `Structure`/`Molecule` API still works
    after dispatch goes through the registry."""

    @pytest.mark.parametrize("fmt", ["cif", "poscar", "cssr", "json", "yaml", "xsf", "res", "pwmat"])
    def test_structure_roundtrip(self, fmt, simple_structure):
        s_str = simple_structure.to(fmt=fmt)
        assert isinstance(s_str, str)
        assert s_str
        s2 = Structure.from_str(s_str, fmt=fmt)
        assert s2.formula == simple_structure.formula

    @pytest.mark.parametrize("fmt", ["xyz", "gjf", "json", "yaml"])
    def test_molecule_roundtrip(self, fmt, simple_molecule):
        m_str = simple_molecule.to(fmt=fmt)
        assert isinstance(m_str, str)
        assert m_str
        m2 = Molecule.from_str(m_str, fmt=fmt)
        assert m2.formula == simple_molecule.formula


class TestPluginOverride:
    """A user-supplied plugin should be able to override a built-in format and the
    public `Structure.from_str` dispatch should use the new handler."""

    def test_user_plugin_overrides_builtin(self, simple_structure):
        call_log: list[str] = []
        original = get_structure_format(name="cif")

        def _intercept_read_str(s, **kwargs):
            call_log.append("intercepted")
            return original.read_str(s, **kwargs)  # type:ignore[misc]

        replacement = StructureFormat(
            name="cif",
            patterns=("*.cif*",),
            read_str=_intercept_read_str,
            write_str=original.write_str,
            write_file=original.write_file,
        )
        try:
            register_structure_format(replacement)
            cif = simple_structure.to(fmt="cif")
            Structure.from_str(cif, fmt="cif")
            assert call_log == ["intercepted"]
        finally:
            # Reinstate the built-in handler so other tests don't see leakage.
            register_structure_format(original)
