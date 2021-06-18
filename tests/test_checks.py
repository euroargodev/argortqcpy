"""Tests for Argo checks."""

import numpy as np
from numpy import ma
import pytest

from argortqcpy.checks import ArgoQcFlag, CheckOutput


def test_check_is_required(fake_check):
    """Check that the base check is required."""
    assert fake_check.is_required()


def test_output_ensure_output_for_property(profile_from_dataset):
    """Test ensuring a property is given an output array."""
    output = CheckOutput(profile=profile_from_dataset)

    output.ensure_output_for_property("PRES")
    flags = output.get_output_flags_for_property("PRES")

    assert flags is not None
    assert isinstance(flags, ma.MaskedArray)
    assert np.all(flags == ArgoQcFlag.NO_QC.value)


def test_output_set_output_flag_for_property(profile_from_dataset):
    """Test setting a flag for a given property."""
    output = CheckOutput(profile=profile_from_dataset)

    output.ensure_output_for_property("PRES")
    output.set_output_flag_for_property("PRES", ArgoQcFlag.GOOD)
    flags = output.get_output_flags_for_property("PRES")

    assert flags is not None
    assert isinstance(flags, ma.MaskedArray)
    assert np.all(flags == ArgoQcFlag.GOOD.value)


def test_output_set_output_flag_for_property_where(profile_from_dataset):
    """Test setting a flag for a given property for a limited set of indices."""
    output = CheckOutput(profile=profile_from_dataset)

    output.ensure_output_for_property("PRES")
    output.set_output_flag_for_property("PRES", ArgoQcFlag.GOOD, where=slice(None, 2))
    flags = output.get_output_flags_for_property("PRES")

    assert flags is not None
    assert isinstance(flags, ma.MaskedArray)
    assert np.all(flags[:2] == ArgoQcFlag.GOOD.value)
    assert np.all(flags[2:] == ArgoQcFlag.NO_QC.value)


def test_output_set_output_flag_for_property_where_array(profile_from_dataset):
    """Test setting a flag for a given property for indices limited by array."""
    output = CheckOutput(profile=profile_from_dataset)

    where = np.full_like(profile_from_dataset.get_property_data("PRES"), False, dtype=bool)
    where[0] = True
    where[-1] = True

    output.ensure_output_for_property("PRES")
    output.set_output_flag_for_property("PRES", ArgoQcFlag.GOOD, where=where)
    flags = output.get_output_flags_for_property("PRES")

    assert flags is not None
    assert isinstance(flags, ma.MaskedArray)
    assert np.all(flags[0] == ArgoQcFlag.GOOD.value)
    assert np.all(flags[1:-1] == ArgoQcFlag.NO_QC.value)
    assert np.all(flags[-1] == ArgoQcFlag.GOOD.value)


@pytest.mark.parametrize(
    "lower,higher",
    (
        (ArgoQcFlag.GOOD, ArgoQcFlag.BAD),
        (ArgoQcFlag.GOOD, ArgoQcFlag.PROBABLY_GOOD),
        (ArgoQcFlag.PROBABLY_BAD, ArgoQcFlag.BAD),
    ),
)
def test_output_set_output_flag_for_property_with_precendence(profile_from_dataset, lower, higher):
    """Test setting a flag for a given property for a limited set of indices."""
    output = CheckOutput(profile=profile_from_dataset)

    output.ensure_output_for_property("PRES")
    output.set_output_flag_for_property("PRES", lower, where=slice(None, 2))
    output.set_output_flag_for_property("PRES", higher, where=slice(None, 1))
    output.set_output_flag_for_property("PRES", lower, where=slice(None, 2))
    flags = output.get_output_flags_for_property("PRES")

    assert flags is not None
    assert isinstance(flags, ma.MaskedArray)
    assert np.all(flags[:1] == higher.value)
    assert np.all(flags[1:2] == lower.value)
    assert np.all(flags[2:] == ArgoQcFlag.NO_QC.value)
