"""Tests for Argo checks."""

import numpy as np
from numpy import ma

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
