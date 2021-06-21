"""Tests for Argo checks."""

import numpy as np
from numpy import ma
import pytest

import argortqcpy.profile
from argortqcpy.checks import ArgoQcFlag, CheckOutput, PressureIncreasingCheck


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
    assert np.all(flags == ArgoQcFlag.GOOD.value)


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
    output.set_output_flag_for_property("PRES", ArgoQcFlag.PROBABLY_GOOD, where=slice(None, 2))
    flags = output.get_output_flags_for_property("PRES")

    assert flags is not None
    assert isinstance(flags, ma.MaskedArray)
    assert np.all(flags[:2] == ArgoQcFlag.PROBABLY_GOOD.value)
    assert np.all(flags[2:] == ArgoQcFlag.GOOD.value)


def test_output_set_output_flag_for_property_where_array(profile_from_dataset):
    """Test setting a flag for a given property for indices limited by array."""
    output = CheckOutput(profile=profile_from_dataset)

    where = np.full_like(profile_from_dataset.get_property_data("PRES"), False, dtype=bool)
    where[0] = True
    where[-1] = True

    output.ensure_output_for_property("PRES")
    output.set_output_flag_for_property("PRES", ArgoQcFlag.PROBABLY_GOOD, where=where)
    flags = output.get_output_flags_for_property("PRES")

    assert flags is not None
    assert isinstance(flags, ma.MaskedArray)
    assert np.all(flags[0] == ArgoQcFlag.PROBABLY_GOOD.value)
    assert np.all(flags[1:-1] == ArgoQcFlag.GOOD.value)
    assert np.all(flags[-1] == ArgoQcFlag.PROBABLY_GOOD.value)


@pytest.mark.parametrize(
    "lower,higher",
    (
        (ArgoQcFlag.PROBABLY_GOOD, ArgoQcFlag.BAD),
        (ArgoQcFlag.PROBABLY_GOOD, ArgoQcFlag.PROBABLY_BAD),
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
    assert np.all(flags[2:] == ArgoQcFlag.GOOD.value)


@pytest.mark.parametrize(
    "pressure_values",
    (
        range(10),
        [1, 3, 5, 10, 100],
        [0, 2, 2.5, 6.85],
    ),
)
def test_pressure_increasing_check_all_pass(mocker, pressure_values):
    """Test that the pressure increasing test succeeds."""
    profile = mocker.patch.object(argortqcpy.profile, "Profile")
    profile.get_property_data = mocker.Mock(return_value=ma.masked_array(pressure_values))

    pic = PressureIncreasingCheck(profile, None)
    output = pic.run()

    assert np.all(output.get_output_flags_for_property("PRES").data == ArgoQcFlag.GOOD.value)


@pytest.mark.parametrize(
    "pressure_values,expected",
    (
        (
            [0, 2, 1, 5],
            [ArgoQcFlag.GOOD.value, ArgoQcFlag.GOOD.value, ArgoQcFlag.BAD.value, ArgoQcFlag.GOOD.value],
        ),
    ),
)
def test_pressure_increasing_check_some_bad(mocker, pressure_values, expected):
    """Test that the pressure increasing works when some values are bad."""
    profile = mocker.patch.object(argortqcpy.profile, "Profile")
    profile.get_property_data = mocker.Mock(return_value=ma.masked_array(pressure_values))

    pic = PressureIncreasingCheck(profile, None)
    output = pic.run()

    assert np.all(output.get_output_flags_for_property("PRES").data == expected)


@pytest.mark.parametrize(
    "pressure_values,expected",
    (
        (
            [0] * 4,
            [ArgoQcFlag.GOOD.value, ArgoQcFlag.BAD.value, ArgoQcFlag.BAD.value, ArgoQcFlag.BAD.value],
        ),
        (
            [0, 1, 1, 2],
            [ArgoQcFlag.GOOD.value, ArgoQcFlag.GOOD.value, ArgoQcFlag.BAD.value, ArgoQcFlag.GOOD.value],
        ),
    ),
)
def test_pressure_increasing_check_some_constants(mocker, pressure_values, expected):
    """Test that the pressure increasing works when some values are constant."""
    profile = mocker.patch.object(argortqcpy.profile, "Profile")
    profile.get_property_data = mocker.Mock(return_value=ma.masked_array(pressure_values))

    pic = PressureIncreasingCheck(profile, None)
    output = pic.run()

    assert np.all(output.get_output_flags_for_property("PRES").data == expected)


@pytest.mark.parametrize(
    "pressure_values,expected",
    (
        (
            [0, 1, 2, 1, 1.5, 3, 5],
            [
                ArgoQcFlag.GOOD.value,
                ArgoQcFlag.GOOD.value,
                ArgoQcFlag.GOOD.value,
                ArgoQcFlag.BAD.value,
                ArgoQcFlag.BAD.value,
                ArgoQcFlag.GOOD.value,
                ArgoQcFlag.GOOD.value,
            ],
        ),
        (
            [
                [0, 1, 2, 3],
                [0, 1, 0, 1],
            ],
            [
                [ArgoQcFlag.GOOD.value, ArgoQcFlag.GOOD.value, ArgoQcFlag.GOOD.value, ArgoQcFlag.GOOD.value],
                [ArgoQcFlag.GOOD.value, ArgoQcFlag.GOOD.value, ArgoQcFlag.BAD.value, ArgoQcFlag.BAD.value],
            ],
        ),
    ),
)
def test_pressure_increasing_check_some_decreasing(mocker, pressure_values, expected):
    """Test that the pressure increasing works when some values are decreasing."""
    profile = mocker.patch.object(argortqcpy.profile, "Profile")
    profile.get_property_data = mocker.Mock(return_value=ma.masked_array(pressure_values))

    pic = PressureIncreasingCheck(profile, None)
    output = pic.run()

    assert np.all(output.get_output_flags_for_property("PRES").data == expected)
