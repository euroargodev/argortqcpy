"""Tests for Argo range checks."""

import numpy as np
from numpy import ma
import pytest

import argortqcpy.profile
from argortqcpy.checks import ArgoQcFlag, PropertyRangeCheck


class FakePropertyRangeCheck(PropertyRangeCheck):
    """A fake property range check class with a notional run method."""

    def run(self):
        """An empty run method to override the abstract method."""


@pytest.mark.parametrize(
    "lower,upper,expected",
    (
        (2.5, 4.5, (True, True, False, False, True)),
        (None, 4.5, (False, False, False, False, True)),
        (None, None, (False, False, False, False, False)),
        (4.5, None, (True, True, True, True, False)),
    ),
)
def test_property_range_check(mocker, lower, upper, expected):
    """Test that the property range check sets the correct data."""
    property_name = "TEST_PROPERTY"
    property_values = np.array((1, 2, 3, 4, 5))
    profile = mocker.patch.object(argortqcpy.profile, "Profile")
    profile.get_property_data = mocker.Mock(return_value=ma.masked_array(property_values))
    output = mocker.Mock()
    prc = FakePropertyRangeCheck(profile, None)

    kwargs_in = {}
    if lower is not None:
        kwargs_in["lower_limit"] = lower
    if upper is not None:
        kwargs_in["upper_limit"] = upper

    prc.set_output_flags_for_value_outside_range(output, property_name, ArgoQcFlag.BAD, **kwargs_in)

    profile.get_property_data.assert_called_once_with(property_name)
    output.ensure_output_for_properties.assert_called_once_with([property_name])
    output.set_output_flag_for_properties.assert_called_once()
    assert output.set_output_flag_for_properties.call_args[0] == ([property_name], ArgoQcFlag.BAD)

    kwargs = output.set_output_flag_for_properties.call_args[1]
    assert "where" in kwargs
    np.testing.assert_equal(kwargs["where"], ma.masked_array(expected))
