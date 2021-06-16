"""Tests for profiles class."""

import pytest

from argortqcpy.profile import Profile


def test_profile_create(fake_profile):
    """Test the creation of a profile."""
    assert fake_profile is not None


def test_profile_create_from_dataset(empty_dataset, profile_from_dataset):
    """Test the creation of a profile from a dataset."""
    assert profile_from_dataset.get_property_data("PRES") is empty_dataset["PRES"][:]
    assert profile_from_dataset.get_property_data("TEMP") is empty_dataset["TEMP"][:]
    assert profile_from_dataset.get_property_data("PSAL") is empty_dataset["PSAL"][:]


@pytest.mark.parametrize("property_name", ("PRES", "TEMP", "PSAL"))
def test_property_name_validation_passes(property_name):
    """Test the validation of valid property names."""
    Profile.raise_if_not_valid_property(property_name=property_name)


@pytest.mark.parametrize("property_name", ("pressure", None, "SAL"))
def test_property_name_validation_fails(property_name):
    """Test the validation of invalid property names."""
    with pytest.raises(KeyError):
        Profile.raise_if_not_valid_property(property_name=property_name)
