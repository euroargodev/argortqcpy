"""Tests for profiles class."""

import pytest
from netCDF4 import Dataset

from argortqcpy.profile import Profile


@pytest.fixture
def empty_dataset(tmp_path):
    filepath = tmp_path / "tmp.nc"

    dataset = Dataset(filepath, mode="w")

    dataset.createVariable("PRES", "f")
    dataset.createVariable("TEMP", "f")
    dataset.createVariable("PSAL", "f")

    return dataset


@pytest.fixture
def profile_from_dataset(empty_dataset):
    return Profile(dataset=empty_dataset)


def test_profile_create(fake_profile):
    """Test the creation of a profile."""
    assert fake_profile is not None


def test_profile_create_from_dataset(empty_dataset, profile_from_dataset):
    """Test the creation of a profile from a dataset."""
    assert profile_from_dataset.pressure is empty_dataset["PRES"][:]
    assert profile_from_dataset.temperature is empty_dataset["TEMP"][:]
    assert profile_from_dataset.salinity is empty_dataset["PSAL"][:]
