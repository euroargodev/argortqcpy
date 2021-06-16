"""Common fixtures for testing."""

import pytest
from numpy import ma
from netCDF4 import Dataset

from argortqcpy.checks import CheckBase
from argortqcpy.profile import ProfileBase, Profile


class FakeProfile(ProfileBase):
    """A fake profile class created for testing."""

    def __init__(self):
        """Initialise some empty data for access."""
        self._data = {
            "PRES": ma.MaskedArray(),
            "TEMP": ma.MaskedArray(),
            "PSAL": ma.MaskedArray(),
        }

    def get_property_data(self, property_name) -> ma.MaskedArray:
        """Retrieve the data from the internal dict."""
        return self._data[property_name]


class FakeCheck(CheckBase):
    """Subclass CheckBase for testing base class implementation."""

    def run(self):
        """Override the run method for the fake check."""
        return None


@pytest.fixture(name="empty_dataset")
def fixture_empty_dataset(tmp_path):
    """Create an empty dataset with sufficient variables for testing."""
    filepath = tmp_path / "tmp.nc"

    dataset = Dataset(filepath, mode="w")

    dataset.createVariable("PRES", "f")
    dataset.createVariable("TEMP", "f")
    dataset.createVariable("PSAL", "f")

    return dataset


@pytest.fixture
def profile_from_dataset(empty_dataset):
    """Create a profile based on the empty dataset."""
    return Profile(dataset=empty_dataset)


@pytest.fixture
def fake_profile():
    """Return a minimal profile."""
    return FakeProfile()


@pytest.fixture
def fake_check(mocker):
    """Return an instance of the FakeCheck class."""
    return FakeCheck(mocker.sentinel.profile, mocker.sentinel.profile_previous)
