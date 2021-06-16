"""Common fixtures for testing."""

import pytest
from numpy import ma

from argortqcpy.profile import ProfileBase


class FakeProfile(ProfileBase):
    """A fake profile class created for testing."""

    _pressure = ma.MaskedArray()
    _temperature = ma.MaskedArray()
    _salinity = ma.MaskedArray()
    
    @property
    def pressure(self) -> ma.MaskedArray:
        """Return the pressure array for the profile."""
        return self._pressure

    @property
    def temperature(self) -> ma.MaskedArray:
        """Return the temperature array for the profile."""
        return self._temperature

    @property
    def salinity(self) -> ma.MaskedArray:
        """Return the salinity array for the profile."""
        return self._salinity


@pytest.fixture
def fake_profile():
    """Return a minimal profile."""

    return FakeProfile()
