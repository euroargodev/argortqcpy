"""Implement classes for holding profile data."""

from abc import ABC, abstractmethod

from numpy import ma
from netCDF4 import Dataset


class ProfileBase(ABC):
    """Class defining the required properties for a profile."""

    @property
    @abstractmethod
    def pressure(self) -> ma.MaskedArray:
        """Return the pressure array for the profile."""

    @property
    @abstractmethod
    def temperature(self) -> ma.MaskedArray:
        """Return the temperature array for the profile."""

    @property
    @abstractmethod
    def salinity(self) -> ma.MaskedArray:
        """Return the salinity array for the profile."""


class Profile(ProfileBase):
    """Class defining a profile based on a netCDF dataset."""

    def __init__(self, dataset: Dataset) -> None:
        """Initialise a profile based on a dataset."""
        self._dataset = dataset

    @property
    def pressure(self) -> ma.MaskedArray:
        """Return the pressure array for the profile."""
        return self._dataset["PRES"][:]

    @property
    def temperature(self) -> ma.MaskedArray:
        """Return the temperature array for the profile."""
        return self._dataset["TEMP"][:]

    @property
    def salinity(self) -> ma.MaskedArray:
        """Return the salinity array for the profile."""
        return self._dataset["PSAL"][:]
