"""Implement classes for holding profile data."""

from abc import ABC, abstractmethod

from numpy import ma
from netCDF4 import Dataset


class ProfileBase(ABC):
    """Class defining the required properties for a profile."""

    valid_properties = {
        "PRES",
        "TEMP",
        "PSAL",
    }

    @classmethod
    def raise_if_not_valid_property(cls, property_name):
        """Check that a given property name is valid."""
        if property_name not in cls.valid_properties:
            raise KeyError(f"{property_name}: not a valid property for Profile.")

    @abstractmethod
    def get_property_data(self, property_name) -> ma.MaskedArray:
        """Return the array of property data from the profile."""


class Profile(ProfileBase):
    """Class defining a profile based on a netCDF dataset."""

    def __init__(self, dataset: Dataset) -> None:
        """Initialise a profile based on a dataset."""
        self._dataset = dataset

    def get_property_data(self, property_name) -> ma.MaskedArray:
        """Return the array of property data from the profile."""
        self.raise_if_not_valid_property(property_name)
        return self._dataset[property_name][:]
