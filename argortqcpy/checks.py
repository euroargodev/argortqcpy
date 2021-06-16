"""Implement Argo checks."""

from abc import ABC, abstractmethod
from typing import Optional

from netCDF4 import Dataset


class CheckBase(ABC):
    """Abstract base class for Argo checks."""

    argo_test_id: int
    argo_test_name: str

    def __init__(self, profile: Dataset, profile_previous: Optional[Dataset]) -> None:
        """Initialise the test with the relevant profile and its precursor.

        Args:
            profile: The profile of interest to be checked.
            profile_previous: The profile prior to the profile of interest.
                ``None`` if the profile of interest is the first.
        """
        self._profile = profile
        self._profile_previous = profile_previous

    @abstractmethod
    def run(self) -> Dataset:
        """Run the check.

        Return: a Dataset of checked columns with given flags
        """

    def is_required(self) -> bool:  # pylint: disable=no-self-use
        """Is the check required to run or not."""
        return True
