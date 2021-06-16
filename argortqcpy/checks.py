"""Implement Argo checks."""

from abc import ABC, abstractmethod
from enum import IntEnum
from typing import Dict, Optional, Set

from netCDF4 import Dataset
import numpy as np
from numpy import ma

from argortqcpy.profile import ProfileBase


class ArgoValueFlag(IntEnum):
    """Flags for check output."""

    NO_QC = "0"
    GOOD = "1"
    PROBABLY_GOOD = "2"
    PROBABLY_BAD = "3"
    BAD = "4"
    CHANGED = "5"
    # "6" not used
    # "7" not used
    ESTIMATED = "8"
    MISSING = "9"
    FILL_VALUE = ""


FLAG_PRECEDENCE: Dict[ArgoValueFlag, Set] = {
    ArgoValueFlag.NO_QC: set(),
    ArgoValueFlag.GOOD: {
        ArgoValueFlag.NO_QC,
    },
    ArgoValueFlag.PROBABLY_GOOD: {
        ArgoValueFlag.NO_QC,
        ArgoValueFlag.GOOD,
        ArgoValueFlag.CHANGED,
    },
    ArgoValueFlag.PROBABLY_BAD: {
        ArgoValueFlag.NO_QC,
        ArgoValueFlag.GOOD,
        ArgoValueFlag.PROBABLY_GOOD,
        ArgoValueFlag.CHANGED,
    },
    ArgoValueFlag.BAD: {
        ArgoValueFlag.NO_QC,
        ArgoValueFlag.GOOD,
        ArgoValueFlag.PROBABLY_GOOD,
        ArgoValueFlag.CHANGED,
        ArgoValueFlag.PROBABLY_BAD,
    },
    ArgoValueFlag.CHANGED: {
        ArgoValueFlag.NO_QC,
    },
    ArgoValueFlag.ESTIMATED: {
        ArgoValueFlag.NO_QC,
        ArgoValueFlag.GOOD,
        ArgoValueFlag.PROBABLY_GOOD,
    },
    ArgoValueFlag.MISSING: {
        ArgoValueFlag.NO_QC,
    },
}


class CheckOutput:
    """Class for storing the output of a single check."""

    def __init__(self, profile: ProfileBase) -> None:
        """Initialise a check output with the profile of interest, and the output data."""
        self._profile: ProfileBase = profile
        self._output: Dict[str, ma.MaskedArray] = {}

    def ensure_output_for_property(self, property_name):
        """Create an output flag array if it does not exist."""
        if property_name not in self._output:
            self._output[property_name] = ma.empty_like(self._profile.get_property_data(property_name), dtype="|S1")
            self._output[property_name][:] = ArgoValueFlag.NO_QC

    def set_output_flag_for_property(
        self, property_name: str, flag: ArgoValueFlag, where: Optional[np.ndarray] = None
    ) -> None:
        """Set a flag for a given property (possibly only on some values) accounting for flag precedence."""
        self.ensure_output_for_property(property_name)
        where = where or slice(None)
        flags = self._output[property_name][where]
        for overridable_flag in FLAG_PRECEDENCE[flag]:
            flags[flags == overridable_flag] = flag


class CheckBase(ABC):
    """Abstract base class for Argo checks."""

    argo_test_id: int
    argo_test_name: str

    def __init__(self, profile: ProfileBase, profile_previous: Optional[ProfileBase]) -> None:
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
