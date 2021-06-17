"""Implement Argo checks."""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Optional, Set

import numpy as np
from numpy import ma

from argortqcpy.profile import ProfileBase


class ArgoQcFlag(Enum):
    """Flags for check output."""

    NO_QC = b"0"
    GOOD = b"1"
    PROBABLY_GOOD = b"2"
    PROBABLY_BAD = b"3"
    BAD = b"4"
    CHANGED = b"5"
    # "6" not used
    # "7" not used
    ESTIMATED = b"8"
    MISSING = b"9"
    FILL_VALUE = b""


FLAG_PRECEDENCE: Dict[ArgoQcFlag, Set[ArgoQcFlag]] = {
    ArgoQcFlag.NO_QC: set(),
    ArgoQcFlag.GOOD: {
        ArgoQcFlag.NO_QC,
    },
    ArgoQcFlag.PROBABLY_GOOD: {
        ArgoQcFlag.NO_QC,
        ArgoQcFlag.GOOD,
        ArgoQcFlag.CHANGED,
    },
    ArgoQcFlag.PROBABLY_BAD: {
        ArgoQcFlag.NO_QC,
        ArgoQcFlag.GOOD,
        ArgoQcFlag.PROBABLY_GOOD,
        ArgoQcFlag.CHANGED,
    },
    ArgoQcFlag.BAD: {
        ArgoQcFlag.NO_QC,
        ArgoQcFlag.GOOD,
        ArgoQcFlag.PROBABLY_GOOD,
        ArgoQcFlag.CHANGED,
        ArgoQcFlag.PROBABLY_BAD,
    },
    ArgoQcFlag.CHANGED: {
        ArgoQcFlag.NO_QC,
    },
    ArgoQcFlag.ESTIMATED: {
        ArgoQcFlag.NO_QC,
        ArgoQcFlag.GOOD,
        ArgoQcFlag.PROBABLY_GOOD,
    },
    ArgoQcFlag.MISSING: {
        ArgoQcFlag.NO_QC,
    },
}


class CheckOutput:
    """Class for storing the output of a single check."""

    def __init__(self, profile: ProfileBase) -> None:
        """Initialise a check output with the profile of interest, and the output data."""
        self._profile: ProfileBase = profile
        self._output: Dict[str, ma.MaskedArray] = {}

    def ensure_output_for_property(self, property_name: str) -> None:
        """Create an output flag array if it does not exist."""
        if property_name not in self._output:
            self._output[property_name] = ma.empty_like(self._profile.get_property_data(property_name), dtype="|S2")
            self._output[property_name][:] = ArgoQcFlag.NO_QC.value

    def set_output_flag_for_property(
        self,
        property_name: str,
        flag: ArgoQcFlag,
        where: Optional[np.ndarray] = None,
    ) -> None:
        """Set a flag for a given property (possibly only on some values) accounting for flag precedence."""
        self.ensure_output_for_property(property_name)
        where = where or slice(None)
        flags = self._output[property_name][where]
        for overridable_flag in FLAG_PRECEDENCE[flag]:
            flags[flags == overridable_flag.value] = flag.value

    def get_output_flags_for_property(self, property_name: str) -> ma.MaskedArray:
        """Return the array of flags for the given property."""
        return self._output[property_name]


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
    def run(self) -> CheckOutput:
        """Run the check.

        Return: a CheckOutput instance with checked properties flagged.
        """

    def is_required(self) -> bool:  # pylint: disable=no-self-use
        """Is the check required to run or not."""
        return True
