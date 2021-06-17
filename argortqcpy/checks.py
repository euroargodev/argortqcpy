"""Implement Argo checks."""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List, Optional, Set

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

    def ensure_output_for_properties(self, property_names: List[str]) -> None:
        """Create an output flag array if it does not exist for each property."""
        for property_name in property_names:
            self.ensure_output_for_property(property_name)

    def set_output_flag_for_property(
        self,
        property_name: str,
        flag: ArgoQcFlag,
        where: Optional[np.ndarray] = None,
    ) -> None:
        """Set a flag for a given property (possibly only on some values) accounting for flag precedence."""
        self.ensure_output_for_property(property_name)
        where = slice(None) if where is None else where
        flags = self._output[property_name][where]
        for overridable_flag in FLAG_PRECEDENCE[flag]:
            flags[flags == overridable_flag.value] = flag.value

        self._output[property_name][where] = flags

    def set_output_flag_for_properties(
        self,
        property_names: List[str],
        flag: ArgoQcFlag,
        where: Optional[np.ndarray] = None,
    ) -> None:
        """Set the same flags for multiple properties."""
        for property_name in property_names:
            self.set_output_flag_for_property(property_name, flag, where=where)

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


class PressureIncreasingCheck(CheckBase):
    """Check for monotonically increasing pressure in a profile."""

    argo_test_id = 8
    argo_test_name = "Pressure increasing test"

    def run(self) -> CheckOutput:
        """Check a profile for monotonically increasing pressure."""
        pressure = self._profile.get_property_data("PRES")

        output = CheckOutput(profile=self._profile)
        output.ensure_output_for_properties(["PRES", "TEMP", "PSAL"])

        # do the first pass checking that every value is increasing
        diff = np.diff(pressure, prepend=-np.inf)  # ensures the first measurement always passes
        non_monotonic_elements = diff < 0.0

        output.set_output_flag_for_properties(
            ["PRES", "TEMP", "PSAL"],
            ArgoQcFlag.BAD,
            where=non_monotonic_elements,
        )

        # do the second pass finding consecutive constant values
        constants = diff == 0.0
        output.set_output_flag_for_properties(
            ["PRES", "TEMP", "PSAL"],
            ArgoQcFlag.BAD,
            where=constants,
        )

        # do the third pass finding any sections where it has been non-montonic and is still below the last good value

        return output
