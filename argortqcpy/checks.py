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

    argo_id: int
    argo_binary_id: int
    argo_name: str
    nvs_uri: str

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

    argo_id = 8
    argo_binary_id = 256
    argo_name = "Pressure increasing test"
    nvs_uri = "http://vocab.nerc.ac.uk/collection/R11/current/8/"

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
        constant = diff == 0.0
        output.set_output_flag_for_properties(
            ["PRES", "TEMP", "PSAL"],
            ArgoQcFlag.BAD,
            where=constant,
        )

        # do the third pass finding any sections where it has been non-montonic and is still below the last good value
        # this is a running maximum, constant parts mean bad values
        running_maximum = np.maximum.accumulate(pressure, axis=-1)
        running_maximum_constant = np.diff(running_maximum, prepend=-np.inf) == 0.0
        output.set_output_flag_for_properties(
            ["PRES", "TEMP", "PSAL"],
            ArgoQcFlag.BAD,
            where=running_maximum_constant,
        )

        return output


class PropertyRangeCheck(CheckBase):
    """A class which provides generalised range checking functionality."""

    def set_output_flags_for_value_outside_range(  # pylint: disable=too-many-arguments
        self,
        output: CheckOutput,
        property_name: str,
        flag: ArgoQcFlag,
        lower_limit: float = -np.inf,
        upper_limit: float = np.inf,
        properties_to_be_flagged: Optional[List[str]] = None,
    ) -> None:
        """Set the output flags based on whether a value is outside a range (inclusive of bounds).

        The flags are set either for the property given, or for a given list of properties.

        Args:
            output: An CheckOutput object to hold output flags.
            property_name: The property to check for out-of-range values.
            flag: The flag to be assigned for out-of-range values.
            lower_limit: Optional lower limit of range, defaults to negative infinity.
                Values less than this will be flagged.
            upper_limit: Optional upper limit of range, defaults to positive infinity.
                Values greater than this will be flagged.
            properties_to_be_flagged: Optional list of properties to be flagged.
                Defaults to the property specified to be checked.
        """
        properties_to_be_flagged = properties_to_be_flagged or [property_name]
        property_values = self._profile.get_property_data(property_name)

        # boolean array where values are above *or* below the specified limits
        bad_values = (property_values < lower_limit) | (property_values > upper_limit)

        output.ensure_output_for_properties(properties_to_be_flagged)
        output.set_output_flag_for_properties(properties_to_be_flagged, flag, where=bad_values)


class GlobalRangeCheck(PropertyRangeCheck):
    """Check the pressure, temperature, and salinity meet global range requirements."""

    argo_id = 6
    argo_binary_id = 64
    argo_name = "Global range test"
    nvs_uri = "http://vocab.nerc.ac.uk/collection/R11/current/6/"

    def run(self) -> CheckOutput:
        """Check a profile for correct value limits."""
        output = CheckOutput(profile=self._profile)

        self.set_output_flags_for_value_outside_range(
            output,
            "PRES",
            ArgoQcFlag.BAD,
            lower_limit=-5.0,
            properties_to_be_flagged=["PRES", "TEMP", "PSAL"],
        )

        self.set_output_flags_for_value_outside_range(
            output,
            "PRES",
            ArgoQcFlag.PROBABLY_BAD,
            lower_limit=-2.4,
            properties_to_be_flagged=["PRES", "TEMP", "PSAL"],
        )

        self.set_output_flags_for_value_outside_range(
            output,
            "TEMP",
            ArgoQcFlag.BAD,
            lower_limit=-2.5,
            upper_limit=40.0,
        )

        self.set_output_flags_for_value_outside_range(
            output,
            "PSAL",
            ArgoQcFlag.BAD,
            lower_limit=2.0,
            upper_limit=41.0,
        )

        return output
