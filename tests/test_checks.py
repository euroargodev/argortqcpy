"""Tests for Argo checks."""

import pytest

from argortqcpy.checks import CheckBase

class FakeCheck(CheckBase):
    """Subclass CheckBase for testing base class implementation."""

    def run(self):
        return None


@pytest.fixture
def fake_check(mocker):
    """Return an instance of the FakeCheck class."""
    return FakeCheck(mocker.sentinel.profile, mocker.sentinel.profile_previous)


def test_check_is_required(fake_check):
    """Check that the base check is required."""
    assert fake_check.is_required()
