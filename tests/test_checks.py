"""Tests for Argo checks."""

from argortqcpy.checks import CheckOutput


def test_check_is_required(fake_check):
    """Check that the base check is required."""
    assert fake_check.is_required()
