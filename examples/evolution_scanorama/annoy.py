"""
Stub module for annoy when it's not installed.

This allows scanorama to import without errors, falling back to exact
nearest neighbors search when approx=False is used.
"""

class AnnoyIndex:
    """Stub AnnoyIndex that raises error if actually used."""

    def __init__(self, *args, **kwargs):
        raise ImportError(
            "annoy is not installed. Use approx=False for exact nearest neighbors."
        )
