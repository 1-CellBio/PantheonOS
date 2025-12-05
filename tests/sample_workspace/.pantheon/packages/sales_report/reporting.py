class SalesReportPackage:
    """Sales report helpers used throughout the suite."""

    def generate(self, date: str, region: str | None = None):
        """Return a deterministic payload describing the requested run."""
        return {
            "status": "ready",
            "date": date,
            "region": region or "ALL",
        }
