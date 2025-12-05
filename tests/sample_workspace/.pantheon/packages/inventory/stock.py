class InventoryPackage:
    """Plain package that exercises the docstring auto-discovery flow."""

    def restock(self, product: str, delta: int):
        """Summarize an inventory adjustment."""
        return {
            "product": product,
            "delta": delta,
            "status": "restocked",
        }
