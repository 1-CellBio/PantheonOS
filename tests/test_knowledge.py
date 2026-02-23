"""
KnowledgeManager Black-box Tests

Test Principles:
- Only test public interfaces (API)
- Don't test internal state (private attributes)
- Verify input/output/functional behavior
- Test data located at tests/data/knowledge/

Test modes (controlled via environment variable QDRANT_LOCATION):
- Not set or local path: Local Path mode (default, data persists)
- ":memory:": In-memory mode (no persistence, fastest)
- "http://...": URL mode (remote Qdrant server, full async support)

Examples:
  # Default local mode
  pytest tests/test_knowledge.py

  # In-memory mode
  QDRANT_LOCATION=:memory: pytest tests/test_knowledge.py

  # URL mode
  QDRANT_LOCATION=http://localhost:6333 pytest tests/test_knowledge.py
"""

import asyncio
import tempfile
from pathlib import Path

import pytest

# Check if llama_index is available
try:
    import llama_index.core
    LLAMA_INDEX_AVAILABLE = True
except ImportError:
    LLAMA_INDEX_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not LLAMA_INDEX_AVAILABLE,
    reason="llama_index.core not installed"
)

# Load environment variables
from dotenv import load_dotenv

load_dotenv()

from pantheon.toolsets.knowledge.knowledge_manager import KnowledgeToolSet

# Test data path
TEST_DATA_DIR = Path(__file__).parent / "data" / "knowledge"
SAMPLE_FILE = TEST_DATA_DIR / "sample.md"
DOCS_DIR = TEST_DATA_DIR / "docs"


def get_test_config(tmp_dir: str) -> dict:
    """
    Return test configuration based on environment variables

    Environment variables:
    - QDRANT_LOCATION: Override qdrant.location
      - ":memory:" : Pure in-memory mode
      - local path : Persist to local file
      - URL : Connect to remote Qdrant server (e.g. "http://localhost:6333")

    If environment variable is not set, defaults to local path mode
    """
    # Base configuration (all tests need independent storage_path)
    config = {"knowledge": {"storage_path": str(Path(tmp_dir) / "storage")}}

    # Environment variables will be automatically overridden in load_config()
    # No need to handle manually here, just provide base configuration
    return config


async def test_knowledge_full_workflow():
    """Complete workflow black-box test - tests all public interfaces"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        import yaml

        # Configuration
        config_path = Path(tmp_dir) / "test_config.yaml"
        config = get_test_config(tmp_dir)

        with open(config_path, "w") as f:
            yaml.dump(config, f)

        # Create KM instance
        km = KnowledgeToolSet(config_path=str(config_path))
        await km.run_setup()

        # ========== Test 1: Collection Management ==========
        # List (initially empty)
        result = await km.list_collections()
        assert result["success"] == True
        assert result["total"] == 0
        assert isinstance(result["collections"], list)

        # Create collection
        result = await km.create_collection(
            name="Test Collection", description="Test collection"
        )
        assert result["success"] == True
        assert "collection" in result
        assert result["collection"]["name"] == "Test Collection"
        assert result["collection"]["description"] == "Test collection"
        collection_id = result["collection"]["id"]

        # List (should have 1)
        result = await km.list_collections()
        assert result["total"] == 1
        assert result["collections"][0]["id"] == collection_id

        # ========== Test 2: Source Management (Files) ==========
        if not SAMPLE_FILE.exists():
            pytest.skip(f"Test file does not exist: {SAMPLE_FILE}")

        # Add file source
        result = await km.add_sources(
            collection_id=collection_id,
            sources={
                "type": "file",
                "path": str(SAMPLE_FILE),
                "name": "Sample File",
            },
        )
        assert result["success"] == True
        assert "source_ids" in result
        source_id_1 = result["source_ids"][0]

        # Wait for processing to complete
        await asyncio.sleep(8)

        # List sources
        result = await km.list_sources(collection_id=collection_id)
        assert result["success"] == True
        assert result["total"] >= 1

        # Verify source information
        source = next(s for s in result["sources"] if s["id"] == source_id_1)
        assert source["name"] == "Sample File"
        assert source["type"] == "file"
        assert source["status"] in ["active", "processing"]

        # ========== Test 3: Source Management (Directories) ==========
        if DOCS_DIR.exists():
            result = await km.add_sources(
                collection_id=collection_id,
                sources={
                    "type": "folder",
                    "path": str(DOCS_DIR),
                    "name": "Docs Folder",
                },
            )
            assert result["success"] == True
            source_id_2 = result["source_ids"][0]

            # Wait for processing
            await asyncio.sleep(8)

            # Verify source count
            result = await km.list_sources(collection_id=collection_id)
            assert result["total"] == 2

        # ========== Test 4: Basic Retrieval ==========
        result = await km.search_knowledge(
            query="LlamaIndex",
            collection_ids=[collection_id],
            top_k=3,
            use_hybrid=False,
        )
        assert result["success"] == True
        assert "results" in result
        assert isinstance(result["results"], list)
        assert "searched_collections" in result
        assert collection_id in result["searched_collections"]

        # Verify result format
        if len(result["results"]) > 0:
            first_result = result["results"][0]
            assert "text" in first_result
            assert "score" in first_result
            assert isinstance(first_result["score"], (int, float))

        # ========== Test 5: Hybrid Retrieval ==========
        result = await km.search_knowledge(
            query="semantic chunking",
            collection_ids=[collection_id],
            top_k=3,
            use_hybrid=True,
        )
        assert result["success"] == True
        assert "results" in result

        # ========== Test 6: Reranking ==========
        result = await km.search_knowledge(
            query="document indexing",
            collection_ids=[collection_id],
            top_k=5,
            use_hybrid=True,
        )
        assert result["success"] == True
        # After reranking, result count may be <= top_k
        assert len(result["results"]) <= 5

        # ========== Test 7: Chat Configuration ==========
        chat_id = "test_chat_001"

        # Get configuration
        result = await km.get_chat_knowledge(chat_id=chat_id)
        assert result["success"] == True
        assert result["config"]["chat_id"] == chat_id
        assert result["config"]["auto_search"] == False
        assert isinstance(result["config"]["active_collection_ids"], list)

        # Enable collection
        result = await km.enable_collection(
            chat_id=chat_id, collection_id=collection_id
        )
        assert result["success"] == True
        assert collection_id in result["config"]["active_collection_ids"]

        # Set auto search
        result = await km.set_auto_search(chat_id=chat_id, enabled=True)
        assert result["success"] == True
        assert result["config"]["auto_search"] == True

        # Verify configuration persistence
        result = await km.get_chat_knowledge(chat_id=chat_id)
        assert result["config"]["auto_search"] == True
        assert collection_id in result["config"]["active_collection_ids"]

        # ========== Test 8: Chat-bound Retrieval ==========
        result = await km.search_knowledge(query="test query", chat_id=chat_id, top_k=2)
        assert result["success"] == True
        assert collection_id in result["searched_collections"]
        # Chat-bound search should only search activated collections
        assert len(result["searched_collections"]) == 1

        # Disable collection
        result = await km.disable_collection(
            chat_id=chat_id, collection_id=collection_id
        )
        assert result["success"] == True
        assert collection_id not in result["config"]["active_collection_ids"]

        # ========== Test 9: Error Handling ==========
        # Non-existent collection
        result = await km.search_knowledge(
            query="test", collection_ids=["col_nonexistent"], top_k=3
        )
        # Should return empty result or error, but should not crash
        assert result["success"] == True or "error" in result

        # Non-existent chat
        result = await km.get_chat_knowledge(chat_id="nonexistent_chat")
        # Should return default configuration
        assert result["success"] == True
        assert result["config"]["chat_id"] == "nonexistent_chat"

        # ========== Test 10: Cleanup Operations ==========
        # Delete source
        result = await km.remove_source(
            collection_id=collection_id, source_id=source_id_1
        )
        assert result["success"] == True

        # Verify deletion
        result = await km.list_sources(collection_id=collection_id)
        remaining_sources = [s for s in result["sources"] if s["id"] == source_id_1]
        assert len(remaining_sources) == 0

        # Delete collection
        result = await km.delete_collection(collection_id=collection_id)
        assert result["success"] == True

        # Verify collection has been deleted
        result = await km.list_collections()
        assert result["total"] == 0


async def test_knowledge_collection_crud():
    """Test Collection CRUD Interface"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        import yaml

        config_path = Path(tmp_dir) / "test_config.yaml"
        config = get_test_config(tmp_dir)

        with open(config_path, "w") as f:
            yaml.dump(config, f)

        km = KnowledgeToolSet(config_path=str(config_path))
        await km.run_setup()

        try:
            # Create
            result = await km.create_collection(
                name="Test Collection", description="Test"
            )
            assert result["success"] == True
            collection_id = result["collection"]["id"]

            # List
            result = await km.list_collections()
            assert result["total"] == 1
            assert result["collections"][0]["name"] == "Test Collection"

            # Delete
            result = await km.delete_collection(collection_id=collection_id)
            assert result["success"] == True

            # Verify deletion
            result = await km.list_collections()
            assert result["total"] == 0
        finally:
            # Explicitly close client to release file lock
            if hasattr(km, "_qdrant_client") and km._qdrant_client:
                km._qdrant_client.close()
            if hasattr(km, "_qdrant_aclient") and km._qdrant_aclient:
                await km._qdrant_aclient.close()


async def test_knowledge_chat_configuration():
    """Test Chat Configuration Interface"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        import yaml

        config_path = Path(tmp_dir) / "test_config.yaml"
        config = get_test_config(tmp_dir)

        with open(config_path, "w") as f:
            yaml.dump(config, f)

        km = KnowledgeToolSet(config_path=str(config_path))
        await km.run_setup()

        try:
            # Create collection
            result = await km.create_collection(name="Test")
            collection_id = result["collection"]["id"]

            chat_id = "test_chat"

            # Get initial configuration
            result = await km.get_chat_knowledge(chat_id=chat_id)
            assert result["success"] == True
            assert result["config"]["auto_search"] == False

            # Enable collection
            result = await km.enable_collection(
                chat_id=chat_id, collection_id=collection_id
            )
            assert result["success"] == True
            assert collection_id in result["config"]["active_collection_ids"]

            # Set auto search
            result = await km.set_auto_search(chat_id=chat_id, enabled=True)
            assert result["success"] == True
            assert result["config"]["auto_search"] == True

            # Disable collection
            result = await km.disable_collection(
                chat_id=chat_id, collection_id=collection_id
            )
            assert result["success"] == True
            assert collection_id not in result["config"]["active_collection_ids"]

            # Cleanup: Delete created collection
            result = await km.delete_collection(collection_id=collection_id)
            assert result["success"] == True
        finally:
            # Explicitly close client to release file lock
            if hasattr(km, "_qdrant_client") and km._qdrant_client:
                km._qdrant_client.close()
            if hasattr(km, "_qdrant_aclient") and km._qdrant_aclient:
                await km._qdrant_aclient.close()


async def test_knowledge_concurrent_operations():
    """Test interface behavior for concurrent operations"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        import yaml

        config_path = Path(tmp_dir) / "test_config.yaml"
        config = get_test_config(tmp_dir)

        with open(config_path, "w") as f:
            yaml.dump(config, f)

        km = KnowledgeToolSet(config_path=str(config_path))
        await km.run_setup()

        try:
            # Concurrently create collections
            tasks = [km.create_collection(name=f"Collection {i}") for i in range(3)]

            results = await asyncio.gather(*tasks)

            # Verify all operations succeeded
            collection_ids = []
            for result in results:
                assert result["success"] == True
                assert "collection" in result
                collection_ids.append(result["collection"]["id"])

            # Verify list
            result = await km.list_collections()
            assert result["total"] == 3

            # Cleanup: Delete all created collections
            for collection_id in collection_ids:
                result = await km.delete_collection(collection_id=collection_id)
                assert result["success"] == True
        finally:
            # Explicitly close client to release file lock
            if hasattr(km, "_qdrant_client") and km._qdrant_client:
                km._qdrant_client.close()
            if hasattr(km, "_qdrant_aclient") and km._qdrant_aclient:
                await km._qdrant_aclient.close()


async def test_knowledge_api_response_format():
    """Test consistency of API response format"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        import yaml

        config_path = Path(tmp_dir) / "test_config.yaml"
        config = get_test_config(tmp_dir)

        with open(config_path, "w") as f:
            yaml.dump(config, f)

        km = KnowledgeToolSet(config_path=str(config_path))
        await km.run_setup()

        try:
            # All APIs should return a dict containing "success" field
            result = await km.list_collections()
            assert isinstance(result, dict)
            assert "success" in result

            result = await km.create_collection(name="Test")
            assert isinstance(result, dict)
            assert "success" in result
            collection_id = result["collection"]["id"]

            result = await km.get_chat_knowledge(chat_id="test")
            assert isinstance(result, dict)
            assert "success" in result

            # Cleanup: Delete created collection
            result = await km.delete_collection(collection_id=collection_id)
            assert result["success"] == True
        finally:
            # Explicitly close client to release file lock
            if hasattr(km, "_qdrant_client") and km._qdrant_client:
                km._qdrant_client.close()
            if hasattr(km, "_qdrant_aclient") and km._qdrant_aclient:
                await km._qdrant_aclient.close()


# If running this file directly
if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
