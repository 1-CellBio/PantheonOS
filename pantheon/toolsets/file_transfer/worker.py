import aiofiles
import base64
import re
import time
import uuid
from pathlib import Path

from pantheon.toolset import tool
from pantheon.toolsets.file.file_manager import FileManagerToolSetBase
from pantheon.utils.log import logger
from pantheon.utils.workspace import build_upload_attachment_metadata


class FileTransferToolSet(FileManagerToolSetBase):
    """File transfer toolset with async I/O and smart flush strategy.

    This class provides basic file transfer functionality with performance optimizations:
    - Async I/O using aiofiles (non-blocking)
    - Smart flush strategy (balances performance and responsiveness)
    - Progress tracking for better UX

    Features:
    - open_file_for_write: Open file for async writing
    - write_chunk: Write data chunks with smart flushing
    - close_file: Close file with final flush
    - open_file_for_read: Open file for async reading
    - read_chunk: Read data chunks
    - read_file: Read entire file (streaming or non-streaming)
    """

    def __init__(
        self,
        name: str,
        path: str | Path,
        **kwargs,
    ):
        super().__init__(name, path, **kwargs)
        self._handles = {}  # {handle_id: aiofiles.File}
        self._file_info = {}  # {handle_id: metadata dict}

        # Smart flush strategy configuration
        self.FLUSH_INTERVAL_CHUNKS = 10  # Flush every N chunks
        self.FLUSH_INTERVAL_BYTES = 5 * 1024 * 1024  # Flush every 5MB
        self.FLUSH_INTERVAL_SECONDS = 2.0  # Flush every 2 seconds
        self._staged_upload_name_re = re.compile(r"^\d+_[A-Za-z0-9]+_(.+)$")

    def _normalize_external_upload_name(self, file_path: str | Path) -> str:
        """Recover the original filename from a staged upload path when possible."""
        filename = Path(file_path).name
        match = self._staged_upload_name_re.match(filename)
        if match:
            return match.group(1)
        return filename

    def _resolve_write_path(self, file_path: str, *, workspace_view: str | None = None) -> Path:
        """Resolve upload destinations while remapping external absolute sources."""
        upload_target = self._resolve_upload_namespace_path(file_path)
        if upload_target is not None:
            return upload_target

        if not Path(file_path).is_absolute():
            return self._resolve_path_for_view(file_path, workspace_view=workspace_view)

        candidate = Path(file_path).resolve()
        root = self._get_root().resolve()
        if candidate == root or candidate.is_relative_to(root):
            return candidate

        # Absolute path outside workspace remapped to .uploaded_files
        upload_dir = root / ".uploaded_files"
        upload_dir.mkdir(parents=True, exist_ok=True)

        filename = self._normalize_external_upload_name(candidate)
        stem = Path(filename).stem or "upload"
        suffix = Path(filename).suffix
        target = upload_dir / filename
        counter = 1
        while target.exists():
            target = upload_dir / f"{stem}_{counter}{suffix}"
            counter += 1

        return target.resolve()

    def _build_attachment_result(self, path: Path) -> dict | None:
        layout = self._get_managed_workspace_layout()
        if layout is None:
            return None
        return build_upload_attachment_metadata(layout, path)

    @tool
    async def open_file_for_write(self, file_path: str, workspace_view: str | None = None):
        """Open a file for writing (async).

        Args:
            file_path: Relative path to the file (cannot contain '..')
            workspace_view: ``"global"`` resolves against the project workspace,
                ``"isolated"`` against the session workspace.

        Returns:
            dict: {"success": True, "handle_id": str} on success
                  {"success": False, "error": str} on failure
        """
        if ".." in file_path:
            return {"error": "File path cannot contain '..'"}

        self._ensure_attachment_bridge()
        path = self._resolve_write_path(file_path, workspace_view=workspace_view)

        # Ensure parent directory exists
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.warning(f"Failed to create parent directory: {e}")

        handle_id = str(uuid.uuid4())

        try:
            # Open file asynchronously
            handle = await aiofiles.open(path, mode="wb")
            attachment = self._build_attachment_result(path)
            self._handles[handle_id] = handle
            self._file_info[handle_id] = {
                "path": path,
                "attachment": attachment,
                "size": 0,
                "chunks_written": 0,
                "bytes_since_flush": 0,
                "last_flush_time": time.time(),
                "created_at": time.time(),
            }
            logger.debug(f"Opened file for write: {path} (handle_id={handle_id})")
            result = {"success": True, "handle_id": handle_id}
            if attachment is not None:
                result["attachment"] = attachment
            return result
        except Exception as e:
            logger.error(f"Failed to open file for write: {path}, error: {e}")
            return {"success": False, "error": str(e)}

    @tool
    async def write_chunk(self, handle_id: str, data):
        """Write a chunk to a file with smart flush strategy.

        Args:
            handle_id: File handle ID from open_file_for_write.
            data: Binary data as bytes (from cloudpickle) or
                  base64-encoded string (from JSON/frontend clients).

        Returns:
            dict: {"success": True, "bytes_written": int} on success
                  {"success": False, "error": str} on failure
        """
        if handle_id not in self._handles:
            return {"success": False, "error": "Handle not found"}

        handle = self._handles[handle_id]
        info = self._file_info[handle_id]

        # Decode data if base64-encoded string
        if isinstance(data, str):
            data = base64.b64decode(data)
        elif not isinstance(data, (bytes, bytearray)):
            return {
                "success": False,
                "error": f"Unsupported data type: {type(data).__name__}",
            }

        try:
            # Write data asynchronously (non-blocking)
            await handle.write(data)

            # Update statistics
            bytes_written = len(data)
            info["size"] += bytes_written
            info["chunks_written"] += 1
            info["bytes_since_flush"] += bytes_written

            # Smart flush strategy: flush if any condition is met
            current_time = time.time()
            last_flush_time = info["last_flush_time"]

            should_flush = (
                # Condition 1: Every N chunks
                info["chunks_written"] % self.FLUSH_INTERVAL_CHUNKS == 0
                or
                # Condition 2: Every N bytes
                info["bytes_since_flush"] >= self.FLUSH_INTERVAL_BYTES
                or
                # Condition 3: Every N seconds
                (current_time - last_flush_time) >= self.FLUSH_INTERVAL_SECONDS
            )

            if should_flush:
                await handle.flush()
                info["bytes_since_flush"] = 0
                info["last_flush_time"] = current_time
                logger.debug(
                    f"Flushed file (handle_id={handle_id}, "
                    f"chunks={info['chunks_written']}, "
                    f"total_size={info['size']} bytes)"
                )

            return {"success": True, "bytes_written": bytes_written}

        except Exception as e:
            logger.error(f"Failed to write chunk (handle_id={handle_id}): {e}")
            return {"success": False, "error": str(e)}

    @tool
    async def close_file(self, handle_id: str):
        """Close a file with final flush (async).

        Args:
            handle_id: File handle ID from open_file_for_write or open_file_for_read.

        Returns:
            dict: {"success": True, "total_size": int, "chunks_written": int} on success (write mode)
                  {"success": True} on success (read mode)
                  {"success": False, "error": str} on failure
        """
        if handle_id not in self._handles:
            return {"success": False, "error": "Handle not found"}

        handle = self._handles[handle_id]
        info = self._file_info[handle_id]

        try:
            # Final flush and close (async, non-blocking)
            await handle.flush()
            await handle.close()

            # Calculate statistics
            total_time = time.time() - info["created_at"]
            total_size = info["size"]

            # Check if this is a write handle (has chunks_written) or read handle
            is_write_handle = "chunks_written" in info

            if is_write_handle:
                chunks_written = info["chunks_written"]
                logger.info(
                    f"Closed file (write): {info['path']} "
                    f"(size={total_size} bytes, chunks={chunks_written}, time={total_time:.2f}s)"
                )
            else:
                logger.info(
                    f"Closed file (read): {info['path']} "
                    f"(size={total_size} bytes, time={total_time:.2f}s)"
                )

            # Cleanup
            del self._handles[handle_id]
            del self._file_info[handle_id]

            # Return appropriate response based on handle type
            if is_write_handle:
                return {
                    "success": True,
                    "total_size": total_size,
                    "chunks_written": chunks_written,
                    "duration_seconds": round(total_time, 2),
                    **(
                        {"attachment": attachment}
                        if (attachment := info.get("attachment")) is not None
                        else {}
                    ),
                }
            else:
                return {"success": True}

        except Exception as e:
            logger.error(f"Failed to close file (handle_id={handle_id}): {e}")

            # Cleanup on error
            try:
                await handle.close()
            except:
                pass

            if handle_id in self._handles:
                del self._handles[handle_id]
            if handle_id in self._file_info:
                del self._file_info[handle_id]

            return {"success": False, "error": str(e)}

    @tool
    async def open_file_for_read(self, file_path: str, workspace_view: str | None = None):
        """Open a file for chunked reading (async). Returns handle_id and total_size.

        Args:
            file_path: Relative path to the file (cannot contain '..')
            workspace_view: ``"global"`` resolves against the project workspace,
                ``"isolated"`` against the session workspace.

        Returns:
            dict: {"success": True, "handle_id": str, "total_size": int} on success
                  {"success": False, "error": str} on failure
        """
        if ".." in file_path:
            return {"success": False, "error": "File path cannot contain '..'"}

        self._ensure_attachment_bridge()
        path = self._resolve_user_visible_path(
            file_path, workspace_view=workspace_view,
        )
        if not path.exists():
            return {"success": False, "error": "File does not exist"}

        handle_id = str(uuid.uuid4())

        try:
            # Open file asynchronously for reading
            handle = await aiofiles.open(path, mode="rb")
            total_size = path.stat().st_size
            self._handles[handle_id] = handle
            self._file_info[handle_id] = {
                "path": path,
                "size": total_size,
                "bytes_read": 0,
                "created_at": time.time(),
            }
            logger.debug(
                f"Opened file for read: {path} (handle_id={handle_id}, size={total_size} bytes)"
            )
            return {"success": True, "handle_id": handle_id, "total_size": total_size}
        except Exception as e:
            logger.error(f"Failed to open file for read: {path}, error: {e}")
            return {"success": False, "error": str(e)}

    @tool
    async def read_chunk(self, handle_id: str, size: int = 512 * 1024):
        """Read a chunk from an open file handle (async).

        Args:
            handle_id: File handle ID from open_file_for_read.
            size: Number of bytes to read (before base64 encoding). Default: 512KB.

        Returns:
            dict: {"success": True, "data": str (base64), "bytes_read": int, "eof": bool}
                  {"success": False, "error": str} on failure
        """
        if handle_id not in self._handles:
            return {"success": False, "error": "Handle not found"}

        handle = self._handles[handle_id]
        info = self._file_info[handle_id]

        try:
            # Read data asynchronously
            data = await handle.read(size)

            if not data:
                return {"success": True, "data": "", "bytes_read": 0, "eof": True}

            bytes_read = len(data)
            info["bytes_read"] += bytes_read

            return {
                "success": True,
                "data": base64.b64encode(data).decode("utf-8"),
                "bytes_read": bytes_read,
                "eof": bytes_read < size,
            }
        except Exception as e:
            logger.error(f"Failed to read chunk (handle_id={handle_id}): {e}")
            return {"success": False, "error": str(e)}

    @tool
    async def read_file(
        self, file_path: str, receive_chunk=None, chunk_size: int = 1024,
        workspace_view: str | None = None,
    ):
        """Read a file (streaming or non-streaming mode).

        Args:
            file_path: Relative path to the file (cannot contain '..')
            receive_chunk: Optional callback for streaming mode (direct connections)
            chunk_size: Chunk size for streaming mode. Default: 1KB.
            workspace_view: ``"global"`` resolves against the project workspace,
                ``"isolated"`` against the session workspace.

        Returns:
            Non-streaming: {"success": True, "data": str (base64), "total_size": int, "encoding": "base64"}
            Streaming: {"success": True}
            Error: {"success": False, "error": str}
        """
        if ".." in file_path:
            return {"success": False, "error": "File path cannot contain '..'"}

        self._ensure_attachment_bridge()
        path = self._resolve_user_visible_path(
            file_path, workspace_view=workspace_view,
        )
        if not path.exists():
            return {"success": False, "error": "File does not exist"}

        if receive_chunk is None:
            # Non-streaming mode: return full file content (base64 encoded for JSON compatibility)
            try:
                async with aiofiles.open(path, mode="rb") as f:
                    file_data = await f.read()
                    return {
                        "success": True,
                        "data": base64.b64encode(file_data).decode("utf-8"),
                        "total_size": len(file_data),
                        "encoding": "base64",
                    }
            except Exception as e:
                logger.error(f"Failed to read file: {path}, error: {e}")
                return {"success": False, "error": str(e)}
        else:
            # Streaming mode: use callback function for direct connections
            try:
                async with aiofiles.open(path, mode="rb") as f:
                    while True:
                        data = await f.read(chunk_size)
                        if not data:
                            break
                        await receive_chunk(data)
                return {"success": True}
            except Exception as e:
                logger.error(f"Failed to stream file: {path}, error: {e}")
                return {"success": False, "error": str(e)}
