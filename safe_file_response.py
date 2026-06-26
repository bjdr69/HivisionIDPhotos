"""
Patch Starlette FileResponse to avoid h11 LocalProtocolError:
"Too little data for declared Content-Length"

Root cause: Gradio's upload handler returns file paths before the
background file copy (shutil.move) completes. When the browser requests
the file, FileResponse calls os.stat() (file may still be 0 bytes or
being written), then reads the file content — getting fewer bytes than
the Content-Length header declared.

Fix: Read file content with retry, and always send the correct
Content-Length based on actual bytes read.
"""
import os
import sys
import asyncio
import anyio
import starlette.responses

# Save original method
_original_handle_simple = starlette.responses.FileResponse._handle_simple

# Maximum retries and delay for waiting on files being written
MAX_RETRIES = 10
RETRY_DELAY = 0.1  # seconds


async def _patched_handle_simple(self, send, send_header_only):
    """
    Read file content and send with correct Content-Length.

    Retries reading if the file is empty (may still be written by
    Gradio's background file copy task). Always sends Content-Length
    matching the actual bytes read, preventing h11 protocol errors.
    """
    if send_header_only:
        return await _original_handle_simple(self, send, send_header_only)

    # Read the file content with retries for files being written
    content = b""
    for attempt in range(MAX_RETRIES):
        try:
            async with await anyio.open_file(self.path, mode="rb") as file:
                content = await file.read()
        except FileNotFoundError:
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(RETRY_DELAY)
                continue
            raise RuntimeError(f"File at path {self.path} does not exist.")

        if len(content) > 0:
            break

        # File exists but is empty — likely being written by background task
        if attempt < MAX_RETRIES - 1:
            await asyncio.sleep(RETRY_DELAY)
        else:
            print(
                f"[SafeFileResponse] WARNING: File is empty after {MAX_RETRIES} retries: {self.path}",
                file=sys.stderr,
            )

    # Fix Content-Length to match actual content size
    actual_size = len(content)
    for i, (k, v) in enumerate(self.raw_headers):
        if k == b"content-length":
            declared_size = int(v)
            if actual_size != declared_size:
                print(
                    f"[SafeFileResponse] Content-Length mismatch: "
                    f"declared={declared_size}, actual={actual_size}, "
                    f"path={self.path}",
                    file=sys.stderr,
                )
                self.raw_headers[i] = (
                    b"content-length",
                    str(actual_size).encode("latin-1"),
                )
            break

    # Send headers
    await send(
        {
            "type": "http.response.start",
            "status": self.status_code,
            "headers": self.raw_headers,
        }
    )

    # Send body in chunks
    chunk_size = self.chunk_size
    offset = 0
    while offset < actual_size:
        chunk = content[offset : offset + chunk_size]
        offset += chunk_size
        more_body = offset < actual_size
        await send(
            {
                "type": "http.response.body",
                "body": chunk,
                "more_body": more_body,
            }
        )


def apply_patch():
    """Apply the SafeFileResponse patch."""
    starlette.responses.FileResponse._handle_simple = _patched_handle_simple
    print(
        "[SafeFileResponse] Patched FileResponse._handle_simple to prevent Content-Length mismatch",
        file=sys.stderr,
    )
