from pathlib import Path

import anyio
from fastapi import UploadFile

_CHUNK_BYTES = 1024 * 1024


class FileTooLarge(Exception):
    pass


class EmptyFile(Exception):
    pass


async def save_upload(upload: UploadFile, dest: Path, max_bytes: int) -> int:
    """Stream an upload to `dest`, enforcing a size limit. Returns bytes written.

    On any error the partial file is removed.
    """
    await anyio.Path(dest.parent).mkdir(parents=True, exist_ok=True)
    written = 0
    try:
        async with await anyio.open_file(dest, "wb") as f:
            while chunk := await upload.read(_CHUNK_BYTES):
                written += len(chunk)
                if written > max_bytes:
                    raise FileTooLarge
                await f.write(chunk)
        if written == 0:
            raise EmptyFile
    except BaseException:
        await delete_file(dest)
        raise
    return written


async def delete_file(path: Path) -> None:
    await anyio.Path(path).unlink(missing_ok=True)
