import os
import struct
import sys
import zlib
from pathlib import Path


LOCAL_FILE_HEADER = b"PK\x03\x04"
CENTRAL_DIRECTORY_HEADER = b"PK\x01\x02"
END_OF_CENTRAL_DIRECTORY = b"PK\x05\x06"


def safe_output_path(root: Path, member_name: str) -> Path:
    target = root / member_name
    resolved = target.resolve()
    root_resolved = root.resolve()
    if not str(resolved).startswith(str(root_resolved) + os.sep) and resolved != root_resolved:
        raise ValueError(f"Unsafe path in archive: {member_name}")
    return resolved


def extract_sequential(zip_path: Path, output_root: Path) -> int:
    extracted = 0
    with zip_path.open("rb") as src:
        while True:
            sig = src.read(4)
            if not sig:
                break
            if sig in {CENTRAL_DIRECTORY_HEADER, END_OF_CENTRAL_DIRECTORY}:
                break
            if sig != LOCAL_FILE_HEADER:
                raise ValueError(f"Unexpected ZIP signature at offset {src.tell() - 4}: {sig!r}")

            header = src.read(26)
            if len(header) != 26:
                raise ValueError("Truncated ZIP local header")

            (
                _version,
                flags,
                method,
                _mtime,
                _mdate,
                _crc32,
                compressed_size,
                _uncompressed_size,
                file_name_length,
                extra_length,
            ) = struct.unpack("<HHHHHIIIHH", header)

            if flags != 0:
                raise ValueError(f"Unsupported ZIP flags {flags} for sequential extraction")

            name_bytes = src.read(file_name_length)
            extra = src.read(extra_length)
            if len(name_bytes) != file_name_length or len(extra) != extra_length:
                raise ValueError("Truncated ZIP filename or extra field")

            try:
                member_name = name_bytes.decode("utf-8")
            except UnicodeDecodeError:
                member_name = name_bytes.decode("cp437")

            target_path = safe_output_path(output_root, member_name)
            if member_name.endswith("/"):
                target_path.mkdir(parents=True, exist_ok=True)
                extracted += 1
                continue

            compressed = src.read(compressed_size)
            if len(compressed) != compressed_size:
                raise ValueError(f"Truncated data for {member_name}")

            if method == 0:
                data = compressed
            elif method == 8:
                data = zlib.decompress(compressed, -zlib.MAX_WBITS)
            else:
                raise ValueError(f"Unsupported compression method {method} for {member_name}")

            target_path.parent.mkdir(parents=True, exist_ok=True)
            target_path.write_bytes(data)
            extracted += 1

    return extracted


def main() -> int:
    if len(sys.argv) != 3:
        print("Usage: python .codex_tmp_extract_zip.py <zip_path> <output_root>")
        return 2

    zip_path = Path(sys.argv[1])
    output_root = Path(sys.argv[2])
    count = extract_sequential(zip_path, output_root)
    print(f"Extracted {count} entries to {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
