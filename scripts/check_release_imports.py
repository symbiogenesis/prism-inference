"""Read PE data only and reject separately installed VC runtimes in bundles.

The bounded header, section, normal-import and delay-import checks follow the
Runner portable-package verifier. Windows system CRT/API-set imports remain
allowed. No image or DLL is loaded, and no external PE tools are required.
"""

import argparse
from pathlib import Path
import re
import struct


def verify(stream, machine):
    """Reject malformed PE32+ metadata and external VC runtime dependencies."""
    stream.seek(0, 2)
    length = stream.tell()

    def invalid():
        raise ValueError("Invalid or unsupported release PE import metadata")

    def read(offset, count):
        if offset < 0 or count < 0 or offset + count > length:
            invalid()
        stream.seek(offset)
        data = stream.read(count)
        if len(data) != count:
            invalid()
        return data

    dos = read(0, 64)
    pe_offset = struct.unpack_from("<I", dos, 60)[0]
    if dos[:2] != b"MZ" or pe_offset < 64:
        invalid()
    pe = read(pe_offset, 24)
    found_machine, count = struct.unpack_from("<HH", pe, 4)
    optional_size = struct.unpack_from("<H", pe, 20)[0]
    if (pe[:4] != b"PE\0\0" or found_machine != machine
            or count > 96 or not 112 <= optional_size <= 4096):
        invalid()
    optional = read(pe_offset + 24, optional_size)
    if struct.unpack_from("<H", optional)[0] != 0x20B:
        invalid()
    base = struct.unpack_from("<Q", optional, 24)[0]
    headers = struct.unpack_from("<I", optional, 60)[0]
    directory_count = struct.unpack_from("<I", optional, 108)[0]
    if directory_count > 16 or 112 + directory_count * 8 > optional_size:
        invalid()
    directories = [struct.unpack_from("<II", optional, 112 + index * 8)
                   for index in range(directory_count)]
    directories.extend([(0, 0)] * (16 - directory_count))
    section_table = pe_offset + 24 + optional_size
    if not section_table + count * 40 <= headers <= length:
        invalid()
    sections = []

    def overlap(start, size, other_start, other_size):
        return (size > 0 and other_size > 0
                and start < other_start + other_size and other_start < start + size)

    for index in range(count):
        section = read(section_table + index * 40, 40)
        virtual_size, rva, raw_size, raw_offset = struct.unpack_from("<IIII", section, 8)
        mapped = max(virtual_size, raw_size)
        if (raw_offset + raw_size > length or rva + mapped > 0xFFFFFFFF
                or (mapped and rva < headers)):
            invalid()
        for other_rva, other_mapped, other_offset, other_raw in sections:
            if (overlap(rva, mapped, other_rva, other_mapped)
                    or overlap(raw_offset, raw_size, other_offset, other_raw)):
                invalid()
        sections.append((rva, mapped, raw_offset, raw_size))

    def location(rva, size):
        if rva < headers:
            if size <= headers - rva:
                return rva, headers - rva
            invalid()
        for start, mapped, offset, raw_size in sections:
            if start <= rva < start + mapped:
                delta = rva - start
                available = raw_size - delta
                if size <= available:
                    return offset + delta, available
                invalid()
        invalid()

    imports = []

    def dll_name(rva):
        if not rva:
            invalid()
        offset, available = location(rva, 1)
        data = read(offset, min(available, 256))
        end = data.find(b"\0")
        if end <= 0 or re.fullmatch(rb"[A-Za-z0-9._-]+", data[:end]) is None:
            invalid()
        name = data[:end].decode("ascii").lower()
        if (name.startswith(("vcruntime", "msvcp", "concrt", "vcomp", "ucrtbased"))
                or (name.startswith("msvcr") and name != "msvcrt.dll")):
            raise ValueError(f"Release executable imports external VC runtime {name}")
        imports.append(name)

    def table(directory, delay):
        rva, size = directory
        if rva == 0 and size == 0:
            return
        width = 32 if delay else 20
        if rva == 0 or not width <= size <= 1024 * 1024:
            invalid()
        offset, _ = location(rva, size)
        for index in range(min(size // width, 4096)):
            descriptor = read(offset + index * width, width)
            if not any(descriptor):
                return
            if delay:
                attributes, name = struct.unpack_from("<II", descriptor)
                if attributes & ~1:
                    invalid()
                if attributes == 0:
                    name -= base
                if not 0 <= name <= 0xFFFFFFFF:
                    invalid()
            else:
                name = struct.unpack_from("<I", descriptor, 12)[0]
            dll_name(name)
        invalid()

    table(directories[1], False)
    table(directories[13], True)
    return imports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--machine", required=True, type=lambda value: int(value, 0),
                        choices=(0xAA64, 0x8664))
    parser.add_argument("images", nargs="+", type=Path)
    options = parser.parse_args()
    for path in options.images:
        try:
            with path.open("rb") as stream:
                verify(stream, options.machine)
        except (OSError, ValueError, struct.error) as error:
            parser.exit(1, f"{path.name}: {error}\n")
    print(f"Release PE import policy passed for {len(options.images)} executable(s).")


if __name__ == "__main__":
    main()
