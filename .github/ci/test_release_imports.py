"""Model-free release PE import controls; no images or DLLs are executed."""

from io import BytesIO
from pathlib import Path
import struct
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from check_release_imports import verify  # noqa: E402


def image(name=None, delay=False, address=False):
    data = bytearray(1024)
    data[:2] = b"MZ"
    struct.pack_into("<I", data, 60, 64)
    data[64:68] = b"PE\0\0"
    struct.pack_into("<HH", data, 68, 0xAA64, 1)
    struct.pack_into("<H", data, 84, 240)
    struct.pack_into("<H", data, 88, 0x20B)
    struct.pack_into("<Q", data, 112, 0x400000)
    struct.pack_into("<I", data, 148, 512)
    struct.pack_into("<I", data, 196, 16)
    struct.pack_into("<IIII", data, 336, 512, 0x1000, 512, 512)
    if name is not None:
        directory = 13 if delay else 1
        struct.pack_into("<II", data, 200 + directory * 8, 0x1000, 64 if delay else 40)
        if delay:
            struct.pack_into("<II", data, 512, 0 if address else 1,
                             0x401080 if address else 0x1080)
        else:
            struct.pack_into("<I", data, 524, 0x1080)
        encoded = name.encode("ascii") + b"\0"
        data[640:640 + len(encoded)] = encoded
    return data


class ReleaseImportTests(unittest.TestCase):
    def test_system_imports_and_no_imports_pass(self):
        self.assertEqual(verify(BytesIO(image()), 0xAA64), [])
        for name in ("KERNEL32.dll", "ucrtbase.dll", "msvcrt.dll", "api-ms-win-crt-runtime-l1-1-0.dll"):
            for delay in (False, True):
                with self.subTest(name=name, delay=delay):
                    self.assertEqual(verify(BytesIO(image(name, delay)), 0xAA64), [name.lower()])

    def test_external_runtime_imports_fail_in_both_tables(self):
        for name in ("VCRUNTIME140.dll", "msvcp140.dll", "concrt140.dll", "vcomp140.dll", "msvcr120.dll", "ucrtbased.dll"):
            for delay in (False, True):
                with self.subTest(name=name, delay=delay):
                    with self.assertRaisesRegex(ValueError, "external VC runtime"):
                        verify(BytesIO(image(name, delay)), 0xAA64)

    def test_delay_addresses_and_invalid_attributes_fail_closed(self):
        self.assertEqual(verify(BytesIO(image("KERNEL32.dll", True, True)), 0xAA64), ["kernel32.dll"])
        for attributes, name in ((2, 0x1080), (0, 1)):
            data = image("KERNEL32.dll", True)
            struct.pack_into("<II", data, 512, attributes, name)
            with self.assertRaises(ValueError):
                verify(BytesIO(data), 0xAA64)

    def test_wrong_machine_truncated_or_unmapped_tables_fail(self):
        for data in (image()[:100], image("kernel32.dll")):
            with self.assertRaises(ValueError):
                verify(BytesIO(data), 0x8664)
        data = image("kernel32.dll")
        struct.pack_into("<II", data, 208, 0x1300, 40)
        with self.assertRaises(ValueError):
            verify(BytesIO(data), 0xAA64)
        struct.pack_into("<II", data, 208, 0x1000, 20)
        with self.assertRaises(ValueError):
            verify(BytesIO(data), 0xAA64)

    def test_names_and_overlapping_sections_fail(self):
        for name in ("", "bad/name.dll", "a" * 256):
            with self.assertRaises(ValueError):
                verify(BytesIO(image(name)), 0xAA64)
        data = image()
        struct.pack_into("<H", data, 70, 2)
        data[368:408] = data[328:368]
        with self.assertRaises(ValueError):
            verify(BytesIO(data), 0xAA64)


if __name__ == "__main__":
    unittest.main()
