import tempfile
import unittest
from pathlib import Path

from src.api.upload_security import (
    resolve_local_import_paths,
    sanitize_upload_filename,
    upload_destination,
)


class UploadSecurityTests(unittest.TestCase):
    def test_upload_filename_is_reduced_to_a_safe_basename(self):
        self.assertEqual("secret.pdf", sanitize_upload_filename("../../secret.pdf"))
        self.assertEqual(
            "report.xlsx",
            sanitize_upload_filename(r"C:\fakepath\report.xlsx"),
        )
        with self.assertRaises(ValueError):
            sanitize_upload_filename("evil\x00.pdf")

    def test_destination_always_stays_below_upload_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = upload_destination(root, "../../outside.pdf")
            self.assertEqual(root.resolve(), destination.parent)

    def test_local_import_is_disabled_by_default(self):
        with self.assertRaises(PermissionError):
            resolve_local_import_paths(["/tmp/file.xlsx"], ["/tmp"], enabled=False)

    def test_local_import_roots_must_be_absolute(self):
        with self.assertRaises(PermissionError):
            resolve_local_import_paths(
                ["/tmp/file.xlsx"], ["relative/root"], enabled=True
            )

    def test_local_import_rejects_outside_and_symlink_escape(self):
        with tempfile.TemporaryDirectory() as allowed_directory:
            with tempfile.TemporaryDirectory() as outside_directory:
                allowed = Path(allowed_directory)
                inside = allowed / "inside.xlsx"
                inside.write_bytes(b"safe")
                outside = Path(outside_directory) / "outside.xlsx"
                outside.write_bytes(b"unsafe")

                self.assertEqual(
                    [str(inside.resolve())],
                    resolve_local_import_paths(
                        [str(inside)], [str(allowed)], enabled=True
                    ),
                )
                with self.assertRaises(PermissionError):
                    resolve_local_import_paths(
                        [str(outside)], [str(allowed)], enabled=True
                    )

                link = allowed / "linked.xlsx"
                try:
                    link.symlink_to(outside)
                except OSError:
                    self.skipTest("Symbolic links are not supported")
                with self.assertRaises(PermissionError):
                    resolve_local_import_paths(
                        [str(link)], [str(allowed)], enabled=True
                    )


if __name__ == "__main__":
    unittest.main()
