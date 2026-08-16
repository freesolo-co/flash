"""Standalone host-side helper programs shipped in the instance-bootstrap capsule.

Each module here runs as its own `python3 <file>` process on the rented box, NOT as an import of
the flash package. They must stay stdlib-only and self-contained.
"""
