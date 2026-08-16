"""Regressions for the seven confirmed env-push secret-scan bypasses."""

from __future__ import annotations

import hashlib
import io
import shutil
import struct
import subprocess
import zipfile
import zlib

import pytest

_KEY = b"fslo_" + b"a1B2c3D4" * 6


def _chunk(kind: bytes, payload: bytes) -> bytes:
    return (
        len(payload).to_bytes(4, "big")
        + kind
        + payload
        + zlib.crc32(kind + payload).to_bytes(4, "big")
    )


def _png(
    rows: list[tuple[int, bytes]],
    *,
    width: int,
    bit_depth: int = 8,
    color_type: int = 0,
    interlace: int = 0,
    extra: tuple[tuple[bytes, bytes], ...] = (),
) -> bytes:
    ihdr = struct.pack(">IIBBBBB", width, len(rows), bit_depth, color_type, 0, 0, interlace)
    body = b"".join(bytes((filter_kind,)) + row for filter_kind, row in rows)
    palette = (_chunk(b"PLTE", b"\x00\x00\x00\xff\xff\xff"),) if color_type == 3 else ()
    return (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", ihdr)
        + b"".join(palette)
        + b"".join(_chunk(kind, payload) for kind, payload in extra)
        + _chunk(b"IDAT", zlib.compress(body))
        + _chunk(b"IEND", b"")
    )


def _paeth(left: int, above: int, upper_left: int) -> int:
    prediction = left + above - upper_left
    distances = (
        abs(prediction - left),
        abs(prediction - above),
        abs(prediction - upper_left),
    )
    return (left, above, upper_left)[distances.index(min(distances))]


def _filtered(raw: bytes, previous: bytes, kind: int, stride: int = 1) -> bytes:
    encoded = bytearray()
    for at, value in enumerate(raw):
        left = raw[at - stride] if at >= stride else 0
        above = previous[at] if previous else 0
        upper_left = previous[at - stride] if previous and at >= stride else 0
        predictor = {
            0: 0,
            1: left,
            2: above,
            3: (left + above) // 2,
            4: _paeth(left, above, upper_left),
        }[kind]
        encoded.append((value - predictor) & 0xFF)
    return bytes(encoded)


def _arrow_payload(kind: str, *, legacy: bool = False) -> bytes:
    import pyarrow as pa

    batch = pa.record_batch([pa.array(["safe", "values"])], names=["value"])
    sink = io.BytesIO()
    options = pa.ipc.IpcWriteOptions(compression="zstd", use_legacy_format=legacy)
    maker = pa.ipc.new_file if kind == "file" else pa.ipc.new_stream
    with maker(sink, batch.schema, options=options) as writer:
        writer.write_batch(batch)
    return sink.getvalue()


def _git_pack(
    *,
    version: int = 2,
    values: tuple[bytes, ...] = (b"x",),
    count: int | None = None,
    valid_digest: bool = True,
    damage_last: bool = False,
) -> bytes:
    records = [bytes((0x30 | len(value),)) + zlib.compress(value) for value in values]
    if damage_last:
        records[-1] = records[-1][:-2] + b"xx"
    declared = len(records) if count is None else count
    body = b"PACK" + version.to_bytes(4, "big") + declared.to_bytes(4, "big") + b"".join(records)
    digest = hashlib.sha1(body).digest()
    return body + (digest if valid_digest else bytes(len(digest)))


def _git_bundle(pack: bytes, *, version: int = 2) -> bytes:
    object_id = b"1" * 40
    return b"# v%d git bundle\n" % version + object_id + b" refs/heads/main\n\n" + pack


def _rpm_lead() -> bytes:
    lead = bytearray(96)
    lead[:6] = b"\xed\xab\xee\xdb\x03\x00"
    lead[6:8] = (0).to_bytes(2, "big")
    lead[8:10] = (1).to_bytes(2, "big")
    lead[10:22] = b"sample-1.rpm"
    lead[76:78] = (1).to_bytes(2, "big")
    lead[78:80] = (5).to_bytes(2, "big")
    return bytes(lead)


@pytest.mark.parametrize("offset", [0, 512, 1024, 4096])
def test_hdf5_signatures_at_superblock_positions_fail_closed(tmp_path, offset):
    from flash.env_secrets import _Unscannable, credential_in_file

    artifact = tmp_path / "dataset.bin"
    artifact.write_bytes(bytes(offset) + b"\x89HDF\r\n\x1a\n" + zlib.compress(_KEY))
    with pytest.raises(_Unscannable, match="HDF5 archive"):
        credential_in_file(artifact)


def test_hdf5_false_positive_controls_remain_clean(tmp_path):
    from flash.env_secrets import credential_in_file

    controls = {
        "filename.h5": b"ordinary values\n",
        "wrong-offset.bin": bytes(511) + b"\x89HDF\r\n\x1a\n",
        "prose.txt": b"the escaped signature is 89 48 44 46 0d 0a 1a 0a\n",
    }
    for name, content in controls.items():
        path = tmp_path / name
        path.write_bytes(content)
        assert credential_in_file(path) is None


@pytest.mark.parametrize(
    "payload",
    [
        _arrow_payload("file"),
        _arrow_payload("stream"),
        _arrow_payload("stream", legacy=True),
    ],
)
def test_arrow_file_and_stream_framing_fail_closed(tmp_path, payload):
    from flash.env_secrets import _Unscannable, credential_in_file

    artifact = tmp_path / "records.bin"
    artifact.write_bytes(payload)
    with pytest.raises(_Unscannable, match="Arrow IPC archive"):
        credential_in_file(artifact)


def test_arrow_false_positive_controls_remain_clean(tmp_path):
    from flash.env_secrets import credential_in_file

    controls = {
        "records.arrow": b"ordinary records\n",
        "prose.txt": b"ARROW1 is the printable file marker\n",
        "bare-continuation.bin": b"\xff\xff\xff\xffordinary bytes",
        "bad-message.bin": b"\xff\xff\xff\xff" + (40).to_bytes(4, "little") + bytes(40),
        "bad-legacy-message.bin": (44).to_bytes(4, "little") + bytes(44),
        "bad-footer.bin": b"ARROW1\x00\x00not a footer\x0c\x00\x00\x00ARROW1",
    }
    for name, content in controls.items():
        path = tmp_path / name
        path.write_bytes(content)
        assert credential_in_file(path) is None


@pytest.mark.parametrize(
    "expression",
    [
        b"(private-key (rsa (n #A1B2C3D4#) (e #010001#) (d #D4C3B2A1#)))",
        (
            b'(protected-private-key (ecc (curve "NIST P-256") (q #A1B2#) '
            b"(protected openpgp-s2k3-sha1-aes-cbc (#A1B2#))))"
        ),
        (b"(shadowed-private-key (rsa (n #A1B2#) (e #010001#)) (shadowed t1-v1 (#A1B2#)))"),
    ],
)
def test_native_gnupg_private_key_sexpressions_are_detected(tmp_path, expression):
    from flash.env_secrets import credential_in_file

    key = tmp_path / "native.key"
    key.write_bytes(
        b"\nKeygrip: 0123456789ABCDEF0123456789ABCDEF01234567\n"
        b"Created: 20260815T120000\nKey: " + expression + b"\n"
    )
    assert credential_in_file(key) == "a private key"


def test_native_gnupg_false_positive_controls_remain_clean(tmp_path):
    from flash.env_secrets import credential_in_file

    controls = {
        "public.key": b"Key: (public-key (rsa (n #A1B2#) (e #010001#)))\n",
        "docs.txt": b"Lisp documentation mentions (private-key ...) but has no Key field.\n",
        "truncated.key": b"Key: (private-key (rsa (n #A1B2#)\n",
        "empty.key": b"Key: (private-key)\n",
        "public-shape.key": b"Key: (private-key (rsa (n #A1B2#) (e #010001#)))\n",
        "odd-hex.key": b"Key: (private-key (rsa (d #ABC#)))\n",
        "bad-hex.key": b"Key: (private-key (rsa (d #ABXZ#)))\n",
        "metadata-prose.txt": b"\nComment: this describes native keys\nordinary prose follows\n",
        "malformed.key": b"Created:\nKey: (private-key)) extra\n",
    }
    for name, content in controls.items():
        path = tmp_path / name
        path.write_bytes(content)
        assert credential_in_file(path) is None


@pytest.mark.parametrize(
    "payload",
    [
        _git_pack(),
        _git_bundle(_git_pack(), version=2),
        _git_bundle(_git_pack(version=3), version=3),
    ],
)
def test_git_pack_and_bundle_fail_closed(tmp_path, payload):
    from flash.env_secrets import _Unscannable, credential_in_file

    artifact = tmp_path / "objects.bin"
    artifact.write_bytes(payload)
    with pytest.raises(_Unscannable, match=r"Git (?:pack|bundle) archive"):
        credential_in_file(artifact)


def test_native_gnupg_oversized_recognized_header_fails_closed(tmp_path, monkeypatch):
    from flash import env_sensitive
    from flash.env_secrets import credential_in_file

    monkeypatch.setattr(env_sensitive, "_MAX_GNUPG_HEADER_BYTES", 96)
    key = tmp_path / "long-native.key"
    key.write_bytes(
        b"Created: 20260815T120000\nComment: "
        + b"x" * 96
        + b"\nKey: (private-key (rsa (d #A1B2#)))\n"
    )
    assert credential_in_file(key) == "a private key"

    blank = tmp_path / "long-blank-native.key"
    blank.write_bytes(b"\n" * 100 + b"Key: (private-key (rsa (d #A1B2#)))\n")
    assert credential_in_file(blank) == "a private key"


def test_git_pack_and_bundle_false_positive_controls_remain_clean(tmp_path):
    from flash.env_secrets import credential_in_file

    controls = {
        "objects.pack": b"ordinary object names\n",
        "backup.bundle": b"ordinary bundle notes\n",
        "prose.txt": b"PACK version 2 and # v2 git bundle are documentation\n",
        "version-one.bin": _git_pack(version=1),
        "version-four.bin": _git_pack(version=4),
        "zero-count.bin": _git_pack(count=0),
        "bad-digest.bin": _git_pack(valid_digest=False),
        "declared-count.bin": _git_pack(values=(b"x", b"y"), count=1),
        "malformed-later.bin": _git_pack(values=(b"x", b"y"), damage_last=True),
        "bundle-prose.bin": b"prefix # v2 git bundle\n" + b"1" * 40 + b" refs/heads/main\n\nPACK",
    }
    for name, content in controls.items():
        path = tmp_path / name
        path.write_bytes(content)
        assert credential_in_file(path) is None


def test_real_git_pack_is_structurally_recognized(tmp_path):
    from flash.env_secrets import _Unscannable, credential_in_file

    if shutil.which("git") is None:
        pytest.skip("git is not installed")
    repo = tmp_path / "repo"
    subprocess.run(["git", "init", "-q", repo], check=True)
    object_ids = []
    for value in (b"first object\n", b"second object\n"):
        written = subprocess.run(
            ["git", "-C", repo, "hash-object", "-w", "--stdin"],
            input=value,
            stdout=subprocess.PIPE,
            check=True,
        )
        object_ids.append(written.stdout)
    packed = subprocess.run(
        ["git", "-C", repo, "pack-objects", "--stdout"],
        input=b"".join(object_ids),
        stdout=subprocess.PIPE,
        check=True,
    ).stdout
    path = tmp_path / "real.pack"
    path.write_bytes(packed)
    with pytest.raises(_Unscannable, match="Git pack archive"):
        credential_in_file(path)


def test_git_bundle_preamble_past_buffer_bound_fails_closed(tmp_path, monkeypatch):
    from flash import env_opaque
    from flash.env_secrets import _Unscannable, credential_in_file

    monkeypatch.setattr(env_opaque, "_MAX_GIT_BUNDLE_HEADER_BYTES", 256)
    preamble = b"# v2 git bundle\n" + b"1" * 40 + b" refs/heads/main\n"
    preamble += (b"-" + b"2" * 40 + b" prerequisite\n") * 8
    path = tmp_path / "large.bundle"
    path.write_bytes(preamble + b"\n" + _git_pack())
    with pytest.raises(_Unscannable, match="Git bundle archive"):
        credential_in_file(path)


def test_uri_userinfo_passwords_are_detected_without_echoing_values(tmp_path):
    from flash.env_secrets import credential_in_file

    plain = tmp_path / "plain.txt"
    plain.write_bytes(b"endpoint=https://build:" + b"a1B2c3D4" * 4 + b"@example.com:443/path\n")
    assert credential_in_file(plain) == "a password"

    encoded = tmp_path / "encoded.txt"
    password = b"a1B2c3D4" * 4
    escaped = b"".join(
        b"%%%02X" % byte if index % 3 == 0 else bytes((byte,))
        for index, byte in enumerate(password)
    )
    encoded.write_bytes(b"ssh+https://builder:" + escaped + b"@git.example/repo\n")
    assert credential_in_file(encoded) == "a password"

    apostrophe = tmp_path / "apostrophe.txt"
    apostrophe.write_bytes(b'url="https://build:a1B2\'c3D4e5F6g7H8@[2001:db8::1]:443/path"\n')
    assert credential_in_file(apostrophe) == "a password"

    delimiters = tmp_path / "encoded-delimiters.txt"
    delimiters.write_bytes(b"https://build:a1B2%40c3D4%2Fe5F6g7H8@example.com/path\n")
    assert credential_in_file(delimiters) == "a password"


def test_uri_userinfo_false_positive_controls_remain_clean(tmp_path):
    from flash.env_secrets import credential_in_file

    controls = {
        "user-only.txt": b"https://user@example.com/path\n",
        "placeholder.txt": b"https://user:YOUR_PASSWORD_HERE@example.com/path\n",
        "docs.txt": b"use https://user:password@example.com in documentation\n",
        "short.txt": b"https://user:a1B2c3@example.com/path\n",
        "malformed.txt": b"https://user:a1B2%ZZc3D4e5F6g7H8@example.com/path\n",
        "malformed-user.txt": b"https://us%ZZer:a1B2c3D4e5F6g7H8@example.com/path\n",
        "empty-port.txt": b"https://user:a1B2c3D4e5F6g7H8@example.com:/path\n",
        "invalid-port.txt": b"https://user:a1B2c3D4e5F6g7H8@example.com:70000/path\n",
        "path-at.txt": b"https://example.com/path@user:a1B2c3D4e5F6g7H8\n",
        "query-at.txt": b"https://example.com?q=user:a1B2c3D4e5F6g7H8@host\n",
        "quoted-before-host.txt": b"url='https://user:a1B2c3D4e5F6g7H8'@example.com'\n",
        "no-authority.txt": b"https:user:a1B2c3D4e5F6g7H8@example.com/path\n",
        "boundary.txt": b".https://user:a1B2c3D4e5F6g7H8@example.com/path\n",
        "encoded-placeholder.txt": b"https://user:YOUR%5FPASSWORD%5FHERE@example.com/path\n",
    }
    for name, content in controls.items():
        path = tmp_path / name
        path.write_bytes(content)
        assert credential_in_file(path) is None


def test_uri_oversized_continuing_authority_fails_closed(tmp_path):
    from flash.env_secrets import credential_in_file

    path = tmp_path / "oversized-uri.txt"
    authority = b"user:a1B2c3D4e5F6g7H8@" + b"a" * 800 + b":70000"
    path.write_bytes(b"https://" + authority + b"/path\n")
    assert credential_in_file(path) == "a password"


@pytest.mark.parametrize("filter_kind", range(5))
def test_png_idat_unfiltering_reconstructs_credentials(tmp_path, filter_kind):
    from flash.env_secrets import credential_in_file

    previous = b"ordinary-pixel-row".ljust(len(_KEY), b".")
    rows = [(0, previous), (filter_kind, _filtered(_KEY, previous, filter_kind))]
    image = tmp_path / f"filter-{filter_kind}.png"
    image.write_bytes(_png(rows, width=len(_KEY)))
    assert credential_in_file(image) == "a Freesolo API key"


def test_png_fdat_unfiltering_reconstructs_credentials(tmp_path):
    from flash.env_secrets import credential_in_file

    ihdr = struct.pack(">IIBBBBB", len(_KEY), 1, 8, 0, 0, 0, 0)
    control = struct.pack(">IIIIIHHBB", 0, len(_KEY), 1, 0, 0, 0, 0, 0, 0)
    second_control = struct.pack(">IIIIIHHBB", 1, len(_KEY), 1, 0, 0, 0, 0, 0, 0)
    default = zlib.compress(b"\x00" + b"." * len(_KEY))
    frame = zlib.compress(b"\x01" + _filtered(_KEY, b"", 1))
    image = (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", ihdr)
        + _chunk(b"acTL", struct.pack(">II", 2, 0))
        + _chunk(b"fcTL", control)
        + _chunk(b"IDAT", default)
        + _chunk(b"fcTL", second_control)
        + _chunk(b"fdAT", (2).to_bytes(4, "big") + frame)
        + _chunk(b"IEND", b"")
    )
    path = tmp_path / "animated.png"
    path.write_bytes(image)
    assert credential_in_file(path) == "a Freesolo API key"


@pytest.mark.parametrize(
    ("bit_depth", "color_type", "width", "row"),
    [
        (1, 0, 8, b"\xaa"),
        (8, 0, 4, b"abcd"),
        (16, 0, 2, b"abcd"),
        (8, 2, 2, b"abcdef"),
        (8, 3, 4, b"\x00\x01\x00\x01"),
        (8, 4, 2, b"abcd"),
        (8, 6, 2, b"abcdefgh"),
    ],
)
def test_clean_png_sample_layouts_remain_clean(tmp_path, bit_depth, color_type, width, row):
    from flash.env_secrets import credential_in_file

    path = tmp_path / f"layout-{color_type}-{bit_depth}.png"
    path.write_bytes(_png([(0, row)], width=width, bit_depth=bit_depth, color_type=color_type))
    assert credential_in_file(path) is None


def test_png_malformed_unsupported_budget_and_deadline_fail_closed(tmp_path):
    from flash import env_deflate
    from flash.env_secrets import _Unscannable, credential_in_file

    unsupported = tmp_path / "interlaced.png"
    unsupported.write_bytes(_png([(0, b"a")], width=1, interlace=1))
    with pytest.raises(_Unscannable, match="filter this cannot undo"):
        credential_in_file(unsupported)

    bad_filter = tmp_path / "filter.png"
    bad_filter.write_bytes(_png([(5, b"a")], width=1))
    with pytest.raises(_Unscannable, match="filter this cannot undo"):
        credential_in_file(bad_filter)

    bad_crc = bytearray(_png([(0, b"a")], width=1))
    bad_crc[-5] ^= 1
    corrupt = tmp_path / "crc.png"
    corrupt.write_bytes(bad_crc)
    with pytest.raises(_Unscannable, match="filter this cannot undo"):
        credential_in_file(corrupt)

    image = _png([(0, b"abcdefgh")], width=8)
    assert list(env_deflate._document_payloads(image, 4)) == [None]
    with pytest.raises(env_deflate._DocumentDeadlineExceeded):
        list(env_deflate._document_payloads(image, 1024, deadline=0.0))


def test_png_metadata_and_icc_payloads_keep_existing_behavior(tmp_path):
    from flash.env_secrets import credential_in_file

    secret_text = tmp_path / "text.png"
    secret_text.write_bytes(
        _png(
            [(0, b"a")],
            width=1,
            extra=((b"zTXt", b"Comment\x00\x00" + zlib.compress(_KEY)),),
        )
    )
    assert credential_in_file(secret_text) == "a Freesolo API key"

    secret_icc = tmp_path / "icc.png"
    secret_icc.write_bytes(
        _png(
            [(0, b"a")],
            width=1,
            extra=((b"iCCP", b"ICC Profile\x00\x00" + zlib.compress(_KEY)),),
        )
    )
    assert credential_in_file(secret_icc) == "a Freesolo API key"

    clean = tmp_path / "clean-metadata.png"
    clean.write_bytes(
        _png(
            [(0, b"a")],
            width=1,
            extra=(
                (b"zTXt", b"Comment\x00\x00" + zlib.compress(b"ordinary comment")),
                (b"iCCP", b"ICC Profile\x00\x00" + zlib.compress(b"ordinary profile")),
            ),
        )
    )
    assert credential_in_file(clean) is None


@pytest.mark.parametrize(
    "payload",
    [
        b"\x89HDF\r\n\x1a\n" + bytes(64),
        _arrow_payload("file"),
        _arrow_payload("stream"),
        _git_pack(),
        _git_bundle(_git_pack()),
        _rpm_lead(),
    ],
)
def test_opaque_formats_fail_closed_inside_archives(tmp_path, payload):
    from flash.env_secrets import _Unscannable, credential_in_file

    archive = tmp_path / "nested.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as packed:
        packed.writestr("payload.bin", payload)
    with pytest.raises(_Unscannable, match="cannot expand"):
        credential_in_file(archive)


def test_rpm_leads_fail_closed_without_filename_guessing(tmp_path):
    from flash.env_secrets import _Unscannable, credential_in_file

    rpm = tmp_path / "payload.bin"
    rpm.write_bytes(_rpm_lead() + zlib.compress(_KEY))
    with pytest.raises(_Unscannable, match="RPM archive"):
        credential_in_file(rpm)

    controls = {
        "package.rpm": b"ordinary package notes\n",
        "short.bin": b"\xed\xab\xee\xdb\x03\x00",
        "wrong-version.bin": b"\xed\xab\xee\xdb\x04\x00" + _rpm_lead()[6:],
        "wrong-type.bin": _rpm_lead()[:6] + (3).to_bytes(2, "big") + _rpm_lead()[8:],
    }
    for name, content in controls.items():
        path = tmp_path / name
        path.write_bytes(content)
        assert credential_in_file(path) is None
