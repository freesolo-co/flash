"""Focused regressions for bounded environment credential scanning."""

from __future__ import annotations

import base64
import bz2
import gzip
import lzma
import random
import struct
import zipfile
import zlib

import pytest

_KEY_BODY = "a1B2c3D4" * 6
_KEY = f"fslo_{_KEY_BODY}".encode()


def _pdf_stream(dictionary: bytes, body: bytes) -> bytes:
    return (
        b"%PDF-1.7\n1 0 obj\n<< "
        + dictionary
        + b" /Length "
        + str(len(body)).encode()
        + b" >>\nstream\n"
        + body
        + b"\nendstream\nendobj\ntrailer\n<< /Root 1 0 R >>\n%%EOF\n"
    )


def _raw_deflate(data: bytes) -> bytes:
    compressor = zlib.compressobj(wbits=-zlib.MAX_WBITS)
    return compressor.compress(data) + compressor.flush()


def _ar(members: list[tuple[str, bytes]]) -> bytes:
    archive = bytearray(b"!<arch>\n")
    for name, body in members:
        archive += (f"{name + '/':<16}{0:<12}{0:<6}{0:<6}{0o100644:<8o}{len(body):<10}`\n").encode()
        archive += body
        if len(body) % 2:
            archive += b"\n"
    return bytes(archive)


def _jks(*tags: int, magic: bytes = b"\xfe\xed\xfe\xed") -> bytes:
    body = bytearray()
    for tag in tags:
        alias = f"entry-{tag}".encode()
        body += struct.pack(">I", tag) + struct.pack(">H", len(alias)) + alias
        body += struct.pack(">Q", 1_700_000_000_000)
        if tag in (1, 3):
            body += struct.pack(">I", 8) + b"keybytes"
            body += struct.pack(">I", 0)
        else:
            body += struct.pack(">H", 5) + b"X.509" + struct.pack(">I", 8) + b"certdata"
    return magic + struct.pack(">II", 2, len(tags)) + body + bytes(20)


def test_pdf_dictionary_index_is_single_pass_and_deadline_bounded(tmp_path, monkeypatch):
    from flash import env_deflate
    from flash.env_secrets import credential_in_file

    body = bytearray(b"%PDF-1.7\n")
    packed = zlib.compress(b"harmless\n")
    for index in range(32):
        body += (
            b"%d 0 obj\n<< /Filter /FlateDecode /Length %d >>\nstream\n%s\nendstream\nendobj\n"
            % (index + 1, len(packed), packed)
        )
    body += b"trailer\n<< /Root 1 0 R >>\n%%EOF\n"
    document = tmp_path / "many-streams.pdf"
    document.write_bytes(body)

    indexed = env_deflate.pdf_dictionary_spans
    calls = 0

    def counted(data, check):
        nonlocal calls
        calls += 1
        return indexed(data, check)

    monkeypatch.setattr(env_deflate, "pdf_dictionary_spans", counted)
    assert credential_in_file(document) is None
    assert calls == 1

    deadline_error = getattr(env_deflate, "_DocumentDeadlineExceeded", RuntimeError)
    with pytest.raises(deadline_error):
        list(env_deflate._document_payloads(bytes(body), 1 << 20, deadline=0.0))


def test_pdf_deadline_expires_inside_lexical_walk(monkeypatch):
    from flash import env_deflate

    packed = zlib.compress(b"harmless")
    document = _pdf_stream(b"/Note (" + b"x" * (32 << 10) + b") /Filter /FlateDecode", packed)
    calls = 0

    def clock():
        nonlocal calls
        calls += 1
        return 2.0 if calls >= 4 else 0.0

    monkeypatch.setattr(env_deflate.time, "monotonic", clock)
    with pytest.raises(env_deflate._DocumentDeadlineExceeded):
        list(env_deflate._document_payloads(document, 1 << 20, deadline=1.0))
    assert calls >= 4


def test_pdf_long_direct_filter_arrays_are_lexically_parsed(tmp_path):
    from flash.env_secrets import _Unscannable, credential_in_file

    gap = b" " * (70 << 10)
    encoded = base64.a85encode(zlib.compress(_KEY)) + b"~>"
    document = tmp_path / "long-filter-array.pdf"
    document.write_bytes(_pdf_stream(b"/Filter [/ASCII85Decode" + gap + b"/FlateDecode]", encoded))
    assert credential_in_file(document) == "a Freesolo API key"

    incomplete = tmp_path / "incomplete-filter-array.pdf"
    incomplete.write_bytes(_pdf_stream(b"/Filter [/ASCII85Decode" + gap, encoded))
    with pytest.raises(_Unscannable, match="filter this cannot undo"):
        credential_in_file(incomplete)

    unsupported = tmp_path / "unsupported-filter-array.pdf"
    unsupported.write_bytes(_pdf_stream(b"/Filter [/ASCII85Decode" + gap + b"/LZWDecode]", encoded))
    with pytest.raises(_Unscannable, match="filter this cannot undo"):
        credential_in_file(unsupported)

    lexical_control = tmp_path / "filter-lexical-control.pdf"
    lexical_control.write_bytes(
        _pdf_stream(
            b"/Note (/Filter [/LZWDecode]) % /Filter /LZWDecode\n /Filter/FlateDecode",
            zlib.compress(_KEY),
        )
    )
    assert credential_in_file(lexical_control) == "a Freesolo API key"


@pytest.mark.parametrize(
    ("declaration", "encode"),
    [
        (b"/F /Fl", lambda value: value),
        (b"/Filter /FlateDecode", lambda value: value),
        (b"/F [/A85 /Fl]", lambda value: base64.a85encode(value) + b"~>"),
        (b"/Filter [/ASCII85Decode /FlateDecode]", lambda value: base64.a85encode(value) + b"~>"),
    ],
)
def test_pdf_inline_image_supported_filter_chains_are_inspected(tmp_path, declaration, encode):
    from flash.env_secrets import credential_in_file

    payload = encode(zlib.compress(_KEY))
    inline = b"BI /W 1 /H 1 " + declaration + b" ID " + payload + b" EI\n"
    document = tmp_path / "inline.pdf"
    document.write_bytes(_pdf_stream(b"", inline))
    assert credential_in_file(document) == "a Freesolo API key"


@pytest.mark.parametrize("filter_name", [b"/AHx", b"/ASCIIHexDecode", b"/LZW", b"/LZWDecode"])
def test_pdf_inline_image_unsupported_filters_fail_closed(tmp_path, filter_name):
    from flash.env_secrets import _Unscannable, credential_in_file

    inline = b"BI /W 1 /H 1 /F " + filter_name + b" ID " + _KEY.hex().encode() + b"> EI\n"
    document = tmp_path / "inline-unsupported.pdf"
    document.write_bytes(_pdf_stream(b"", inline))
    with pytest.raises(_Unscannable, match="filter this cannot undo"):
        credential_in_file(document)


def test_pdf_inline_image_escaped_filter_keys_preserve_boundaries(tmp_path):
    from flash.env_secrets import _Unscannable, credential_in_file

    supported = tmp_path / "inline-escaped-supported.pdf"
    supported.write_bytes(_pdf_stream(b"", b"BI /#46 /Fl ID " + zlib.compress(_KEY) + b" EI\n"))
    assert credential_in_file(supported) == "a Freesolo API key"

    supported_long = tmp_path / "inline-escaped-long-supported.pdf"
    supported_long.write_bytes(
        _pdf_stream(b"", b"BI /#46ilter /FlateDecode ID " + zlib.compress(_KEY) + b" EI\n")
    )
    assert credential_in_file(supported_long) == "a Freesolo API key"

    unsupported = tmp_path / "inline-escaped-unsupported.pdf"
    unsupported.write_bytes(_pdf_stream(b"", b"BI /#46ilter /LZW ID opaque EI\n"))
    with pytest.raises(_Unscannable, match="filter this cannot undo"):
        credential_in_file(unsupported)

    boundary = tmp_path / "inline-filter-boundary.pdf"
    boundary.write_bytes(
        _pdf_stream(
            b"",
            b"BI /Filterish /LZW /F /Fl ID " + zlib.compress(_KEY) + b" EI\n",
        )
    )
    assert credential_in_file(boundary) == "a Freesolo API key"


def test_pdf_inline_image_markers_ignore_strings_and_comments(tmp_path):
    from flash.env_secrets import credential_in_file

    controls = (
        b"(BI /F /LZW ID opaque EI)",
        b"(outer \\( BI /F /LZW ID opaque \\) inner)",
        b"% BI /F /LZW ID opaque EI\nq",
    )
    for index, content in enumerate(controls):
        document = tmp_path / f"inline-lexical-{index}.pdf"
        document.write_bytes(_pdf_stream(b"", content))
        assert credential_in_file(document) is None

    real = tmp_path / "inline-real-supported.pdf"
    real.write_bytes(_pdf_stream(b"", b"BI /F /Fl ID " + zlib.compress(_KEY) + b" EI\n"))
    assert credential_in_file(real) == "a Freesolo API key"


def test_ar_members_preserve_resolved_filename_context(tmp_path):
    from flash.env_secrets import credential_in_file

    escaped = _KEY.decode().replace("B", "\\x42", 1).encode()
    standalone = tmp_path / "member.sh"
    standalone.write_bytes(b"API_KEY='" + escaped + b"'\n")
    assert credential_in_file(standalone) is None

    archive = tmp_path / "scripts.a"
    archive.write_bytes(_ar([("member.sh", standalone.read_bytes())]))
    assert credential_in_file(archive) is None

    keyed = tmp_path / "keyed.a"
    keyed.write_bytes(_ar([("member.sh", b"API_KEY='" + _KEY + b"'\n")]))
    assert credential_in_file(keyed) == "a Freesolo API key"


def test_parquet_magic_fails_closed_without_using_the_filename(tmp_path):
    from flash.env_secrets import _Unscannable, credential_in_file

    parquet = tmp_path / "dataset.bin"
    footer = b"minimal parquet metadata"
    parquet.write_bytes(
        b"PAR1" + b"opaque column data" + footer + len(footer).to_bytes(4, "little") + b"PAR1"
    )
    with pytest.raises(_Unscannable, match="Parquet"):
        credential_in_file(parquet)

    fake = tmp_path / "dataset.parquet"
    fake.write_text("ordinary rows with no credential\n")
    assert credential_in_file(fake) is None


@pytest.mark.parametrize(
    ("name", "contents"),
    [
        ("literal.yaml", "note: 'quoted ''text'' before fslo_AbCd\\x45f0123456789AbCdEf'\n"),
        ("literal.toml", "note = 'text before fslo_AbCd\\x45f0123456789AbCdEf'\n"),
        ("multiline.toml", "note = '''\ntext before fslo_AbCd\\x45f0123456789AbCdEf\n'''\n"),
    ],
)
def test_yaml_and_toml_literal_strings_preserve_backslashes(tmp_path, name, contents):
    from flash.env_secrets import credential_in_file

    literal = tmp_path / name
    literal.write_text(contents)
    assert credential_in_file(literal) is None

    basic = tmp_path / "basic.toml"
    basic.write_text('note = "fslo_AbCd\\x45f0123456789AbCdEf"\n')
    assert credential_in_file(basic) == "a Freesolo API key"


def test_yaml_and_toml_literal_lexers_ignore_apostrophes_in_other_states(tmp_path):
    from flash.env_secrets import credential_in_file

    escaped = "fslo_AbCd\\x45f0123456789AbCdEf"
    yaml_basic = tmp_path / "adversarial.yaml"
    yaml_basic.write_text(f"note: \"apostrophe ' before {escaped} ' after\"\n")
    assert credential_in_file(yaml_basic) == "a Freesolo API key"

    yaml_comment = tmp_path / "comment.yaml"
    yaml_comment.write_text(f'# apostrophe \'\nnote: "{escaped}"\n')
    assert credential_in_file(yaml_comment) == "a Freesolo API key"

    yaml_block = tmp_path / "block.yaml"
    yaml_block.write_text(f"note: |\n  apostrophe ' and literal {escaped}\nnext: harmless\n")
    assert credential_in_file(yaml_block) is None

    toml_basic = tmp_path / "adversarial.toml"
    toml_basic.write_text(f"note = \"apostrophe ' before {escaped} ' after\"\n")
    assert credential_in_file(toml_basic) == "a Freesolo API key"

    toml_multiline = tmp_path / "multiline-basic.toml"
    toml_multiline.write_text(f'note = """apostrophe \' before {escaped} \' after"""\n')
    assert credential_in_file(toml_multiline) == "a Freesolo API key"

    toml_comment = tmp_path / "comment.toml"
    toml_comment.write_text(f'# apostrophe \'\nnote = "{escaped}"\n')
    assert credential_in_file(toml_comment) == "a Freesolo API key"

    literal = tmp_path / "literal-control.toml"
    literal.write_text(f"note = '{escaped}'\n")
    assert credential_in_file(literal) is None


@pytest.mark.parametrize(
    ("suffix", "compress"),
    [
        ("gz", lambda value: gzip.compress(value, mtime=0)),
        ("bz2", bz2.compress),
        ("xz", lzma.compress),
        ("lzma", lambda value: lzma.compress(value, format=lzma.FORMAT_ALONE)),
        (
            "lzma-small-dict",
            lambda value: lzma.compress(
                value,
                format=lzma.FORMAT_ALONE,
                filters=[{"id": lzma.FILTER_LZMA1, "dict_size": 4096}],
            ),
        ),
    ],
)
def test_nested_overflow_retains_overlay_refusal(tmp_path, monkeypatch, suffix, compress):
    from flash import env_secrets

    monkeypatch.setattr(env_secrets, "_MAX_NESTED_BUFFER_BYTES", 64 << 10)
    member = b"#!/bin/sh\n" + b"x" * (70 << 10) + compress(_KEY)
    archive = tmp_path / f"overlay-{suffix}.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zipped:
        zipped.writestr(f"installer.{suffix}.run", member)
    with pytest.raises(env_secrets._Unscannable, match="appended archive"):
        env_secrets.credential_in_file(archive)

    plain = tmp_path / f"plain-{suffix}.zip"
    with zipfile.ZipFile(plain, "w", zipfile.ZIP_DEFLATED) as zipped:
        zipped.writestr("large.txt", (b"ordinary text and numbers 12345\n" * 3000)[: 70 << 10])
    assert env_secrets.credential_in_file(plain) is None


def test_oversized_base64_refusal_is_limited_to_containers(tmp_path):
    from flash.env_secrets import _Unscannable, credential_in_file

    ordinary = tmp_path / "image.b64"
    ordinary.write_bytes(base64.b64encode(b"A" * 3_200_000))
    assert len(ordinary.read_bytes()) > 4 << 20
    assert credential_in_file(ordinary) is None

    false_zlib = tmp_path / "false-zlib.b64"
    false_zlib.write_bytes(base64.b64encode(b"\x78\x9c" + b"\xff" * 3_200_000))
    assert len(false_zlib.read_bytes()) > 4 << 20
    assert credential_in_file(false_zlib) is None

    fdict = tmp_path / "fdict.b64"
    fdict.write_bytes(base64.b64encode(b"\x78\x20" + b"\xff" * 3_200_000))
    with pytest.raises(_Unscannable, match="base64 run too long"):
        credential_in_file(fdict)

    rng = random.Random(20260815)
    packed = gzip.compress(rng.randbytes(3_250_000), mtime=0) + gzip.compress(_KEY, mtime=0)
    encoded = base64.b64encode(packed)
    assert len(encoded) > 4 << 20
    assert _KEY not in packed
    container = tmp_path / "container.b64"
    container.write_bytes(encoded)
    with pytest.raises(_Unscannable, match="base64 run too long"):
        credential_in_file(container)

    zlib_container = tmp_path / "zlib-container.b64"
    zlib_encoded = base64.b64encode(zlib.compress(rng.randbytes(3_250_000) + _KEY))
    assert len(zlib_encoded) > 4 << 20
    zlib_container.write_bytes(zlib_encoded)
    with pytest.raises(_Unscannable, match="base64 run too long"):
        credential_in_file(zlib_container)


@pytest.mark.parametrize("location", ["comment", "extra"])
def test_zip_metadata_uses_the_full_bounded_scanner(tmp_path, location):
    from flash.env_secrets import credential_in_file

    packed = zlib.compress(_KEY)
    archive = tmp_path / f"metadata-{location}.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        info = zipfile.ZipInfo("harmless.txt")
        if location == "extra":
            info.extra = b"ZZ" + len(packed).to_bytes(2, "little") + packed
        zipped.writestr(info, b"harmless\n")
        if location == "comment":
            zipped.comment = packed
    assert credential_in_file(archive) == "a Freesolo API key"


def test_openpgp_exact_nonfinal_packet_boundary_is_undecided(tmp_path):
    from flash.env_buffers import _SCAN_CHUNK_BYTES
    from flash.env_secrets import _Unscannable, credential_in_file

    body_size = _SCAN_CHUNK_BYTES - 5
    public = b"\x9a" + body_size.to_bytes(4, "big") + b"\x04\0\0\0\0\x01" + bytes(body_size - 6)
    assert len(public) == _SCAN_CHUNK_BYTES
    secret = b"\xc5\x06\x04\0\0\0\0\x01"

    compact = tmp_path / "compact.pgp"
    compact.write_bytes(b"\x99\0\x06\x04\0\0\0\0\x01" + secret)
    assert credential_in_file(compact) == "a private key"

    boundary = tmp_path / "boundary.pgp"
    boundary.write_bytes(public + secret)
    with pytest.raises(_Unscannable, match="cannot walk to the end"):
        credential_in_file(boundary)


def test_long_openpgp_message_headers_do_not_cross_windows_cleanly(tmp_path):
    from flash.env_buffers import _SCAN_CHUNK_BYTES
    from flash.env_secrets import _Unscannable, credential_in_file

    armored = tmp_path / "message.asc"
    armored.write_bytes(
        b"-----BEGIN PGP MESSAGE-----\nComment: "
        + b"x" * (_SCAN_CHUNK_BYTES + 4096)
        + b"\n\n"
        + b"A" * 64
        + b"\n-----END PGP MESSAGE-----\n"
    )
    with pytest.raises(_Unscannable, match="OpenPGP message armor header"):
        credential_in_file(armored)

    prose = tmp_path / "README.md"
    prose.write_text("documentation mentions -----BEGIN PGP MESSAGE----- as marker prose\n")
    assert credential_in_file(prose) is None

    exact_line = tmp_path / "MARKERS.md"
    exact_line.write_text("-----BEGIN PGP MESSAGE-----\nthis line explains the marker\n")
    assert credential_in_file(exact_line) is None


def test_terminal_brotli_sidecars_are_rejected_by_name(tmp_path):
    from flash.env_secrets import _Unscannable, credential_in_file, credential_in_name

    top_level = tmp_path / "payload.br"
    top_level.write_bytes(b"opaque brotli bytes")
    with pytest.raises(_Unscannable, match="Brotli"):
        credential_in_file(top_level)
    with pytest.raises(_Unscannable, match="Brotli"):
        credential_in_name("nested/payload.br")

    archive = tmp_path / "brotli.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr("nested/payload.br", b"opaque brotli bytes")
    with pytest.raises(_Unscannable, match="Brotli"):
        credential_in_file(archive)

    fake = tmp_path / "payload.br.txt"
    fake.write_text("ordinary text\n")
    assert credential_in_file(fake) is None


def test_concatenated_raw_deflate_records_share_bounds(tmp_path):
    from flash.env_secrets import credential_in_file

    concatenated = tmp_path / "records.deflate"
    concatenated.write_bytes(
        _raw_deflate(b"harmless\n") + _raw_deflate(b"A" * 10_000 + _KEY + b"A" * 10_000)
    )
    assert credential_in_file(concatenated) == "a Freesolo API key"

    empty_first = tmp_path / "empty-first.deflate"
    empty_first.write_bytes(_raw_deflate(b"") + _raw_deflate(_KEY))
    assert credential_in_file(empty_first) == "a Freesolo API key"

    partial_key = _raw_deflate(_KEY + b"x" * 4096)[:-1]
    probe = zlib.decompressobj(-zlib.MAX_WBITS)
    emitted = probe.decompress(partial_key)
    assert _KEY in emitted
    assert not probe.eof
    truncated_key = tmp_path / "truncated-key.deflate"
    truncated_key.write_bytes(_raw_deflate(b"harmless\n") + partial_key)
    assert credential_in_file(truncated_key) == "a Freesolo API key"

    partial_harmless = _raw_deflate(b"harmless\n" + b"x" * 4096)[:-1]
    truncated_harmless = tmp_path / "truncated-harmless.deflate"
    truncated_harmless.write_bytes(_raw_deflate(b"harmless\n") + partial_harmless)
    assert credential_in_file(truncated_harmless) is None

    footer = tmp_path / "footer.deflate"
    footer.write_bytes(_raw_deflate(b"harmless\n") + b"invalid footer")
    assert credential_in_file(footer) is None


def test_exact_base64_decodes_run_openpgp_and_keystore_checks(tmp_path):
    from flash.env_secrets import credential_in_file

    secret_packet = b"\xc5\x20\x04\0\0\0\0\x01" + bytes(26)
    public_packet = b"\xc6" + secret_packet[1:]
    key_store = _jks(1)
    secret_store = _jks(3, magic=b"\xce\xce\xce\xce")
    trust_store = _jks(2)

    for name, raw, expected in (
        ("openpgp", secret_packet, "a private key"),
        ("openpgp-sequence", public_packet + secret_packet, "a private key"),
        ("openpgp-marker", b"\xca\x03PGP" + secret_packet, "a private key"),
        ("jks", key_store, "a key store"),
        ("jceks", secret_store, "a key store"),
    ):
        control = tmp_path / f"{name}.bin"
        control.write_bytes(raw)
        assert credential_in_file(control) == expected

        encoded = tmp_path / f"{name}.yaml"
        encoded.write_text("value: " + base64.b64encode(raw).decode() + "\n")
        assert credential_in_file(encoded) == expected

    trust = tmp_path / "trust.yaml"
    trust.write_text("value: " + base64.b64encode(trust_store).decode() + "\n")
    assert credential_in_file(trust) is None
