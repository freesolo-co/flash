"""Focused regressions for bounded environment credential scanning."""

from __future__ import annotations

import base64
import gzip
import struct
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


def _avro_long(value: int) -> bytes:
    encoded = bytearray()
    value <<= 1
    while value > 0x7F:
        encoded.append((value & 0x7F) | 0x80)
        value >>= 7
    encoded.append(value)
    return bytes(encoded)


def _avro_bytes(value: bytes) -> bytes:
    return _avro_long(len(value)) + value


def _avro_ocf(value: bytes) -> bytes:
    schema = b'{"type":"bytes"}'
    metadata = (
        _avro_long(2)
        + _avro_bytes(b"avro.schema")
        + _avro_bytes(schema)
        + _avro_bytes(b"avro.codec")
        + _avro_bytes(b"deflate")
        + _avro_long(0)
    )
    sync = b"0123456789abcdef"
    plain = _avro_bytes(value)
    compressor = zlib.compressobj(wbits=-zlib.MAX_WBITS)
    block = compressor.compress(plain) + compressor.flush()
    return b"Obj\x01" + metadata + sync + _avro_long(1) + _avro_bytes(block) + sync


def _gzip_with_extra(extra: bytes, payload: bytes = b"harmless\n") -> bytes:
    packed = gzip.compress(payload, mtime=0)
    header = bytearray(packed[:10])
    header[3] |= 0x04
    field = b"ZZ" + len(extra).to_bytes(2, "little") + extra
    return bytes(header) + len(field).to_bytes(2, "little") + field + packed[10:]


def _crc32c(value: bytes) -> int:
    crc = 0xFFFFFFFF
    for byte in value:
        crc ^= byte
        for _ in range(8):
            crc = (crc >> 1) ^ (0x82F63B78 if crc & 1 else 0)
    return crc ^ 0xFFFFFFFF


def _snappy_key_frame() -> bytes:
    literal = b"fslo_a1B2c3D4"
    copy_length = len(_KEY) - len(literal)
    raw = bytes((len(_KEY), (len(literal) - 1) << 2)) + literal
    raw += bytes((((copy_length - 1) << 2) | 2, 8, 0))
    checksum = _crc32c(_KEY)
    masked = ((checksum >> 15) | (checksum << 17) & 0xFFFFFFFF) + 0xA282EAD8
    chunk = (masked & 0xFFFFFFFF).to_bytes(4, "little") + raw
    return b"\xff\x06\x00\x00sNaPpY\x00" + len(chunk).to_bytes(3, "little") + chunk


def test_pdf_dictionary_index_is_single_pass_and_deadline_bounded(tmp_path, monkeypatch):
    from flash.envscan import deflate as env_deflate
    from flash.envscan.secrets import credential_in_file

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
    from flash.envscan import deflate as env_deflate

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
    from flash.envscan.secrets import _Unscannable, credential_in_file

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
    from flash.envscan.secrets import credential_in_file

    payload = encode(zlib.compress(_KEY))
    inline = b"BI /W 1 /H 1 " + declaration + b" ID " + payload + b" EI\n"
    document = tmp_path / "inline.pdf"
    document.write_bytes(_pdf_stream(b"", inline))
    assert credential_in_file(document) == "a Freesolo API key"


@pytest.mark.parametrize("filter_name", [b"/AHx", b"/ASCIIHexDecode", b"/LZW", b"/LZWDecode"])
def test_pdf_inline_image_unsupported_filters_fail_closed(tmp_path, filter_name):
    from flash.envscan.secrets import _Unscannable, credential_in_file

    inline = b"BI /W 1 /H 1 /F " + filter_name + b" ID " + _KEY.hex().encode() + b"> EI\n"
    document = tmp_path / "inline-unsupported.pdf"
    document.write_bytes(_pdf_stream(b"", inline))
    with pytest.raises(_Unscannable, match="filter this cannot undo"):
        credential_in_file(document)


def test_pdf_inline_image_escaped_filter_keys_preserve_boundaries(tmp_path):
    from flash.envscan.secrets import _Unscannable, credential_in_file

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
    from flash.envscan.secrets import credential_in_file

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


def test_pdf_inline_image_decode_parameters_are_lexically_paired(tmp_path):
    from flash.envscan.secrets import _Unscannable, credential_in_file

    def document(name: str, header: bytes, payload: bytes):
        path = tmp_path / name
        path.write_bytes(_pdf_stream(b"", b"BI /W 1 /H 1 " + header + b" ID " + payload + b" EI\n"))
        return path

    packed = zlib.compress(_KEY)
    identities = (
        (b"/F /Fl", packed),
        (b"/F /Fl /DP null", packed),
        (b"/F /Fl /DP << /Predictor 1 >>", packed),
        (b"/Filter /FlateDecode /DecodeParms << /Predictor 1 >>", packed),
        (b"/#46 /Fl /#44P << /#50redictor 1 >>", packed),
        (b"/Filter /FlateDecode /#44ecodeParms << /#50redictor 1 >>", packed),
        (
            b"/F [/A85 /Fl] /DP [null << /Predictor 1 >>]",
            base64.a85encode(packed) + b"~>",
        ),
        (
            b"/Note (/DP << /Predictor 12 >>) % /DP << /Predictor 12 >>\n /F /Fl",
            packed,
        ),
    )
    for index, (header, payload) in enumerate(identities):
        assert credential_in_file(document(f"inline-identity-{index}.pdf", header, payload)) == (
            "a Freesolo API key"
        )

    unreadable = (
        b"/F /Fl /DP << /Predictor 12 >>",
        b"/F /Fl /DP 2 0 R",
        b"/F /Fl /DP << /Predictor /One >>",
        b"/F /Fl /DP [",
        b"/F [/A85 /Fl] /DP [null]",
        b"/F [/A85 /Fl] /DP [null << /Predictor 12 >>]",
    )
    for index, header in enumerate(unreadable):
        payload = base64.a85encode(packed) + b"~>" if b"/A85" in header else packed
        with pytest.raises(_Unscannable, match="filter this cannot undo"):
            credential_in_file(document(f"inline-unreadable-{index}.pdf", header, payload))


def test_parquet_magic_fails_closed_without_using_the_filename(tmp_path):
    from flash.envscan.secrets import _Unscannable, credential_in_file

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


def test_framed_snappy_streams_fail_closed_on_the_complete_identifier(tmp_path):
    from flash.envscan.secrets import _Unscannable, credential_in_file

    framed = _snappy_key_frame()
    assert _KEY not in framed
    stream = tmp_path / "snappy.bin"
    stream.write_bytes(framed)
    with pytest.raises(_Unscannable, match="Snappy"):
        credential_in_file(stream)

    incomplete = tmp_path / "incomplete-snappy.bin"
    incomplete.write_bytes(b"\xff\x06\x00\x00sNaPp")
    assert credential_in_file(incomplete) is None

    prose = tmp_path / "snappy.txt"
    prose.write_bytes(b"the standard stream identifier spells sNaPpY in documentation\n")
    assert credential_in_file(prose) is None


def test_npm_access_tokens_require_exact_issued_boundaries(tmp_path):
    from flash.envscan.secrets import credential_in_file

    body = b"Ab3dE5fG7hJ9kLmN2pQr4sTu6vWx8yZ01aB2"
    assert len(body) == 36
    token = b"npm_" + body
    valid = tmp_path / "npm-token.txt"
    valid.write_bytes(token)
    assert credential_in_file(valid) == "an npm access token"

    punctuated = tmp_path / "npm-punctuated.txt"
    punctuated.write_bytes(token + b",")
    assert credential_in_file(punctuated) == "an npm access token"

    controls = (
        b"npm_" + b"A" * 36,
        b"npm_" + b"a" * 36,
        b"npm_" + b"1" * 36,
        b"npm_" + body[:-1],
        b"npm_" + body + b"X",
        b"x" + token,
        b"_" + token,
        token + b"_",
    )
    for index, content in enumerate(controls):
        control = tmp_path / f"npm-control-{index}.txt"
        control.write_bytes(content)
        assert credential_in_file(control) is None


def test_ansible_vault_requires_a_supported_header_and_hex_body(tmp_path):
    from flash.envscan.secrets import _Unscannable, credential_in_file

    def vault_body(ciphertext: bytes = b"c" * 64) -> bytes:
        payload = b"a" * 64 + b"\n" + b"b" * 64 + b"\n" + ciphertext
        wrapped = payload.hex().encode()
        return (
            b"\n".join(wrapped[index : index + 80] for index in range(0, len(wrapped), 80)) + b"\n"
        )

    ciphertext = vault_body()
    standalone = tmp_path / "env_secrets.txt"
    standalone.write_bytes(b"$ANSIBLE_VAULT;1.1;AES256\n" + ciphertext)
    with pytest.raises(_Unscannable, match="Ansible Vault"):
        credential_in_file(standalone)

    embedded = tmp_path / "env_secrets.yaml"
    embedded.write_bytes(
        b"password: !vault |\n  $ANSIBLE_VAULT;1.2;AES256;production\n  "
        + ciphertext.replace(b"\n", b"\n  ")
    )
    with pytest.raises(_Unscannable, match="Ansible Vault"):
        credential_in_file(embedded)

    arbitrary_hex = b"0123456789abcdef" * 4 + b"\n"
    uppercase_payload = vault_body(b"C" * 64)
    controls = (
        b"documentation mentions $ANSIBLE_VAULT;1.1;AES256 inline\n",
        b"$ANSIBLE_VAULT;1.1;AES256\n",
        b"$ANSIBLE_VAULT;1.1;AES256\n" + arbitrary_hex,
        b"$ANSIBLE_VAULT;1.3;AES256\n" + ciphertext,
        b"$ANSIBLE_VAULT;1.2;AES256;\n" + ciphertext,
        b"$ANSIBLE_VAULT;1.1;AES256\n0123456789abcdef-not-hex\n",
        b"$ANSIBLE_VAULT;1.1;AES256\n" + uppercase_payload,
    )
    for index, content in enumerate(controls):
        control = tmp_path / f"ansible-control-{index}.txt"
        control.write_bytes(content)
        assert credential_in_file(control) is None

    fake_name = tmp_path / "ordinary.vault"
    fake_name.write_text("ordinary rows with no credential\n")
    assert credential_in_file(fake_name) is None


def test_container_timeout_cannot_be_suppressed_by_a_settled_handler(tmp_path, monkeypatch):
    from flash.envscan import secrets as env_secrets

    keyed = tmp_path / "keyed.gz"
    keyed.write_bytes(gzip.compress(_KEY, mtime=0))
    assert env_secrets.credential_in_file(keyed) == "a Freesolo API key"

    harmless = tmp_path / "harmless.gz"
    harmless.write_bytes(gzip.compress(b"ordinary rows\n", mtime=0))
    assert env_secrets.credential_in_file(harmless) is None

    expired = False

    def settled(*_args, **_kwargs):
        return None

    def timeout(*_args, **_kwargs):
        nonlocal expired
        expired = True
        raise env_secrets._Unscannable("takes too long to decompress")

    monkeypatch.setattr(env_secrets, "_credential_in_zip", settled)
    monkeypatch.setattr(env_secrets, "_credential_in_tar", settled)
    monkeypatch.setattr(env_secrets, "_credential_in_ar", settled)
    monkeypatch.setattr(env_secrets, "_credential_in_compressed", timeout)
    monkeypatch.setattr(env_secrets, "_credential_in_overlay", settled)
    monkeypatch.setattr(env_secrets, "_credential_in_raw_deflate", settled)
    monkeypatch.setattr(env_secrets, "_credential_in_pdf", settled)
    monkeypatch.setattr(env_secrets.time, "monotonic", lambda: 2.0 if expired else 0.0)
    with pytest.raises(env_secrets._Unscannable, match="too long"):
        env_secrets._credential_in_container(b"not a container", deadline=1.0, depth=1)

    expired = False

    def format_refusal(*_args, **_kwargs):
        raise env_secrets._Unscannable("not this speculative format")

    monkeypatch.setattr(env_secrets, "_credential_in_compressed", format_refusal)
    assert env_secrets._credential_in_container(b"not a container", deadline=1.0, depth=1) is None


def test_gzip_metadata_uses_name_and_raw_container_scanning(tmp_path):
    from flash.envscan.secrets import credential_in_file

    nested = tmp_path / "nested-extra.gz"
    nested.write_bytes(_gzip_with_extra(zlib.compress(_KEY)))
    assert credential_in_file(nested) == "a Freesolo API key"

    named = tmp_path / "named-extra.gz"
    named.write_bytes(_gzip_with_extra(_KEY))
    assert credential_in_file(named) == "a Freesolo API key"

    harmless = tmp_path / "harmless-extra.gz"
    harmless.write_bytes(_gzip_with_extra(zlib.compress(b"ordinary metadata\n")))
    assert credential_in_file(harmless) is None


def test_concatenated_raw_deflate_records_share_bounds(tmp_path):
    from flash.envscan.secrets import credential_in_file

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
