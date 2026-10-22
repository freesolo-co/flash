"""Malformed image metadata must fail at validation without escaping package bounds."""

import json

import pytest

from flash.content import multimodal as mm
from flash.content.reasoning_normalization import messages_for_chat_template


@pytest.mark.parametrize("value", [[], {}])
@pytest.mark.parametrize("field", ["role", "type", "detail"])
def test_image_prompt_rejects_non_string_tags_with_value_error(value, field):
    block = {"type": "image_url", "image_url": "dataset/image.png"}
    message = {"role": "user", "content": [block]}
    if field == "role":
        message[field] = value
    else:
        block[field] = value
    with pytest.raises(ValueError, match=r"role|block"):
        mm.normalize_prompt_images({}, [message], None)


@pytest.mark.parametrize("value", [[], {}])
def test_content_classifiers_tolerate_unknown_block_type_shapes(value):
    content = [{"type": value, "text": "ignored"}]
    assert mm.content_has_images(content) is False
    assert mm.completion_has_images([{"content": content}]) is False
    assert mm.message_content_text(content) == ""
    message = {"role": "assistant", "content": content}
    assert messages_for_chat_template([message]) == [message]


def test_image_descriptor_rejects_dataset_directory_symlink_escape(tmp_path):
    root = tmp_path / "package"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "private.png").write_bytes(b"private-file")
    (root / "dataset").symlink_to(outside, target_is_directory=True)
    descriptor = json.dumps({"kind": "path", "value": "dataset/private.png"})

    with pytest.raises(ValueError, match="dataset/"):
        mm.read_descriptor_source(descriptor, root)


def test_missing_decoded_image_is_a_validation_error():
    message = {"role": "user", "content": [{"type": "image"}]}
    with pytest.raises(ValueError, match="more image blocks"):
        mm.messages_with_decoded_images([message], [])
