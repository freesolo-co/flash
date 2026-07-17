# Image color training example

This package contains one 2 by 2 red PNG and one training row. The reward accepts only the exact lowercase answer `red` after surrounding whitespace is stripped.

```bash
cd examples/image-color
flash env push --name image-color .
```

Paste the returned environment id into `configs/sft.toml` or `configs/grpo.toml`, then run a dry run before submitting paid GPU work.

```bash
flash train configs/sft.toml --dry-run
flash train configs/grpo.toml --dry-run
```

The image stays under the singular `dataset/` package boundary and the record references it as `dataset/red.png`.
