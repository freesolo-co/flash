# Managed OPD teacher refresh — 2026-10-22

The previous DeepSeek Pro route returned HTTP 404 during hosted OPD after GPU allocation. This refresh selects DeepSeek V4.1 Flash 1017, upgrades the default teacher to GLM 5.3, and adds Qwen3.8 27B. Kimi K3 remains the latest Kimi in the authenticated Parasail catalog. The existing Qwen3.5 397B and Qwen3-VL 235B options remain available: the newer 27B checkpoint is a different size, not an equivalent replacement for either large teacher.

## Selected routes and quote rates

| Flash alias | Parasail route | Input / output USD per million tokens | Image OPD |
| --- | --- | --- | --- |
| `glm-5.3` (default) | `parasail-glm-53` | 1.40 / 4.40 | No |
| `deepseek-v4.1-flash-1017` | `parasail-deepseek-v41-flash-1017` | 0.30 / 1.20 | No |
| `qwen3.8-27b` | `parasail-qwen38-27b` | 0.24 / 2.20 | Yes |
| `kimi-k3` | `parasail-kimi-k3-fast` | 3.00 / 15.00 | No |
| `qwen3.5-397b-a17b` | `parasail-qwen35-397b-a17b` | 0.50 / 3.60 | Yes |
| `qwen3-vl-235b` | `parasail-qwen3-vl-235b-a22b-instruct` | 0.21 / 1.90 | Yes |

Rates were checked against [Parasail's published serverless pricing](https://www.parasail.io/pricing). Qwen3.8 is absent from that table; its quote uses [OpenRouter's live Parasail endpoint listing](https://openrouter.ai/api/v1/models/qwen/qwen3.8-27b/endpoints). That listing also corroborates [DeepSeek 1017's rate](https://openrouter.ai/api/v1/models/deepseek/deepseek-v4.1-flash-1017/endpoints). These are quote inputs, not confirmation of a direct-account invoice or negotiated discount.

## Tokenizer pins

| Teacher | Official repository | Revision | SHA-256 of tokenizer.json |
| --- | --- | --- | --- |
| DeepSeek | [DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) | `1bae0e1088993b9b82de08f59de5be5a34e42170` | `c90dfa01249db1be4245780a052ede752e1361c612ac6d08e2bdada7d599476b` |
| GLM | [GLM-5.3](https://huggingface.co/zai-org/GLM-5.3) | `ba32f8c19c8738e2c57c71798c601d477a7bf844` | `19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d` |
| Qwen | [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` | `0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3` |

Each pin was downloaded and hashed. DeepSeek 1017's tokenizer differs from the former Pro tokenizer; changing the provider route without updating this pin would fail exact token alignment. GLM 5.3's tokenizer bytes match the previous GLM 5.2 pin.

## Live qualification

Authenticated calls went directly to Parasail using the deployment's existing credential, held only in memory. No deployment configuration or database records were changed.

- All six selected routes appear in `/v1/models`.
- The three new routes passed nine `/v1/completions` requests: ordinary text, mixed Unicode plus JSON, and a Qwen-formatted conversation. Responses passed the actual Flash broker validator and worker normalizer, including exact local/provider token-ID equality and realized completion scores.
- Qwen3.8 passed eight `/v1/chat/completions` image-scoring cases: red/blue PNGs, 32/224-pixel sizes, a separate assistant turn or continued thinking prefill with a leading newline. Responses passed the actual chat broker validator and `TeacherClient._score_one_multimodal`, including image-pad expansion and completion-span attribution.
- A separate Qwen3.8 visual check named red and blue correctly. Increasing the image from 32 to 1024 pixels increased reported image tokens from 64 to 1024, confirming that the endpoint processes image content.
- DeepSeek remains text-only in Flash despite native model vision support. Its tokenizer does not encode Flash's required `<|image_pad|>` and `<|im_end|>` markers as single tokens. Enabling image OPD requires a separate teacher-template implementation and qualification.
- Kimi K3 passed ordinary and Unicode scoring but returned HTTP 500 for the Qwen-formatted prompt, on an initial call and repeat through the fast route and again through the standard route. This is an unresolved provider-side limitation; switching Kimi routes did not fix it. Its model and route are unchanged by this refresh.

These checks qualify provider scoring and token alignment. They are not a hosted GPU training or serving rerun of the changed release, and this PR does not deploy it. The broader production audit covers the preceding release.

## Local validation

The full offline non-wallclock suite and refreshed teacher regressions passed on the latest `dev` base. Ruff lint/format, source-layout, file-size and function-size gates passed. The serial wallclock suite passed on rerun; the first run missed one bootstrap shutdown cutoff by about 95 ms. No timing code or tolerances were changed.

## Configuration migration and rollout

Update explicit `teacher_model = "glm-5.2"` to `"glm-5.3"`, and `"deepseek-v4-pro"` to `"deepseek-v4.1-flash-1017"`. Removed aliases fail validation with the supported choices; they do not silently change the selected model family. An omitted teacher now selects GLM 5.3. Coordinate control-plane and worker versions and let in-flight old-teacher runs finish before rollout.

Provider catalogs change independently of Flash releases. A future release gate should score a small request for every advertised teacher, including actual student template text, before spending on GPU training. A catalog entry alone does not establish scoring compatibility. Qwen3.8's direct-account price and Kimi's reproducible 500 should also be checked before expanding production use.
