"""parent-side renderer for the isolated GRPO reward bridge module."""

from __future__ import annotations


def render_reward_module(url_env: str = "FLASH_VERL_REWARD_URL") -> str:
    """render the stdlib-only reward module copied into the verl child."""
    return (
        '"""flash reward bridge shim (generated). posts each completion to the flash worker."""\n'
        "import os\n"
        "from flash_grpo_multiturn import post_json\n"
        "\n"
        f"_URL = os.environ.get({url_env!r}, '')\n"
        "\n"
        "\n"
        "def compute_score(data_source, solution_str, ground_truth, extra_info=None):\n"
        "    idx = (extra_info or {}).get('index')\n"
        "    if idx is None:\n"
        "        raise RuntimeError('flash reward bridge received no example index')\n"
        "    if not _URL:\n"
        "        raise RuntimeError('flash reward bridge url is not configured')\n"
        "    if isinstance(idx, bool) or getattr(getattr(idx, 'dtype', None), 'kind', None) == 'b':\n"
        "        raise RuntimeError('flash reward bridge received an invalid example index: %r' % idx)\n"
        "    try:\n"
        "        exact_idx = int(idx)\n"
        "    except (TypeError, ValueError, OverflowError) as exc:\n"
        "        raise RuntimeError('flash reward bridge received an invalid example index: %r' % idx) from exc\n"
        "    if exact_idx != idx:\n"
        "        raise RuntimeError('flash reward bridge received an invalid example index: %r' % idx)\n"
        "    idx = exact_idx\n"
        "    identity = (extra_info or {}).get('flash_rollout_identity')\n"
        "    if not isinstance(identity, dict):\n"
        "        raise RuntimeError('flash reward bridge received no rollout identity')\n"
        "    payload = post_json(\n"
        "        _URL,\n"
        "        '/score',\n"
        "        {'index': idx, 'solution_str': solution_str or '', 'identity': dict(identity)},\n"
        "        error_style='reward',\n"
        "    )\n"
        "    try:\n"
        "        return float(payload['score'])\n"
        "    except Exception as exc:\n"
        "        raise RuntimeError('flash reward bridge returned an invalid response: %s' % exc) from exc\n"
    )
