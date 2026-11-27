"""Router helpers for the Lambda-only hosted GPU pool."""


async def unregister_active(engines, org_id, model_id, expected_generation):
    """Evict an adapter from live pools without booting an idle GPU."""
    for engine in engines:
        stats = await engine.health.get_current_stats.aio()
        if any((stats.num_total_runners, stats.num_running_inputs, stats.backlog)):
            await engine.unregister.remote.aio(org_id, model_id, expected_generation)


def bind_lambda_control():
    """Keep request routing independent of the serialized VM mutation queue."""
    import modal

    from flash.serving.lambda_cloud.config import APP_NAME

    return modal.Cls.from_name(APP_NAME, "LambdaFleet")()
