"""Validate persisted training identities before recovery and resource mutations."""


def parse_handle(data):
    if data.get("provider") == "spot":
        from flash.providers.nebius.handles import SpotJobHandle

        return SpotJobHandle.from_dict(data)
    from flash.providers.modal.handles import ModalJobHandle

    return ModalJobHandle.from_dict(data)
