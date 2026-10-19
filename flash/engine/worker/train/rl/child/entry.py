"""Packaged GRPO process entrypoint."""

from flash.engine.worker.train.core.child.launch import run


def main():
    run("grpo")


if __name__ == "__main__":
    main()
