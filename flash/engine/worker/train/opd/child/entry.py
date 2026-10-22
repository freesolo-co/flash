"""packaged OPD entrypoint with explicit runtime construction."""

from flash.engine.worker.train.core.child.launch import run


def main():
    run("opd")


if __name__ == "__main__":
    main()
