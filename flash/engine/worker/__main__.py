from flash.engine.worker import main

if __name__ == "__main__":
    try:
        main()
    except Exception:
        raise SystemExit(1) from None
