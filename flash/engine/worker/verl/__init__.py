"""Flash-side Verl capability probing, checkpoint handling, and child observation.

Training children import Flash from its verified package root alongside the pinned Verl
installation. Never add flash/engine/worker to sys.path: that would shadow the external
verl package with this integration package. Scratch directories contain launch data only.
"""
