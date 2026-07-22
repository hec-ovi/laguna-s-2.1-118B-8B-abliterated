"""Refusal-direction abliteration tooling for poolside Laguna-S-2.1.

Pure-python modules (arch, data, and the refusal-detector half of scoring) import
without torch so the contract tests run on a bare interpreter. Everything that
touches weights or activations (projection, weights, engine, probe, KL) imports
torch lazily or at module top and is only exercised inside the gfx1151 venv.
"""

__version__ = "0.0.1"
