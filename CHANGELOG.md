# Changelog

This file records the notable changes made to `RandomTilings`.

## Unreleased

Changes completed after the latest official release will be listed here.

## Version 1.0.0 — First official release
*Released on 19.07.2026* 

Version 1.0.0 is the first official release of `RandomTilings`. It introduces a new implementation of the core algorithms and replaces the previous function-based workflow with an object-oriented interface. The new implementation reduces the memory complexity from cubic to quadratic, resulting in significantly better memory efficiency and improved runtime performance.

### Added

* Added the `Aztec` class for generating random tilings of the Aztec diamond.
* Added the `Hexagon` class for generating random tilings of the hexagon.
* Added an object-oriented interface for constructing random-tiling models.
* Added the ability to generate multiple random tilings from the same model without reconstructing it.
* Added separate `shuffle()` and `plot()` methods for sampling and displaying tilings.
* Added access to the corresponding Matplotlib figure through the `fig` attribute.
* Added a `close()` method for closing a model and releasing its resources.
* Added additional options for modifying the appearance of generated plots.

### Changed

* Reimplemented the core shuffling algorithms.
* Reduced the memory complexity from cubic to quadratic in the size of the tiling.
* Significantly improved runtime performance.
* Improved support for generating larger tilings.
* Replaced the previous function-based interface with an object-oriented interface.
* Separated the construction of a model from the sampling and plotting of individual tilings.
* Updated the documentation and examples for the new interface.

### Compatibility

The interface of Version 1.0.0 differs substantially from that of the earlier implementation. Code written for Version 0 may therefore require changes.

The earlier implementation remains available in the [`Version-0`](https://github.com/MaxvanHorssen/random-tilings/tree/Version-0) branch.

