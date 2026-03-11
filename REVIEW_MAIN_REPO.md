# Review: Main Repo Doc Cleanup

**Branch:** `doc/cleanup-main`
**Scope:** Documentation trimming, redundancy removal, Docker cross-refs

## Changes

### EPOCH_SETUP.md
- **Removed** "Parameter Optimization" section (10 lines) — misplaced content that belongs in `costsci_tools/docs/epoch_1d.md`, not a setup guide
- **Fixed** `dt_multipler` typo (now removed with the section)
- **Added** Docker cross-ref note at end

### EULER_2D_SETUP.md
- **Removed** verbose directory tree (23 lines) — internal structure detail not needed for setup
- **Removed** "Performance Issues" section (7 lines) — generic advice, not setup-relevant
- **Added** Docker cross-ref note at end

### FEM_2D_SETUP.md
- **Removed** "Development Notes" section — code structure internals (15 lines)
- **Removed** "Python Path Requirements" — handled automatically by the runner (3 lines)
- **Removed** "Working with the Git Submodule" section — dev workflow, not user setup (13 lines)
- **Removed** "References" — empty placeholder (2 lines)
- **Added** Docker cross-ref note at end

### Net effect
- ~115 lines removed across 3 files
- All 3 setup guides now end with a consistent Docker cross-ref
- No content loss — removed material was either misplaced, dev-internal, or filler

## Not included in this PR (pre-existing unstaged changes)
The following files have pre-existing changes unrelated to this cleanup and are **not staged**:
- `.gitmodules`, `README.md`, `configs/custom_models.json`, `evaluation/README.md`, `scripts/README.md`, `.env.example`
