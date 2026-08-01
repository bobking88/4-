# Task 3 self-review

## Scope

Implemented the Chinese theory-aware network architecture figure for `HierarchicalRoleAwareEfficientNet` in the requested worktree. The changes are limited to the Task 3 implementation, test, generated PNG, and this review.

## Requirement review

- Added `plot_theory_aware_hierarchical_architecture(output_path: Path) -> None` to `scripts/generate_paper_figures.py`.
- Added the figure-generation test to `tests/test_generate_paper_figures.py`.
- Added the generated artifact at `outputs/paper_figures_v1/fig10_theory_aware_hierarchical_architecture_cn.png`.
- Added the function to the existing `generate_paper_figures.py` entry point so regeneration produces the new artifact.
- The graph shows `矿物图像 -> EfficientNet-B0共享主干 -> 共享特征 h`.
- The graph shows exactly four independent sibling heads: `角色头 p_r`, `种类头 p_s`, `目标代理二分类头 p_b`, and `投影头 e`.
- There are no arrows between sibling heads.
- The species branch carries the only role-aggregation mapping label, `A p_s = p_tilde_r`, into the consistency-loss box; the direct role output also enters that same box.
- The hard-negative annotation is confined to `目标矿物 - 含钛干扰` and `目标矿物 - 金属光泽困难干扰`.
- The footer contains `L = L_role + alpha L_species + beta L_cons + gamma L_binary + eta L_hard`.
- The renderer selects an installed Chinese font so the Chinese labels render in the PNG without missing-glyph warnings.

## Test-first evidence

The new test was run before implementation using the repository's actual test class and failed with an `ImportError` because `plot_theory_aware_hierarchical_architecture` was absent. The literal command from the brief names `PaperFigureTests`, but this repository defines `FigureSummaryTests`; the literal command therefore fails during unittest discovery before reaching the test.

## Verification

- Focused test: `FigureSummaryTests.test_theory_aware_architecture_figure_is_created` passed.
- Full figure test module: `tests.test_generate_paper_figures` passed, 10 tests, 0 failures.
- Syntax verification passed with `py_compile` for the modified Python files.
- Figure regeneration passed with `scripts/generate_paper_figures.py`.
- Visual inspection of `fig10_theory_aware_hierarchical_architecture_cn.png` confirmed readable Chinese labels, four sibling branches, the shared backbone and feature node, the species-to-role mapping, the shared consistency box, and the restricted hard-negative note.

## Concerns

1. The brief's focused-test command uses the stale class name `PaperFigureTests`; it should be updated to `FigureSummaryTests` if the brief is maintained.
2. The requested PNG-only interface is implemented for the new figure. Existing figures continue to use their established PNG/PDF/SVG export behavior.
3. The diagram is a faithful conceptual architecture figure; it does not attempt to render tensor dimensions or a literal matrix graphic for `A`, keeping the mapping explicit in the annotated edge as requested.

## Fix round 1 review

Reviewer feedback was interpreted together with the user clarification. The direct role posterior must remain an input to the consistency loss, alongside the species-derived aggregated role posterior. The figure now makes that relationship explicit:

- The role-head arrow into the consistency box is labeled `p_r`.
- The species-head arrow into the same box is labeled `A p_s = p_tilde_r`.
- The consistency box states `L_cons = KL(p_r || p_tilde_r)`.
- The direct role arrow was retained; no sibling-head arrow was added.
- A regression assertion now checks all three labels/formula strings in the renderer source.

The consistency box was enlarged and its arrow endpoints adjusted so both posterior inputs remain visually distinct and readable in the regenerated PNG. The existing four-head topology and restricted hard-negative annotation were left unchanged.

Fix-round verification completed with the focused figure test and the full `tests.test_generate_paper_figures` module. The regenerated PNG was visually inspected after the label update. The only remaining repository concern is the pre-existing modified plan file, which was not included in this fix commit.
