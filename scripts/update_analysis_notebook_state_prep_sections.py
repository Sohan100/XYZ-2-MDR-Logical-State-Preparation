"""

update_analysis_notebook_state_prep_sections.py
----------------------------------------------------------------------------
Command-line helpers for updating analysis notebook state prep sections
content.
"""

from __future__ import annotations

import json
from pathlib import Path

NOTEBOOK_PATH = Path("analysis_plots.ipynb")

XYZ2_SECTION_HEADING = "## XYZ^2 Logical-X Error vs Physical Error Rate"
SURFACE_SECTION_HEADING = (
    "## Surface Code Logical-X Error vs Physical Error Rate"
)
COMPARISON_SECTION_HEADING = "## XYZ^2 vs Surface d=3 and d=11 Comparison"


def code_cell(source: str) -> dict:
    """
    Code cell.

    Args:
    source: Source.

    Returns:
    Computed value returned by this helper.
    """
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def markdown_cell(source: str) -> dict:
    """
    Markdown cell.

    Args:
    source: Source.

    Returns:
    Computed value returned by this helper.
    """
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def ensure_setup_block(nb: dict) -> None:
    """
    Ensure setup block is configured before continuing.

    Args:
    nb: Nb.
    """
    setup_cell = nb["cells"][2]
    setup_source = "".join(setup_cell["source"])
    setup_source = setup_source.replace(
        """from mdr.analysis_reporting import (
    NotebookCodeComparisonAnalysis,
    NotebookFinalRoundAnalysis,
    NotebookThresholdAnalysis,
)""",
        """from mdr.analysis_reporting import (
    NotebookFinalRoundAnalysis,
    NotebookThresholdAnalysis,
)""",
    )

    additions = []
    if "state_prep_threshold_analysis" not in setup_source:
        additions.append(
            """
state_prep_threshold_dir = threshold_dir / 'state_prep_error'
state_prep_threshold_dir.mkdir(parents=True, exist_ok=True)
state_prep_threshold_analysis = NotebookThresholdAnalysis(
    results_dir=results_dir,
    plots_dir=state_prep_threshold_dir,
    distances=distances,
    noise_models=noise_models,
    code_family=code_family,
)

surface_results_dir = (
    repo_root / 'data' / 'simulation_results' / surface_code_family
)
surface_threshold_dir = surface_plots_dir / 'thresholds' / 'state_prep_error'
surface_threshold_dir.mkdir(parents=True, exist_ok=True)
surface_threshold_analysis = NotebookThresholdAnalysis(
    results_dir=surface_results_dir,
    plots_dir=surface_threshold_dir,
    distances=distances,
    noise_models=noise_models,
    code_family=surface_code_family,
)
"""
        )

    setup_source = setup_source.replace(
        """
comparison_plots_dir = (
    repo_root / 'data' / 'plots' / 'analysis_notebook' / 'comparisons'
)
comparison_plots_dir.mkdir(parents=True, exist_ok=True)
code_comparison_analysis = NotebookCodeComparisonAnalysis(
    family_results_dirs={
        'xyz2': results_dir,
        'surface': surface_results_dir,
    },
    plots_dir=comparison_plots_dir,
    distances=[3, 11],
    noise_models=noise_models,
    family_labels={
        'xyz2': 'XYZ^2',
        'surface': 'Surface Code',
    },
)
""",
        "",
    )

    if additions:
        setup_source = setup_source + "".join(additions)
    setup_cell["source"] = setup_source.splitlines(keepends=True)


def remove_existing_state_prep_sections(nb: dict) -> None:
    """
    Remove existing state prep sections from the active data structure.

    Args:
    nb: Nb.
    """
    filtered_cells = []
    state_prep_headings = {
        XYZ2_SECTION_HEADING,
        SURFACE_SECTION_HEADING,
        COMPARISON_SECTION_HEADING,
    }
    pending_code_removals = 0
    for cell in nb["cells"]:
        if pending_code_removals and cell.get("cell_type") == "code":
            pending_code_removals -= 1
            continue
        src = "".join(cell.get("source", []))
        if any(src.startswith(heading) for heading in state_prep_headings):
            pending_code_removals = 1
            continue
        filtered_cells.append(cell)
    nb["cells"] = filtered_cells


def append_sections(nb: dict) -> None:
    """
    Append sections to the accumulating output object.

    Args:
    nb: Nb.
    """
    while nb["cells"]:
        last_cell = nb["cells"][-1]
        if last_cell.get("cell_type") != "code":
            break
        if "".join(last_cell.get("source", [])).strip():
            break
        nb["cells"].pop()

    xyz2_markdown = markdown_cell(
        """## XYZ^2 Logical-X Error vs Physical Error Rate

Plot the saved XYZ^2 logical-state-preparation error `1 - |<X_L>|` against the
physical error parameter `p` for `p_spam = 1.339e-3`, covering all three noise
models and all available distances.
"""
    )
    xyz2_code = code_cell(
        """xyz2_state_prep_sweeps, xyz2_state_prep_files = (
    state_prep_threshold_analysis.load_sweeps_for_p_spam(
    with_spam_p,
    recovery_mode='each_round',
    correction_mode='physical',
    )
)
display(xyz2_state_prep_files)
if xyz2_state_prep_files.empty:
    print('No saved XYZ^2 with-SPAM physical-recovery results were found.')
else:
    xyz2_state_prep_paths = state_prep_threshold_analysis.plot_threshold_suite(
        sweeps_by_model=xyz2_state_prep_sweeps,
        output_label='with_spam_logical_x_error',
        metric='state_prep_error',
    )
    for path in xyz2_state_prep_paths:
        print(f'Saved {path}')
"""
    )

    surface_markdown = markdown_cell(
        """## Surface Code Logical-X Error vs Physical Error Rate

Plot the saved surface-code logical-state-preparation error `1 - |<X_L>|`
against the physical error parameter `p` for `p_spam = 1.339e-3`, covering all
three noise models and all available distances.
"""
    )
    surface_code = code_cell(
        """surface_state_prep_sweeps, surface_state_prep_files = (
    surface_threshold_analysis.load_sweeps_for_p_spam(
    with_spam_p,
    recovery_mode='each_round',
    correction_mode='physical',
    )
)
display(surface_state_prep_files)
if surface_state_prep_files.empty:
    print(
        'No saved surface-code with-SPAM physical-recovery results '
        'were found.'
    )
else:
    surface_state_prep_paths = surface_threshold_analysis.plot_threshold_suite(
        sweeps_by_model=surface_state_prep_sweeps,
        output_label='with_spam_logical_x_error',
        metric='state_prep_error',
    )
    for path in surface_state_prep_paths:
        print(f'Saved {path}')
"""
    )

    comparison_markdown = markdown_cell(
        """## XYZ^2 vs Surface d=3 and d=11 Comparison

Compare `1 - |<X_L>|` against `p` on the same axes for `XYZ^2` and the surface
code at distances `d=3` and `d=11`. The section saves one figure for each noise
model.
"""
    )
    comparison_code = code_cell(
        """import pandas as pd
from matplotlib import pyplot as plt

from mdr import MdrNoiseSweep

comparison_plots_dir = (
    repo_root / 'data' / 'plots' / 'analysis_notebook' / 'comparisons'
)
comparison_plots_dir.mkdir(parents=True, exist_ok=True)

comparison_records = []
comparison_sweeps = {noise_model: [] for noise_model in noise_models}
comparison_distances = [3, 11]
family_configs = [
    ('xyz2', 'XYZ^2', state_prep_threshold_analysis),
    ('surface', 'Surface Code', surface_threshold_analysis),
]

for noise_model in noise_models:
    for distance in comparison_distances:
        for family_key, family_label, family_analysis in family_configs:
            try:
                csv_path = family_analysis.resolve_result_csv(
                    noise_model=noise_model,
                    distance=distance,
                    p_spam=with_spam_p,
                    recovery_mode='each_round',
                    correction_mode='physical',
                )
            except FileNotFoundError:
                continue

            display_label = f'{family_label} (d={distance})'
            comparison_sweeps[noise_model].append(
                {
                    'display_label': display_label,
                    'code_family': family_key,
                    'distance': distance,
                    'sweep': MdrNoiseSweep(load_data_filename=csv_path),
                }
            )
            comparison_records.append(
                {
                    'noise_model': noise_model,
                    'display_label': display_label,
                    'code_family': family_key,
                    'distance': distance,
                    'csv_path': str(csv_path),
                }
            )

comparison_files = pd.DataFrame(comparison_records)
if not comparison_files.empty:
    family_order = {'xyz2': 0, 'surface': 1}
    comparison_files['family_order'] = comparison_files[
        'code_family'
    ].map(family_order)
    comparison_files = comparison_files.sort_values(
        ['noise_model', 'distance', 'family_order']
    ).drop(columns=['family_order']).reset_index(drop=True)
display(comparison_files)

if comparison_files.empty:
    print('No saved cross-code comparison results were found.')
else:
    curve_colors = {
        ('xyz2', 3): '#1f77b4',
        ('surface', 3): '#d62728',
        ('xyz2', 11): '#2ca02c',
        ('surface', 11): '#9467bd',
    }
    distance_markers = {3: 'o', 11: 's'}
    comparison_paths = []

    for noise_model, noise_label in noise_models.items():
        entries = comparison_sweeps.get(noise_model, [])
        if not entries:
            continue

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.grid(True, alpha=0.3)

        for entry in entries:
            p_vals, y_vals, y_errs = (
                entry['sweep']._metric_series_for_operator(
                round_idx=1,
                operator='Logical X',
                metric='state_prep_error',
                )
            )
            ax.errorbar(
                p_vals,
                y_vals,
                yerr=y_errs,
                fmt=f"-{distance_markers[entry['distance']]}",
                color=curve_colors[(entry['code_family'], entry['distance'])],
                capsize=4,
                label=entry['display_label'],
            )

        ax.set_xscale('log')
        ax.set_title(f'{noise_label} Noise', fontsize=16)
        ax.set_xlabel('p', fontsize=14)
        ax.set_ylabel('Error rate (1 - |<X_L>|)', fontsize=14)
        ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
        fig.tight_layout(rect=[0, 0, 0.82, 1.0])

        save_path = (
            comparison_plots_dir
            / (
                f'comparison_xyz2_surface_{noise_model}_with_spam'
                f'_d3_d11_logical_x_error.pdf'
            )
        )
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        plt.close(fig)
        comparison_paths.append(save_path)

    for path in comparison_paths:
        print(f'Saved {path}')
"""
    )

    nb["cells"].extend(
        [
            xyz2_markdown,
            xyz2_code,
            surface_markdown,
            surface_code,
            comparison_markdown,
            comparison_code,
        ]
    )


def main() -> None:
    """
    Run the script entry point and coordinate the top-level workflow.
    """
    nb = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    ensure_setup_block(nb)
    remove_existing_state_prep_sections(nb)
    append_sections(nb)
    NOTEBOOK_PATH.write_text(
        json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
