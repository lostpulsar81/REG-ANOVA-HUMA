import io
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats
from scipy.optimize import curve_fit
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests

st.set_page_config(layout="wide")
st.title("Interactive Regression Analysis")


def poly(x, *coeffs):
    return sum(c * x**i for i, c in enumerate(coeffs))


def format_equation(coeffs):
    parts = []
    for i, c in enumerate(coeffs):
        if i == 0:
            parts.append(f"{c:.4g}")
        elif i == 1:
            parts.append(f"{c:+.4g}·x")
        else:
            parts.append(f"{c:+.4g}·x^{i}")
    return "y = " + " ".join(parts)


def fit_model(x, y, yerr, degree, use_weights):
    sigma = yerr if use_weights else None
    absolute_sigma = use_weights
    p0 = np.ones(degree + 1, dtype=float)

    params, cov = curve_fit(
        lambda xx, *a: poly(xx, *a),
        x,
        y,
        p0=p0,
        sigma=sigma,
        absolute_sigma=absolute_sigma,
        maxfev=20000,
    )

    y_pred = poly(x, *params)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else np.nan

    n = len(x)
    p = degree
    denom_df = n - (p + 1)
    if denom_df > 0 and ss_res > 0:
        f_stat = ((ss_tot - ss_res) / p) / (ss_res / denom_df) if p > 0 else np.nan
        p_value = 1 - stats.f.cdf(f_stat, p, denom_df) if p > 0 else np.nan
    else:
        f_stat = np.nan
        p_value = np.nan

    param_errors = np.sqrt(np.diag(cov)) if cov is not None else np.full(len(params), np.nan)
    return params, param_errors, y_pred, r2, f_stat, p_value


def build_long_df(concentrations, data_df):
    long_rows = []
    for col_idx, conc in enumerate(concentrations):
        col_values = pd.to_numeric(data_df.iloc[:, col_idx], errors="coerce").dropna().to_numpy()
        for value in col_values:
            long_rows.append({"group": str(conc), "concentration": float(conc), "value": float(value)})
    return pd.DataFrame(long_rows)


def run_anova(long_df):
    grouped = [grp["value"].to_numpy() for _, grp in long_df.groupby("group")]
    if len(grouped) < 2:
        return np.nan, np.nan
    f_stat, p_value = stats.f_oneway(*grouped)
    return f_stat, p_value


def run_bonferroni(long_df):
    groups = sorted(long_df["group"].unique(), key=lambda x: float(x))
    rows = []
    pvals = []
    pairs = []
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            g1 = groups[i]
            g2 = groups[j]
            v1 = long_df.loc[long_df["group"] == g1, "value"].to_numpy()
            v2 = long_df.loc[long_df["group"] == g2, "value"].to_numpy()
            t_stat, p_raw = stats.ttest_ind(v1, v2, equal_var=False, nan_policy="omit")
            mean_diff = np.mean(v1) - np.mean(v2)
            rows.append([g1, g2, mean_diff, p_raw])
            pvals.append(p_raw)
            pairs.append((g1, g2))
    reject, p_adj, _, _ = multipletests(pvals, method="bonferroni") if pvals else ([], [], [], [])
    out = []
    for idx, (g1, g2) in enumerate(pairs):
        out.append({
            "group_1": g1,
            "group_2": g2,
            "mean_diff": rows[idx][2],
            "p_raw": rows[idx][3],
            "p_adj": p_adj[idx],
            "significant": bool(reject[idx]),
        })
    return pd.DataFrame(out)


def run_dunnett_like(long_df, control_group):
    groups = sorted(long_df["group"].unique(), key=lambda x: float(x))
    if control_group not in groups:
        return pd.DataFrame()
    control_vals = long_df.loc[long_df["group"] == control_group, "value"].to_numpy()
    rows = []
    pvals = []
    comparisons = []
    for grp in groups:
        if grp == control_group:
            continue
        vals = long_df.loc[long_df["group"] == grp, "value"].to_numpy()
        t_stat, p_raw = stats.ttest_ind(control_vals, vals, equal_var=False, nan_policy="omit")
        mean_diff = np.mean(vals) - np.mean(control_vals)
        rows.append([control_group, grp, mean_diff, p_raw])
        pvals.append(p_raw)
        comparisons.append((control_group, grp))
    reject, p_adj, _, _ = multipletests(pvals, method="bonferroni") if pvals else ([], [], [], [])
    out = []
    for idx, (g1, g2) in enumerate(comparisons):
        out.append({
            "control": g1,
            "treatment": g2,
            "mean_diff_treatment_minus_control": rows[idx][2],
            "p_raw": rows[idx][3],
            "p_adj": p_adj[idx],
            "significant": bool(reject[idx]),
        })
    return pd.DataFrame(out)


left_col, right_col = st.columns([1, 2])

with left_col:
    st.subheader("Data")
    uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx", "xls"])
    sheet_name = st.text_input("Sheet name", value="Sheet1")

    st.subheader("Analysis")
    analysis_mode = st.selectbox("Analysis mode", ["Regression", "ANOVA"])

    if analysis_mode == "Regression":
        st.subheader("Model")
        line_color = st.color_picker("Regression curve color", value="#1f77b4")
        model_type = st.selectbox("Regression type", ["Linear", "Polynomial"])
        degree = 1 if model_type == "Linear" else st.slider("Polynomial degree", 2, 10, 2)
        use_weights = st.checkbox("Use weighted regression", value=False)
        chart_type = st.selectbox("Chart type", ["Line", "Bar"])
    else:
        st.subheader("ANOVA settings")
        posthoc_method = st.selectbox("Post hoc test", ["None", "Tukey HSD", "Bonferroni", "Dunnett vs control"])
        control_group_default = "0"
        control_group = st.text_input("Control group for Dunnett-like comparison", value=control_group_default)
        chart_type = st.selectbox("Chart type", ["Bar", "Line"])

    st.subheader("Text")
    title = st.text_input("Plot title", f"{sheet_name} analysis" if sheet_name else "Analysis")
    x_label_override = st.text_input("X-axis label", "")
    y_label = st.text_input("Y-axis label", sheet_name if sheet_name else "Response variable")
    legend_label = st.text_input("Regression curve label", "Regression curve")
    data_label = st.text_input("Experimental data label", "Experimental data")

    st.subheader("Fonts")
    font_size_legend = st.slider("Legend font size", 6, 24, 10)
    font_size_tick = st.slider("Axis tick font size", 6, 24, 10)
    font_size_axis_titles = st.slider("Axis title font size", 6, 24, 12)
    font_size_title = st.slider("Plot title font size", 6, 30, 14)

    st.subheader("Saving")
    default_downloads = str(Path.home() / "Downloads")
    save_dir = st.text_input("Save folder", value=default_downloads)
    file_name = st.text_input("File name", value="analysis_plot")
    save_format = st.selectbox("Format", ["png", "jpg"], index=0)

with right_col:
    if uploaded_file and sheet_name:
        try:
            raw_df = pd.read_excel(uploaded_file, sheet_name=sheet_name, header=None)

            if raw_df.shape[0] < 2 or raw_df.shape[1] < 2:
                st.error(
                    "The selected sheet must contain at least 2 rows and 2 columns: concentrations in row 1 starting from A1, and replicate data below."
                )
                st.stop()

            concentration_cells = raw_df.iloc[0, :]
            try:
                concentrations = pd.to_numeric(concentration_cells, errors="raise").to_numpy(dtype=float)
            except Exception:
                st.error("All cells in the first row, starting from A1, must be numeric concentration values.")
                st.stop()

            data_df = raw_df.iloc[1:, :].apply(pd.to_numeric, errors="coerce")
            data_df = data_df.dropna(how="all")

            if data_df.empty:
                st.error("No replicate data found below the concentration row.")
                st.stop()

            mean_heights = data_df.mean().to_numpy().flatten()
            std_heights = data_df.std().to_numpy().flatten()

            if len(mean_heights) != len(concentrations):
                st.error(
                    f"Incompatible number of points: the sheet provides {len(mean_heights)} data columns, but {len(concentrations)} concentration values were found."
                )
                st.stop()

            if len(concentrations) < 2:
                st.error("At least two concentration columns are required.")
                st.stop()

            x_label = x_label_override.strip() if x_label_override.strip() else "Concentration"
            long_df = build_long_df(concentrations, data_df)

            fig, ax = plt.subplots(figsize=(10, 6))

            if analysis_mode == "Regression":
                if use_weights:
                    if np.any(std_heights <= 0) or np.any(~np.isfinite(std_heights)):
                        st.error("For weighted regression, all standard deviations must be finite and greater than zero.")
                        st.stop()

                params, param_errors, y_pred, r2, f_stat, p_value = fit_model(
                    concentrations,
                    mean_heights,
                    std_heights,
                    degree,
                    use_weights,
                )

                x_fit = np.linspace(float(np.min(concentrations)), float(np.max(concentrations)), 600)
                y_fit = poly(x_fit, *params)
                max_index = int(np.argmax(y_fit))
                max_concentration = float(x_fit[max_index])
                max_height = float(y_fit[max_index])
                equation = format_equation(params)

                max_data_index = int(np.argmax(mean_heights))
                max_data_concentration = float(concentrations[max_data_index])
                max_data_height = float(mean_heights[max_data_index])

                if chart_type == "Line":
                    ax.errorbar(
                        concentrations,
                        mean_heights,
                        yerr=std_heights,
                        fmt="o",
                        capsize=5,
                        color="black",
                        label=data_label,
                    )
                    curve_name = f"{legend_label} ({'weighted' if use_weights else 'unweighted'})"
                    curve_label = (
                        f"{curve_name}\n"
                        f"R² = {r2:.4f}    p = {p_value:.4g}\n"
                        f"{equation}"
                    )
                    ax.plot(x_fit, y_fit, color=line_color, label=curve_label, linewidth=2)
                    ax.axvline(x=max_concentration, linestyle="--", color="orange", label=f"Max reg: {max_concentration:.1f} {x_label}")
                    ax.scatter(max_concentration, max_height, color="orange", zorder=5)

                    x_span = float(np.max(concentrations) - np.min(concentrations))
                    y_span = float(np.max(y_fit) - np.min(y_fit))
                    if y_span == 0:
                        y_span = max(abs(max_height), 1.0)
                    ax.text(
                        max_concentration - 0.18 * x_span,
                        max_height + 0.06 * y_span,
                        f"Max reg: {max_concentration:.1f} {x_label}",
                        color="orange",
                        fontsize=max(font_size_tick, 8),
                        ha="left",
                        va="bottom",
                    )
                else:
                    bar_positions = np.arange(len(concentrations))
                    ax.bar(
                        bar_positions,
                        mean_heights,
                        yerr=std_heights,
                        capsize=5,
                        color=line_color,
                        label=data_label,
                    )
                    ax.set_xticks(bar_positions)
                    ax.set_xticklabels([f"{c:g}" for c in concentrations])
                    ax.scatter(
                        max_data_index,
                        max_data_height,
                        color="orange",
                        zorder=5,
                        label=f"Max data: {max_data_concentration:.1f} {x_label}",
                    )
                    y_span = float(np.max(mean_heights) - np.min(mean_heights))
                    if y_span == 0:
                        y_span = max(abs(max_data_height), 1.0)
                    ax.text(
                        max_data_index,
                        max_data_height + 0.06 * y_span,
                        f"Max data: {max_data_concentration:.1f} {x_label}",
                        color="orange",
                        fontsize=max(font_size_tick, 8),
                        ha="center",
                        va="bottom",
                    )
                    stats_text = (
                        f"{legend_label} ({'weighted' if use_weights else 'unweighted'})\n"
                        f"R² = {r2:.4f}    p = {p_value:.4g}\n"
                        f"{equation}"
                    )
                    ax.text(
                        0.02,
                        0.98,
                        stats_text,
                        transform=ax.transAxes,
                        fontsize=font_size_legend,
                        va="top",
                        ha="left",
                        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
                    )

                ax.set_xlabel(x_label, fontsize=font_size_axis_titles)
                ax.set_ylabel(y_label, fontsize=font_size_axis_titles)
                ax.set_title(title, fontsize=font_size_title)
                ax.tick_params(axis="both", labelsize=font_size_tick)
                ax.legend(loc="upper right", bbox_to_anchor=(1, 1), fontsize=font_size_legend, frameon=True)
                ax.grid(True)
                fig.tight_layout()
                st.pyplot(fig, clear_figure=False)

                st.subheader("Results")
                res_col1, res_col2 = st.columns(2)
                with res_col1:
                    st.write(f"**R²:** {r2:.6f}")
                    st.write(f"**p-value:** {p_value:.6g}")
                    st.write(f"**F statistic:** {f_stat:.6g}" if np.isfinite(f_stat) else "**F statistic:** not available")
                    st.write(f"**Max reg:** {max_concentration:.3f} {x_label}")
                    st.write(f"**Max data:** {max_data_concentration:.3f} {x_label}")
                with res_col2:
                    st.write(f"**Equation:** {equation}")
                    st.write("**Definitions:**")
                    st.write("- **Max reg** = maximum estimated from the fitted regression curve")
                    st.write("- **Max data** = maximum observed mean value in the experimental data")
                    for i, (par, err) in enumerate(zip(params, param_errors)):
                        st.write(f"**Parameter a{i}:** {par:.6g} ± {err:.3g}")

            else:
                anova_f, anova_p = run_anova(long_df)
                group_means = long_df.groupby("group", as_index=False)["value"].mean()
                group_stds = long_df.groupby("group", as_index=False)["value"].std().fillna(0)
                group_order = sorted(long_df["group"].unique(), key=lambda x: float(x))
                mean_map = dict(zip(group_means["group"], group_means["value"]))
                std_map = dict(zip(group_stds["group"], group_stds["value"]))
                ordered_means = np.array([mean_map[g] for g in group_order], dtype=float)
                ordered_stds = np.array([std_map[g] for g in group_order], dtype=float)
                ordered_concs = np.array([float(g) for g in group_order], dtype=float)

                if chart_type == "Bar":
                    bar_positions = np.arange(len(ordered_concs))
                    ax.bar(bar_positions, ordered_means, yerr=ordered_stds, capsize=5, color="#4C78A8", label=data_label)
                    ax.set_xticks(bar_positions)
                    ax.set_xticklabels([f"{c:g}" for c in ordered_concs])
                    max_idx = int(np.argmax(ordered_means))
                    ax.scatter(max_idx, ordered_means[max_idx], color="orange", zorder=5, label=f"Max data: {ordered_concs[max_idx]:.1f} {x_label}")
                    y_span = float(np.max(ordered_means) - np.min(ordered_means))
                    if y_span == 0:
                        y_span = max(abs(ordered_means[max_idx]), 1.0)
                    ax.text(max_idx, ordered_means[max_idx] + 0.06 * y_span, f"Max data: {ordered_concs[max_idx]:.1f} {x_label}", color="orange", fontsize=max(font_size_tick, 8), ha="center", va="bottom")
                else:
                    ax.errorbar(ordered_concs, ordered_means, yerr=ordered_stds, fmt="o-", capsize=5, color="#4C78A8", label=data_label)
                    max_idx = int(np.argmax(ordered_means))
                    ax.axvline(ordered_concs[max_idx], linestyle="--", color="orange", label=f"Max data: {ordered_concs[max_idx]:.1f} {x_label}")
                    ax.scatter(ordered_concs[max_idx], ordered_means[max_idx], color="orange", zorder=5)

                ax.set_xlabel(x_label, fontsize=font_size_axis_titles)
                ax.set_ylabel(y_label, fontsize=font_size_axis_titles)
                ax.set_title(title, fontsize=font_size_title)
                ax.tick_params(axis="both", labelsize=font_size_tick)
                ax.legend(loc="upper right", bbox_to_anchor=(1, 1), fontsize=font_size_legend, frameon=True)
                ax.grid(True)
                fig.tight_layout()
                st.pyplot(fig, clear_figure=False)

                st.subheader("Results")
                st.write(f"**ANOVA F statistic:** {anova_f:.6g}" if np.isfinite(anova_f) else "**ANOVA F statistic:** not available")
                st.write(f"**ANOVA p-value:** {anova_p:.6g}" if np.isfinite(anova_p) else "**ANOVA p-value:** not available")

                if posthoc_method == "Tukey HSD":
                    tukey = pairwise_tukeyhsd(endog=long_df["value"], groups=long_df["group"], alpha=0.05)
                    tukey_df = pd.DataFrame(tukey.summary().data[1:], columns=tukey.summary().data[0])
                    st.write("**Post hoc: Tukey HSD**")
                    st.dataframe(tukey_df, use_container_width=True)
                elif posthoc_method == "Bonferroni":
                    bonf_df = run_bonferroni(long_df)
                    st.write("**Post hoc: Bonferroni-corrected pairwise comparisons**")
                    st.dataframe(bonf_df, use_container_width=True)
                elif posthoc_method == "Dunnett vs control":
                    dunnett_df = run_dunnett_like(long_df, control_group)
                    st.write("**Post hoc: Dunnett-like comparisons vs control (Bonferroni-corrected pairwise control comparisons)**")
                    st.dataframe(dunnett_df, use_container_width=True)

            img_buffer = io.BytesIO()
            fig.savefig(img_buffer, format=save_format, dpi=300, bbox_inches="tight")
            img_buffer.seek(0)

            st.download_button(
                label=f"Download plot ({save_format.upper()})",
                data=img_buffer,
                file_name=f"{file_name}.{save_format}",
                mime=f"image/{'jpeg' if save_format == 'jpg' else 'png'}",
            )

            if st.button("Save plot to selected folder"):
                try:
                    os.makedirs(save_dir, exist_ok=True)
                    output_path = os.path.join(save_dir, f"{file_name}.{save_format}")
                    fig.savefig(output_path, format=save_format, dpi=300, bbox_inches="tight")
                    st.success(f"Plot saved to: {output_path}")
                except Exception as save_error:
                    st.error(f"Error while saving plot: {save_error}")

        except Exception as e:
            st.error(f"Error while loading or analyzing the file: {e}")
    else:
        st.info("Upload an Excel file and enter the sheet name to display the analysis.")
