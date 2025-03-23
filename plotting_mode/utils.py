import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import statsmodels.api as sm
from plotly.subplots import make_subplots
from scipy.stats import linregress, norm

# --------------------------------------------------------------------
# 1) Example "mpl_dark" template for dark mode
# --------------------------------------------------------------------
mpl_dark_template = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor="black",
        plot_bgcolor="black",
        font=dict(color="lightgray"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="lightgray")),
        xaxis=dict(
            linecolor="lightgray",
            gridcolor="lightgray",
            zerolinecolor="lightgray",
            tickfont=dict(color="lightgray"),
            title_font=dict(color="lightgray"),
        ),
        yaxis=dict(
            linecolor="lightgray",
            gridcolor="lightgray",
            zerolinecolor="lightgray",
            tickfont=dict(color="lightgray"),
            title_font=dict(color="lightgray"),
        ),
    )
)
pio.templates["mpl_dark"] = mpl_dark_template

# --------------------------------------------------------------------
# 2) Helper functions for color picking
# --------------------------------------------------------------------


def get_env_color(env_idx, is_dark_mode=False):
    """
    Pick a bright color from a palette for environment lines,
    ensuring good visibility in both dark and light mode.
    """
    # A palette of 10 bright colors (similar to Plotly default)
    bright_colors = [
        "#1f77b4",  # Blue
        "#ff7f0e",  # Orange
        "#2ca02c",  # Green
        "#d62728",  # Red
        "#9467bd",  # Purple
        "#8c564b",  # Brown
        "#e377c2",  # Pink
        "#7f7f7f",  # Gray
        "#bcbd22",  # Olive
        "#17becf",  # Teal
    ]
    return bright_colors[env_idx % len(bright_colors)]


def get_special_line_color(line_name, is_dark_mode=False):
    """
    Assign line colors for special lines (y=x, overall fit, optimal line, etc.)
    depending on dark or light mode.
    """
    if line_name == "y=x":
        # White stands out in dark mode, red is more visible in light mode
        return "white" if is_dark_mode else "red"
    elif line_name == "optimal":
        # Lime is bright on dark, green is more standard on light
        return "lime" if is_dark_mode else "green"
    elif line_name == "overall_fit":
        # Cyan on dark, Blue on light
        return "cyan" if is_dark_mode else "blue"
    else:
        # Fallback color
        return "gray"


# --------------------------------------------------------------------
# 3) Utility functions (rescale, validate, display_env_data)
# --------------------------------------------------------------------
def rescale(data, scaling=None):
    """
    Rescale the data according to the specified scaling method.
    """
    if scaling == "Probit":
        return norm.ppf(data)
    elif scaling == "Linear":
        return data
    raise NotImplementedError(
        f"Scaling method '{scaling}' is not implemented."
    )


def validate_stability(values):
    """
    Validate the stability of the values. Returns the values if they are stable, otherwise clips them.
    """
    if np.any(values <= 0) or np.any(values >= 1):
        return np.clip(values, 1e-10, 1 - 1e-10)
    return values


def display_env_data(df, env, scaling="Probit"):
    """
    Create a DataFrame to display slope, R2, p-value, and standard error for each environment.
    """
    left_out_env = df["test_env"].values[0]
    x_vals = validate_stability(df["x"].values.reshape(-1, 1))
    y_vals = validate_stability(df["y"].values.reshape(-1, 1))

    scaled_xs = rescale(x_vals, scaling)
    scaled_ys = rescale(y_vals, scaling)

    unique_envs = np.unique(env)
    env_data = []
    for unique_env in unique_envs:
        env_mask = env == unique_env
        slope, intercept, pearson_corr, p_value, std_err = linregress(
            scaled_xs[env_mask].flatten(), scaled_ys[env_mask].flatten()
        )
        env_data.append(
            {
                "OOD Environment": left_out_env,
                "ID Environment": unique_env,
                "Slope": f"{slope:.2f}",
                "Intercept": f"{intercept:.2f}",
                "Pearson Correlation R": f"{pearson_corr:.2f}",
                "P-value": f"{p_value:.2f}",
                "Standard Error": f"{std_err:.2f}",
                "|R| < 0.5": "Yes" if abs(pearson_corr) < 0.5 else "No",
            }
        )

    env_df = pd.DataFrame(env_data)
    return env_df


# --------------------------------------------------------------------
# 4) Main scatter plot function
# --------------------------------------------------------------------
def scatter_plot(
    df,
    env,
    scaling="Probit",
    do_domain=True,
    legend_change=False,
    show_linear_fits=True,
    is_dark_mode=False,
):
    """
    Scatter plot X vs Y, optionally scaling the data.
    No color bar; a single color or environment-based colors.
    """
    x_vals = validate_stability(df["x"].values.reshape(-1, 1))
    y_vals = validate_stability(df["y"].values.reshape(-1, 1))
    scaled_xs = rescale(x_vals, scaling)
    scaled_ys = rescale(y_vals, scaling)

    def label_point(i):
        """
        Generate hover label for each point.
        """
        x = x_vals[i, 0]
        y = y_vals[i, 0]
        label = f"Environment {env[i]}<br>"
        label += f"Algorithm: {df['algorithm'].values[i]}<br>"
        label += f"Model Arch: {df['model_arch'].values[i]}<br>"
        label += f"Transfer: {df['transfer'].values[i]}<br>"
        label += f"X: {x:.3f} <br>"
        label += f"Y: {y:.3f} <br>"
        return label

    # Scatter points: single color for all points
    # (If you want environment-based colors for the points themselves,
    #  you can do so by building a color array.)
    traces = [
        go.Scatter(
            x=scaled_xs.flatten(),
            y=scaled_ys.flatten(),
            mode="markers",
            hoverinfo="text",
            text=[label_point(i) for i in range(len(scaled_xs))],
            showlegend=False,
            marker=dict(
                size=8,
                color=(
                    "steelblue" if not is_dark_mode else "#17becf"
                ),  # e.g. teal in dark mode
                showscale=False,
            ),
        )
    ]

    # Per-environment or per-architecture lines
    if not legend_change:
        if show_linear_fits and do_domain:
            unique_envs = np.unique(env)
            for i, unique_env in enumerate(unique_envs):
                env_mask = env == unique_env
                slope, intercept, r_val, _, _ = linregress(
                    scaled_xs[env_mask].flatten(),
                    scaled_ys[env_mask].flatten(),
                )
                sample_pts = np.linspace(scaled_xs.min(), scaled_xs.max(), 50)
                line_y = slope * sample_pts + intercept
                env_color = get_env_color(i, is_dark_mode)
                traces.append(
                    go.Scatter(
                        mode="lines",
                        x=sample_pts,
                        y=line_y,
                        name=f"Env {unique_env}, slope={slope:.2f}, R={r_val:.2f}",
                        line=dict(color=env_color),
                        showlegend=True,
                    )
                )
    else:
        # Legend by model architecture
        unique_model_archs = df["model_arch"].unique()
        for i, arch in enumerate(unique_model_archs):
            arch_mask = df["model_arch"] == arch
            slope, intercept, r_val, _, _ = linregress(
                scaled_xs[arch_mask].flatten(), scaled_ys[arch_mask].flatten()
            )
            sample_pts = np.linspace(scaled_xs.min(), scaled_xs.max(), 50)
            line_y = slope * sample_pts + intercept
            arch_color = get_env_color(i, is_dark_mode)
            traces.append(
                go.Scatter(
                    mode="lines",
                    x=sample_pts,
                    y=line_y,
                    name=f"Arch {arch}, slope={slope:.2f}, R={r_val:.2f}",
                    line=dict(color=arch_color),
                    showlegend=True,
                )
            )

    # Overall fit
    slope, intercept, r_val, _, _ = linregress(
        scaled_xs.flatten(), scaled_ys.flatten()
    )
    r_squared = r_val**2
    sample_pts = np.linspace(scaled_xs.min(), scaled_xs.max(), 50)
    line_y = slope * sample_pts + intercept
    overall_fit_color = get_special_line_color("overall_fit", is_dark_mode)
    traces.append(
        go.Scatter(
            mode="lines",
            x=sample_pts,
            y=line_y,
            name=f"Linear fit (slope={slope:.2f}, R²={r_squared:.2f})",
            line=dict(color=overall_fit_color, width=2),
        )
    )

    # Prepare the environment DataFrame
    env_df = display_env_data(df, env, scaling)
    return traces, env_df


# --------------------------------------------------------------------
# 5) Main plot function
# --------------------------------------------------------------------
def plot(
    df,
    scaling="Probit",
    show_linear_fits=True,
    metric="Accuracy",
    all_shown=False,
    legend_change=False,
    is_dark_mode=False,
):
    """
    Generate an interactive scatter plot with:
    - optional dark mode,
    - border around the plot,
    - legend on the right outside the plot area,
    - special lines with colors adapted to dark/light mode.
    """
    chosen_template = "mpl_dark" if is_dark_mode else None

    fig = make_subplots(rows=1, cols=1)

    # Basic layout
    fig.update_layout(
        template=chosen_template,
        title=dict(
            text=f"ID vs OOD Performance ({scaling} Scale)",
            x=0.5,
            xanchor="center",
        ),
        width=800,
        height=600,
        # Move legend to the right outside plot
        legend=dict(
            x=1.02,
            y=1,
            xanchor="left",
            yanchor="auto",
            bgcolor="rgba(0,0,0,0)",
        ),
        # Give extra right margin so legend isn't cut off
        margin=dict(r=150),
    )

    # Add scatter & environment fits
    plot_traces, env_df = scatter_plot(
        df,
        env=df["train_env"].values,
        scaling=scaling,
        legend_change=legend_change,
        show_linear_fits=show_linear_fits,
        is_dark_mode=is_dark_mode,
    )

    # Add traces
    for trace in plot_traces:
        fig.add_trace(trace, row=1, col=1)

    # 1) y=x line
    metric_min, metric_max = 0.31, 0.90
    x_vals = np.linspace(metric_min, metric_max, 50)
    xy_line_color = get_special_line_color("y=x", is_dark_mode)
    fig.add_trace(
        go.Scatter(
            mode="lines",
            x=rescale(x_vals, scaling),
            y=rescale(x_vals, scaling),
            name="y=x",
            line=dict(color=xy_line_color, dash="dash"),
        ),
        row=1,
        col=1,
    )

    # Axis ticks
    tickmarks = np.linspace(metric_min, metric_max, 5)
    tick_dict = dict(
        tickmode="array",
        tickvals=rescale(tickmarks, scaling),
        ticktext=[f"{mark:.2f}" for mark in tickmarks],
        showgrid=True,
        gridcolor="lightgray",
    )
    fig.update_xaxes(title="ID Accuracy", **tick_dict)
    fig.update_yaxes(title="OOD Accuracy", **tick_dict)

    # Add plot border
    border_color = "lightgray" if is_dark_mode else "black"
    fig.update_xaxes(
        showline=True, linewidth=2, linecolor=border_color, mirror=True
    )
    fig.update_yaxes(
        showline=True, linewidth=2, linecolor=border_color, mirror=True
    )

    return fig, env_df
