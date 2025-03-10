# utils.py

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import statsmodels.api as sm
from plotly.subplots import make_subplots
from scipy.stats import linregress, norm

# Color map for different environments or model architectures
COLOR_MAP = {
    0: "blue",
    1: "red",
    2: "green",
    3: "orange",
    4: "purple",
    5: "brown",
    6: "pink",
    7: "gray",
    8: "olive",
    9: "cyan",
    10: "magenta",
    11: "yellow",
    12: "black",
    13: "darkblue",
    14: "darkred",
    15: "darkgreen",
    16: "darkorange",
    17: "darkpurple",
    18: "darkbrown",
    19: "darkpink",
    20: "darkgray",
    21: "darkolive",
    22: "darkcyan",
    23: "darkmagenta",
    24: "darkyellow",
}


def rescale(data, scaling=None):
    """
    Rescale the data according to the specified scaling method.
    """
    if scaling == "Probit":
        return norm.ppf(data)
    elif scaling == "Logit":
        return np.log(data / (1 - data))
    elif scaling == "Linear":
        return data
    elif scaling == "Square Root":
        return np.sqrt(data)
    raise NotImplementedError(
        f"Scaling method '{scaling}' is not implemented."
    )


def linear_fit(x, y):
    """
    Perform linear regression and return the bias and slope.
    """
    x, y = np.array(x), np.array(y)

    # Remove invalid entries
    valid_idx = ~np.isnan(x) & ~np.isinf(x) & ~np.isnan(y) & ~np.isinf(y)
    x, y = x[valid_idx], y[valid_idx]

    model = sm.OLS(y, sm.add_constant(x))
    result = model.fit()
    return result.params, result.rsquared


def parse_to_html(data):
    """
    Parse a string representation of a dictionary to HTML format.
    """
    data = data.strip("{} ")
    items = data.split(", ")

    html_output = ""
    for item in items:
        key, value = item.split(": ", 1)
        key = key.strip().strip("'")
        value = value.strip().strip("'")
        html_output += f"{key}: {value}<br>"

    return html_output


def validate_stability(values):
    """
    Validate the stability of the values. Returns the values if they are stable, otherwise clips them.
    """
    if np.any(values <= 0) or np.any(values >= 1):
        return np.clip(values, 1e-10, 1 - 1e-10)
    return values


def scatter_plot(
    df,
    env,
    scaling="Probit",
    do_domain=True,
    legend_change=False,
    show_linear_fits=True,
    colors="blue",
    fitcolor="black",
):
    """
    Scatter plot Xs against Ys, optionally scaling the data.
    """
    x_vals = validate_stability(np.array(df["x"].values).reshape(-1, 1))
    y_vals = validate_stability(np.array(df["y"].values).reshape(-1, 1))

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
        label += f"Model Architecture: {df['model_arch'].values[i]}<br>"
        label += f"Transfer Learning: {df['transfer'].values[i]}<br>"
        label += f"X: {x:.2f} <br>"
        label += f"Y: {y:.2f} <br>"
        return label

    # Create scatter plot
    traces = [
        go.Scatter(
            x=scaled_xs.flatten(),
            y=scaled_ys.flatten(),
            hoverinfo="text",
            mode="markers",
            marker=dict(color=colors, size=6),
            text=[label_point(i) for i in range(len(scaled_xs))],
            showlegend=False,
        )
    ]

    # Show linear fit for each environment
    if not legend_change:
        if show_linear_fits and do_domain:
            unique_envs = np.unique(env)
            for unique_env in unique_envs:
                env_mask = env == unique_env
                slope, intercept, pearson_corr, _, _ = linregress(
                    scaled_xs[env_mask].flatten(),
                    scaled_ys[env_mask].flatten(),
                )
                sample_pts = rescale(np.arange(0.0, 1.0, 0.01), scaling)
                traces.append(
                    go.Scatter(
                        mode="lines",
                        x=sample_pts,
                        y=slope * sample_pts + intercept,
                        name=f"Env {unique_env} Slope: {slope:.2f}<br>R: {pearson_corr:.2f}",
                        line=dict(color=COLOR_MAP[unique_env]),
                        showlegend=True,
                    )
                )
    else:
        # Change legend to model architecture
        unique_model_archs = df["model_arch"].unique()
        model_arch_to_color_idx = zip(
            range(len(unique_model_archs)), unique_model_archs
        )
        model_arch_color_dict = {
            model_arch: COLOR_MAP[idx]
            for idx, model_arch in model_arch_to_color_idx
        }

        for model_arch in unique_model_archs:
            model_arch_mask = df["model_arch"] == model_arch
            slope, intercept, pearson_corr, _, _ = linregress(
                scaled_xs[model_arch_mask].flatten(),
                scaled_ys[model_arch_mask].flatten(),
            )
            sample_pts = rescale(np.arange(0.0, 1.0, 0.01), scaling)
            traces.append(
                go.Scatter(
                    mode="lines",
                    x=sample_pts,
                    y=slope * sample_pts + intercept,
                    name=f"Model Arch: {model_arch} Slope: {slope:.2f}<br>R: {pearson_corr:.2f}",
                    line=dict(color=model_arch_color_dict[model_arch]),
                    showlegend=True,
                )
            )

    # Add linear fit for overall
    slope, intercept, pearson_corr, _, _ = linregress(
        scaled_xs.flatten(), scaled_ys.flatten()
    )
    sample_pts = rescale(np.arange(0.0, 1.0, 0.01), scaling)
    traces.append(
        go.Scatter(
            mode="lines",
            x=sample_pts,
            y=slope * sample_pts + intercept,
            name=f"Overall Slope: {slope:.2f}<br>R: {pearson_corr:.2f}",
            line=dict(color=fitcolor),
        )
    )
    env_df = display_env_data(df, env, scaling)
    return traces, env_df


def display_env_data(df, env, scaling="Probit"):
    """
    Create a DataFrame to display slope, R2, p-value, and standard error for each environment.
    """
    left_out_env = df["test_env"].values[0]
    x_vals = validate_stability(np.array(df["x"].values).reshape(-1, 1))
    y_vals = validate_stability(np.array(df["y"].values).reshape(-1, 1))

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


def plot(
    df,
    scaling="Probit",
    show_linear_fits=True,
    metric="Accuracy",
    all_shown=False,
    legend_change=False,
):
    """
    Generate an interactive scatter plot.
    """
    # Create the figure with a single explicit title instead of subplot titles
    fig = make_subplots(rows=1, cols=1)

    # Set the main title directly in the layout
    fig.update_layout(
        title=dict(
            text=f"Accuracy on the Line for Test Domain ({scaling})",
            x=0.5,  # Center the title
            xanchor="center",
        )
    )

    traces = []

    # Obtain color mapping for model architectures and environments
    model_arch_to_color_idx = zip(
        range(len(df["model_arch"].unique())), df["model_arch"].unique()
    )
    model_arch_color_dict = {
        model_arch: COLOR_MAP[idx]
        for idx, model_arch in model_arch_to_color_idx
    }
    model_arch_color_mapping = (
        df["model_arch"].map(model_arch_color_dict).values
    )
    env_color_mapping = [COLOR_MAP[x] for x in df["train_env"].values]

    # Plot scatter plot
    plot_traces, env_df = scatter_plot(
        df,
        env=df["train_env"].values,
        scaling=scaling,
        legend_change=legend_change,
        show_linear_fits=show_linear_fits,
        colors=(
            env_color_mapping
            if not legend_change
            else model_arch_color_mapping
        ),
    )

    # Plot line for y=x
    traces.extend(plot_traces)
    metric_min, metric_max = 0.01, 0.99
    traces.append(
        go.Scatter(
            mode="lines",
            x=rescale(np.arange(metric_min, metric_max + 0.01, 0.01), scaling),
            y=rescale(np.arange(metric_min, metric_max + 0.01, 0.01), scaling),
            name="y=x",
            line=dict(color="black", dash="dashdot"),
        )
    )

    # Add traces to figure
    for trace in traces:
        fig.add_trace(trace, row=1, col=1)

    # Adjust tickmarks
    tickmarks = np.array([0.1, 0.25, 0.5, 0.7, 0.8, 0.9, 0.95, metric_max])
    ticks = dict(
        tickmode="array",
        tickvals=rescale(tickmarks, scaling),
        ticktext=[f"{mark:.2f}" for mark in tickmarks],
        tickfont=dict(color="black"),
    )

    # Upload layout
    fig.update_layout(width=1000, height=700, xaxis=ticks, yaxis=ticks)
    return fig, env_df

    title = f"Accuracy on the Line for Test Domain"
    fig = make_subplots(
        rows=1,
        cols=1,
        subplot_titles=((f"{title} ({scaling})"),),
    )
    fig.update_annotations(font=dict(color="black"))
    traces = []

    # Obtain color mapping for model architectures and environments
    model_arch_to_color_idx = zip(
        range(len(df["model_arch"].unique())), df["model_arch"].unique()
    )
    model_arch_color_dict = {
        model_arch: COLOR_MAP[idx]
        for idx, model_arch in model_arch_to_color_idx
    }
    model_arch_color_mapping = (
        df["model_arch"].map(model_arch_color_dict).values
    )
    env_color_mapping = [COLOR_MAP[x] for x in df["train_env"].values]

    # Plot scatter plot
    plot_traces, env_df = scatter_plot(
        df,
        env=df["train_env"].values,
        scaling=scaling,
        legend_change=legend_change,
        show_linear_fits=show_linear_fits,
        colors=(
            env_color_mapping
            if not legend_change
            else model_arch_color_mapping
        ),
    )

    # Plot line for y=x
    traces.extend(plot_traces)
    metric_min, metric_max = 0.01, 0.99
    traces.append(
        go.Scatter(
            mode="lines",
            x=rescale(np.arange(metric_min, metric_max + 0.01, 0.01), scaling),
            y=rescale(np.arange(metric_min, metric_max + 0.01, 0.01), scaling),
            name="y=x",
            line=dict(color="black", dash="dashdot"),
        )
    )

    # Add traces to figure
    for trace in traces:
        fig.add_trace(trace, row=1, col=1)

    # Adjust tickmarks
    tickmarks = np.array([0.1, 0.25, 0.5, 0.7, 0.8, 0.9, 0.95, metric_max])
    ticks = dict(
        tickmode="array",
        tickvals=rescale(tickmarks, scaling),
        ticktext=[f"{mark:.2f}" for mark in tickmarks],
        tickfont=dict(color="black"),
    )

    # Upload layout
    fig.update_layout(width=1000, height=700, xaxis=ticks, yaxis=ticks)
    return fig, env_df
