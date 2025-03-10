# simulation.py

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split

from .utils import (
    evaluate_models,
    generate_matrix,
    generate_samples,
    plot_accuracies,
)


def format_matrix(matrix):
    """Format numpy array as string for display with 1 decimal place"""
    return "\n".join([",".join([f"{x:.1f}" for x in row]) for row in matrix])


def format_vector(vector):
    """Format numpy array as comma-separated string with 1 decimal place"""
    return ",".join([f"{x:.1f}" for x in vector])


def run_simulation():
    st.title(
        "Domain Generalization Benchmarks and Accuracy on the Line Simulation"
    )

    # Basic parameters
    st.sidebar.header("Parameters")
    # Add dark mode toggle at the top of parameters
    is_dark_mode = st.sidebar.checkbox("Use dark mode for plots")

    dg_dim = st.sidebar.number_input(
        "Domain General Dimension", min_value=1, max_value=10, value=2
    )
    spu_dim = st.sidebar.number_input(
        "Spurious Dimension", min_value=1, max_value=10, value=2
    )

    # Feature predictiveness
    st.sidebar.subheader("Feature Predictiveness")
    dg_pred = st.sidebar.slider(
        "Domain General Predictiveness",
        min_value=0.5,
        max_value=1.0,
        value=0.7,
        step=0.05,
        help="How predictive domain-general features are of Y (0.5 = random, 1.0 = perfect)",
    )
    spu_pred = st.sidebar.slider(
        "Spurious Predictiveness",
        min_value=0.5,
        max_value=1.0,
        value=0.8,
        step=0.05,
        help="How predictive spurious features are of Y (0.5 = random, 1.0 = perfect)",
    )
    joint_pred = st.sidebar.slider(
        "Joint Predictiveness",
        min_value=0.5,
        max_value=1.0,
        value=0.9,
        step=0.05,
        help="How predictive all features together are of Y (0.5 = random, 1.0 = perfect)",
    )

    # Number of samples and models
    n_samples = st.sidebar.number_input(
        "Number of Samples", min_value=10, max_value=10000, value=1000
    )
    n_models = st.sidebar.number_input(
        "Number of Models", min_value=2, max_value=1000, value=20
    )

    st.sidebar.markdown("---")

    # Domain General Features
    with st.sidebar.expander("Domain General Features", expanded=True):
        # Mean parameters
        mean_dg_type = st.selectbox(
            "Mean Type (Domain General)", ["ones", "random", "manual"], index=0
        )
        mean_dg_scale = st.number_input(
            "Mean Scale (Domain General)", value=1.0, step=0.1
        )

        # Generate initial mean vector based on type
        if mean_dg_type == "ones":
            mean_dg_default = format_vector(np.ones(dg_dim))
        elif mean_dg_type == "random":
            mean_dg_default = format_vector(np.random.randn(dg_dim))
        else:  # manual
            mean_dg_default = ",".join(["1"] * dg_dim)

        # Show editable mean vector
        mean_input_dg = st.text_area(
            "Mean Vector (Domain General, comma-separated):", mean_dg_default
        )

        try:
            mean_dg = (
                np.array([float(x) for x in mean_input_dg.split(",")])
                * mean_dg_scale
            )
            if len(mean_dg) != dg_dim:
                st.error(f"Mean vector must have {dg_dim} elements")
                return
        except:
            st.error("Invalid mean vector format for Domain General features")
            return

        # Covariance parameters
        cov_dg_type = st.selectbox(
            "Covariance Type (Domain General)",
            ["identity", "random", "manual"],
            index=0,
        )
        cov_dg_scale = st.number_input(
            "Covariance Scale (Domain General)", value=1.0, step=0.1
        )

        # Generate initial covariance matrix based on type
        if cov_dg_type == "identity":
            cov_dg_default = format_matrix(np.eye(dg_dim))
        elif cov_dg_type == "random":
            A = np.random.randn(dg_dim, dg_dim)
            cov_dg_default = format_matrix(A @ A.T + np.eye(dg_dim))
        else:  # manual
            cov_dg_default = "\n".join(
                [
                    ",".join(["1" if i == j else "0" for j in range(dg_dim)])
                    for i in range(dg_dim)
                ]
            )

        # Show editable covariance matrix
        cov_input_dg = st.text_area(
            "Covariance Matrix (Domain General, comma-separated rows):",
            cov_dg_default,
        )

        try:
            cov_dg = (
                np.array(
                    [
                        list(map(float, row.split(",")))
                        for row in cov_input_dg.strip().split("\n")
                    ]
                )
                * cov_dg_scale
            )
            if cov_dg.shape != (dg_dim, dg_dim):
                st.error(f"Covariance matrix must be {dg_dim}x{dg_dim}")
                return
        except:
            st.error(
                "Invalid covariance matrix format for Domain General features"
            )
            return

        if not np.all(np.linalg.eigvals(cov_dg) > 0):
            st.error(
                "Domain General covariance matrix is not positive definite"
            )
            return

    # Spurious Features
    with st.sidebar.expander("Spurious Features", expanded=True):
        # Mean parameters
        mean_spu_type = st.selectbox(
            "Mean Type (Spurious)", ["ones", "random", "manual"], index=0
        )
        mean_spu_scale = st.number_input(
            "Mean Scale (Spurious)", value=1.0, step=0.1
        )

        # Generate initial mean vector based on type
        if mean_spu_type == "ones":
            mean_spu_default = format_vector(np.ones(spu_dim))
        elif mean_spu_type == "random":
            mean_spu_default = format_vector(np.random.randn(spu_dim))
        else:  # manual
            mean_spu_default = ",".join(["1"] * spu_dim)

        # Show editable mean vector
        mean_input_spu = st.text_area(
            "Mean Vector (Spurious, comma-separated):", mean_spu_default
        )

        try:
            mean_spu = (
                np.array([float(x) for x in mean_input_spu.split(",")])
                * mean_spu_scale
            )
            if len(mean_spu) != spu_dim:
                st.error(f"Mean vector must have {spu_dim} elements")
                return
        except:
            st.error("Invalid mean vector format for Spurious features")
            return

        # Covariance parameters
        cov_spu_type = st.selectbox(
            "Covariance Type (Spurious)",
            ["identity", "random", "manual"],
            index=0,
        )
        cov_spu_scale = st.number_input(
            "Covariance Scale (Spurious)", value=1.0, step=0.1
        )

        # Generate initial covariance matrix based on type
        if cov_spu_type == "identity":
            cov_spu_default = format_matrix(np.eye(spu_dim))
        elif cov_spu_type == "random":
            A = np.random.randn(spu_dim, spu_dim)
            cov_spu_default = format_matrix(A @ A.T + np.eye(spu_dim))
        else:  # manual
            cov_spu_default = "\n".join(
                [
                    ",".join(["1" if i == j else "0" for j in range(spu_dim)])
                    for i in range(spu_dim)
                ]
            )

        # Show editable covariance matrix
        cov_input_spu = st.text_area(
            "Covariance Matrix (Spurious, comma-separated rows):",
            cov_spu_default,
        )

        try:
            cov_spu = (
                np.array(
                    [
                        list(map(float, row.split(",")))
                        for row in cov_input_spu.strip().split("\n")
                    ]
                )
                * cov_spu_scale
            )
            if cov_spu.shape != (spu_dim, spu_dim):
                st.error(f"Covariance matrix must be {spu_dim}x{spu_dim}")
                return
        except:
            st.error("Invalid covariance matrix format for Spurious features")
            return

        if not np.all(np.linalg.eigvals(cov_spu) > 0):
            st.error("Spurious covariance matrix is not positive definite")
            return

    # Shift matrix
    with st.sidebar.expander("Shift Matrix", expanded=True):
        shift_type = st.selectbox(
            "Shift Matrix Type",
            ["identity", "pd", "psd", "nd", "nsd", "manual"],
            index=0,
        )
        shift_scale = st.number_input("Shift Scale", value=1.0, step=0.1)

        # Generate initial shift matrix based on type
        if shift_type == "manual":
            shift_default = "\n".join(
                [
                    ",".join(["1" if i == j else "0" for j in range(spu_dim)])
                    for i in range(spu_dim)
                ]
            )
        else:
            shift_matrix = generate_matrix(spu_dim, shift_type, shift_scale)
            shift_default = format_matrix(shift_matrix)

        # Show editable shift matrix
        shift_input = st.text_area(
            "Shift Matrix (comma-separated rows):", shift_default
        )

        try:
            shift_matrix = (
                np.array(
                    [
                        list(map(float, row.split(",")))
                        for row in shift_input.strip().split("\n")
                    ]
                )
                * shift_scale
            )
            if shift_matrix.shape != (spu_dim, spu_dim):
                st.error(f"Shift matrix must be {spu_dim}x{spu_dim}")
                return
        except:
            st.error("Invalid shift matrix format")
            return

    if st.button("Generate and Evaluate"):
        # Generate samples
        samples = generate_samples(
            mean_dg,
            cov_dg,
            mean_spu,
            cov_spu,
            shift_matrix,
            dg_pred=dg_pred,
            spu_pred=spu_pred,
            joint_pred=joint_pred,
            n_samples=n_samples,
        )

        # Evaluate models
        id_accs, ood_accs, weights = evaluate_models(
            samples["X_id_train"],
            samples["y_train"],
            samples["X_id_test"],
            samples["y_test"],
            samples["X_ood_test"],
            samples["y_ood_test"],
            dg_dim=dg_dim,
            n_models=n_models,
        )

        # Update plot colors based on theme
        plt.style.use("dark_background" if is_dark_mode else "default")

        # Plot results with adjusted colors
        fig = plot_accuracies(
            id_accs,
            ood_accs,
            weights,
            samples["optimal_dg_acc"],
            is_dark_mode=is_dark_mode,
        )

        # Reset style to prevent affecting other plots
        plt.style.use("default")
        st.pyplot(fig)

        # Display statistics
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("ID Performance")
            st.write(f"Mean: {np.mean(id_accs):.3f}")
            st.write(f"Std: {np.std(id_accs):.3f}")

        with col2:
            st.subheader("OOD Performance")
            st.write(f"Mean: {np.mean(ood_accs):.3f}")
            st.write(f"Std: {np.std(ood_accs):.3f}")

        # Additional correlation analysis
        st.subheader("Analysis")
        corr = np.corrcoef(id_accs, ood_accs)[0, 1]
        st.write(f"Correlation between ID and OOD accuracy: {corr:.3f}")

        # Weight ratio analysis
        st.write(
            f"Average domain-general feature weight ratio: {np.mean(weights):.3f}"
        )

        # Show correlation between weight ratio and performance
        weight_id_corr = np.corrcoef(weights, id_accs)[0, 1]
        weight_ood_corr = np.corrcoef(weights, ood_accs)[0, 1]
        st.write(
            f"Correlation between weight ratio and ID accuracy: {weight_id_corr:.3f}"
        )
        st.write(
            f"Correlation between weight ratio and OOD accuracy: {weight_ood_corr:.3f}"
        )

        # Display actual predictiveness achieved
        st.subheader("Achieved Predictiveness")
        st.write(
            f"Domain General Predictiveness: {samples['optimal_dg_acc']:.3f}"
        )
        st.write(f"Spurious Predictiveness: {samples['optimal_spu_acc']:.3f}")
        st.write(
            f"Joint Predictiveness: {samples['optimal_combined_acc']:.3f}"
        )
