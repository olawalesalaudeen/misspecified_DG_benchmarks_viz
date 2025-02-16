import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.stats import norm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def generate_matrix(size, matrix_type, scale=1, seed=42):
    """Generate different types of matrices."""
    np.random.seed(seed)
    A = np.random.randn(size, size)

    if matrix_type == "identity":
        return np.identity(size) * scale
    elif matrix_type == "pd":
        return np.dot(A, A.T) * scale + np.eye(size) * 1e-3
    elif matrix_type == "psd":
        return np.dot(A, A.T) * scale
    elif matrix_type == "nd":
        return -np.dot(A, A.T) * scale
    elif matrix_type == "nsd":
        return -np.dot(A, A.T) * scale - np.eye(size) * 1e-3
    return A * scale


def compute_optimal_dg_accuracy(Z_dg_train, y_train, Z_dg_test, y_test):
    """Compute optimal accuracy achievable using only domain-general features"""
    # Try several models with good parameters to get best possible accuracy
    best_acc = 0
    models = [
        LogisticRegression(C=1.0),
        LogisticRegression(C=10.0),
        SVC(kernel="rbf", C=1.0),
        RandomForestClassifier(n_estimators=100),
    ]

    for model in models:
        model.fit(Z_dg_train, y_train)
        acc = model.score(Z_dg_test, y_test)
        best_acc = max(best_acc, acc)

    return best_acc


def adjust_gaussian_params(mean, cov, target_acc):
    """Adjust Gaussian parameters to achieve target accuracy"""
    # Scale mean to control separation between classes
    scale = (target_acc - 0.5) * 4  # Heuristic scaling
    return mean * scale, cov


def generate_samples(
    mean_dg,
    cov_dg,
    mean_spu,
    cov_spu,
    shift_matrix,
    dg_pred=0.7,
    spu_pred=0.8,
    joint_pred=0.9,
    n_samples=1000,
    seed=42,
):
    """
    Generate samples with controlled predictiveness for each feature type.

    Parameters:
    - mean_dg, cov_dg: domain-general feature parameters
    - mean_spu, cov_spu: spurious feature parameters
    - shift_matrix: matrix for OOD transformation
    - dg_pred: target predictiveness of domain-general features (0.5-1.0)
    - spu_pred: target predictiveness of spurious features (0.5-1.0)
    - joint_pred: target predictiveness of combined features (0.5-1.0)
    """
    np.random.seed(seed)

    # Adjust means to achieve target predictiveness
    mean_dg_scaled, cov_dg_scaled = adjust_gaussian_params(
        mean_dg, cov_dg, dg_pred
    )
    mean_spu_scaled, cov_spu_scaled = adjust_gaussian_params(
        mean_spu, cov_spu, spu_pred
    )

    # Sample labels for ID data
    Y_id = np.random.choice([0, 1], size=n_samples)

    # Generate domain-general features for ID data
    Z_dg_id = np.array(
        [
            np.random.multivariate_normal(y * mean_dg_scaled, cov_dg_scaled)
            for y in Y_id
        ]
    )

    # Generate spurious features for ID data
    Z_spu_id = np.array(
        [
            np.random.multivariate_normal(y * mean_spu_scaled, cov_spu_scaled)
            for y in Y_id
        ]
    )

    # Generate OOD data
    Y_ood = np.random.choice([0, 1], size=n_samples)

    # Generate domain-general features for OOD (same distribution)
    Z_dg_ood = np.array(
        [
            np.random.multivariate_normal(y * mean_dg_scaled, cov_dg_scaled)
            for y in Y_ood
        ]
    )

    # Generate spurious features for OOD and apply shift
    Z_spu_ood_base = np.array(
        [
            np.random.multivariate_normal(y * mean_spu_scaled, cov_spu_scaled)
            for y in Y_ood
        ]
    )
    Z_spu_ood = Z_spu_ood_base @ shift_matrix

    # Combine features
    X_id = np.hstack([Z_dg_id, Z_spu_id])
    X_ood = np.hstack([Z_dg_ood, Z_spu_ood])

    # Split ID data into train/test
    X_id_train, X_id_test, y_train, y_test = train_test_split(
        X_id, Y_id, test_size=0.2, random_state=seed
    )

    # Compute accuracies for each feature type
    Z_dg_train = X_id_train[:, : Z_dg_id.shape[1]]
    Z_dg_test = X_id_test[:, : Z_dg_id.shape[1]]
    optimal_dg_acc = compute_optimal_dg_accuracy(
        Z_dg_train, y_train, Z_dg_test, y_test
    )

    Z_spu_train = X_id_train[:, Z_dg_id.shape[1] :]
    Z_spu_test = X_id_test[:, Z_dg_id.shape[1] :]
    optimal_spu_acc = compute_optimal_dg_accuracy(
        Z_spu_train, y_train, Z_spu_test, y_test
    )

    optimal_combined_acc = compute_optimal_dg_accuracy(
        X_id_train, y_train, X_id_test, y_test
    )

    return {
        "X_id_train": X_id_train,
        "X_id_test": X_id_test,
        "y_train": y_train,
        "y_test": y_test,
        "X_ood_test": X_ood,
        "y_ood_test": Y_ood,
        "Z_dg_id": Z_dg_id,
        "Z_spu_id": Z_spu_id,
        "Z_dg_ood": Z_dg_ood,
        "Z_spu_ood": Z_spu_ood,
        "optimal_dg_acc": optimal_dg_acc,
        "optimal_spu_acc": optimal_spu_acc,
        "optimal_combined_acc": optimal_combined_acc,
    }


def plot_accuracies(
    id_accs, ood_accs, weights_ratio=None, optimal_dg_acc=None
):
    """Plot accuracies with optimal Z_dg accuracy line."""
    eps = 1e-10
    id_accs_adj = np.clip(id_accs, eps, 1 - eps)
    ood_accs_adj = np.clip(ood_accs, eps, 1 - eps)
    id_accs_probit = norm.ppf(id_accs_adj)
    ood_accs_probit = norm.ppf(ood_accs_adj)

    # Calculate axis ranges with a 5% buffer
    min_probit = min(id_accs_probit.min(), ood_accs_probit.min())
    max_probit = max(id_accs_probit.max(), ood_accs_probit.max())
    range_buffer = (max_probit - min_probit) * 0.05
    x_range = np.array([min_probit - range_buffer, max_probit + range_buffer])

    fig, ax = plt.subplots(figsize=(8, 8))
    if weights_ratio is not None:
        scatter = ax.scatter(
            id_accs_probit,
            ood_accs_probit,
            c=weights_ratio,
            cmap="viridis",
            alpha=0.6,
        )
        plt.colorbar(scatter, label="Domain General Feature Weight Ratio")
    else:
        ax.scatter(id_accs_probit, ood_accs_probit, alpha=0.6)

    slope, intercept, r_value, p_value, std_err = stats.linregress(
        id_accs_probit, ood_accs_probit
    )
    y_fit = slope * x_range + intercept
    ax.plot(
        x_range,
        y_fit,
        "b-",
        label=f"Linear fit (slope={slope:.2f}, R²={r_value**2:.2f})",
    )
    ax.plot(x_range, x_range, "r--", label="y=x")

    if optimal_dg_acc is not None:
        optimal_dg_acc_probit = norm.ppf(np.clip(optimal_dg_acc, eps, 1 - eps))
        ax.axhline(
            y=optimal_dg_acc_probit,
            color="g",
            linestyle=":",
            label=f"Optimal Z_dg acc: {optimal_dg_acc:.3f}",
        )

    ax.set_xlim(x_range)
    ax.set_ylim(x_range)

    probit_ticks = np.linspace(x_range[0], x_range[1], 7)
    acc_ticks = norm.cdf(probit_ticks)

    ax.set_xticks(probit_ticks)
    ax.set_yticks(probit_ticks)
    ax.set_xticklabels([f"{x:.3f}" for x in acc_ticks])
    ax.set_yticklabels([f"{x:.3f}" for x in acc_ticks])

    ax.set_xlabel("ID Accuracy")
    ax.set_ylabel("OOD Accuracy")
    ax.set_title("ID vs OOD Performance (Probit Scale)")
    ax.grid(True)
    ax.legend()
    return fig


def create_model():
    """Create a random model with various architectures, favoring potentially good models"""
    model_type = np.random.choice(
        [
            "logistic",
            "logistic",
            "logistic",
            "logistic",  # Higher weight to logistic
            "svm",
            "svm",
            "mlp",
            "random_forest",
        ]
    )

    is_poor = np.random.random() < 0.02  # 2% chance of poor model

    if model_type == "logistic":
        if is_poor:
            params = {
                "C": np.exp(np.random.uniform(-20, -15)),
                "penalty": np.random.choice(["l1", "l2"]),
                "solver": "saga",
                "max_iter": np.random.randint(100, 300),
                "random_state": np.random.randint(0, 10000),
            }
        else:
            params = {
                "C": np.exp(np.random.uniform(-3, 3)),
                "penalty": np.random.choice(["l1", "l2", "elasticnet"]),
                "solver": "saga",
                "max_iter": 2000,
                "random_state": np.random.randint(0, 10000),
            }
            if params["penalty"] == "elasticnet":
                params["l1_ratio"] = np.random.beta(
                    2, 2
                )  # Favor middle values
        model = LogisticRegression(**params)

    elif model_type == "svm":
        if is_poor:
            params = {
                "C": np.exp(np.random.uniform(-20, -15)),
                "kernel": "rbf",
                "gamma": np.exp(np.random.uniform(3, 5)),
                "random_state": np.random.randint(0, 10000),
            }
        else:
            params = {
                "C": np.exp(np.random.uniform(-3, 3)),
                "kernel": np.random.choice(["linear", "rbf"]),
                "random_state": np.random.randint(0, 10000),
            }
            if params["kernel"] == "rbf":
                params["gamma"] = np.exp(np.random.uniform(-5, 0))
        model = SVC(**params)

    elif model_type == "mlp":
        if is_poor:
            params = {
                "hidden_layer_sizes": (1,),
                "alpha": np.exp(np.random.uniform(-15, -10)),
                "learning_rate_init": np.exp(np.random.uniform(-8, -6)),
                "max_iter": np.random.randint(100, 300),
                "random_state": np.random.randint(0, 10000),
            }
        else:
            params = {
                "hidden_layer_sizes": tuple(
                    np.random.randint(5, 50, size=np.random.randint(1, 3))
                ),
                "alpha": np.exp(np.random.uniform(-5, 0)),
                "learning_rate_init": np.exp(np.random.uniform(-4, -2)),
                "max_iter": 1000,
                "random_state": np.random.randint(0, 10000),
            }
        model = MLPClassifier(**params)

    else:  # random_forest
        if is_poor:
            params = {
                "n_estimators": np.random.randint(1, 5),
                "max_depth": 1,
                "min_samples_split": np.random.randint(50, 100),
                "random_state": np.random.randint(0, 10000),
            }
        else:
            params = {
                "n_estimators": np.random.randint(50, 200),
                "max_depth": np.random.randint(3, 10),
                "min_samples_split": np.random.randint(2, 10),
                "random_state": np.random.randint(0, 10000),
            }
        model = RandomForestClassifier(**params)

    return model, is_poor


def generate_feature_weights(n_features, dg_dim, strategy):
    """Generate feature weights based on different strategies"""
    feature_weights = np.ones(n_features)

    if strategy == "uniform":
        feature_weights = np.random.uniform(0.5, 1.5, n_features)
    elif strategy == "beta":
        a = np.random.uniform(1, 3)  # More centered values
        b = np.random.uniform(1, 3)
        feature_weights = np.random.beta(a, b, n_features)
    elif strategy == "gaussian":
        mu = 1.0
        sigma = 0.3
        feature_weights = np.abs(np.random.normal(mu, sigma, n_features))
    elif strategy == "balanced":
        feature_weights = np.random.normal(1, 0.1, n_features)
    elif strategy == "extreme":
        cutoff = np.random.uniform(0.1, 0.9)
        feature_weights = np.where(
            np.random.random(n_features) > cutoff,
            np.random.uniform(0.1, 0.5),
            np.random.uniform(1.5, 3),
        )
    elif strategy == "zero_heavy":
        feature_weights = np.where(
            np.random.random(n_features) > 0.2,
            0.1,
            np.random.uniform(0.5, 2, n_features),
        )
    elif strategy == "single_feature":
        feature_weights = np.ones(n_features) * 0.1
        feature_weights[np.random.randint(n_features)] = np.random.uniform(
            2, 5
        )

    return feature_weights


def evaluate_models(
    X_id_train,
    y_train,
    X_id_test,
    y_test,
    X_ood_test,
    y_ood_test,
    dg_dim,
    n_models=100,
):
    """Generate diverse models with focus on meaningful performance range"""
    id_accuracies = []
    ood_accuracies = []
    weights_ratio = []

    base_strategies = [
        "balanced",
        "gaussian",
        "beta",
    ]  # Most reasonable strategies
    other_strategies = ["uniform", "single_feature"]  # Less extreme variations
    extreme_strategies = ["extreme", "zero_heavy"]  # More extreme cases

    scaler = StandardScaler()
    X_id_train_scaled = scaler.fit_transform(X_id_train)
    X_id_test_scaled = scaler.transform(X_id_test)
    X_ood_test_scaled = scaler.transform(X_ood_test)

    for _ in range(n_models):
        X_train = X_id_train_scaled.copy()
        y_train_model = y_train.copy()
        n_features = X_train.shape[1]

        # Strategy selection with controlled probabilities
        rand_val = np.random.random()
        if rand_val < 0.7:  # 70% chance for base strategies
            strategy = np.random.choice(base_strategies)
        elif rand_val < 0.9:  # 20% chance for other strategies
            strategy = np.random.choice(other_strategies)
        else:  # 10% chance for extreme strategies
            strategy = np.random.choice(extreme_strategies)

        feature_weights = generate_feature_weights(
            n_features, dg_dim, strategy
        )

        # Reduced probability of data modifications
        if np.random.random() < 0.05:  # 5% chance for feature corruption
            corrupt_features = np.random.choice(
                n_features,
                size=np.random.randint(1, max(2, n_features // 4)),
                replace=False,
            )
            noise_level = np.random.uniform(0.1, 5)  # Reduced noise level
            X_train[:, corrupt_features] += np.random.normal(
                0, noise_level, size=X_train[:, corrupt_features].shape
            )
            X_id_test_scaled[:, corrupt_features] += np.random.normal(
                0,
                noise_level,
                size=X_id_test_scaled[:, corrupt_features].shape,
            )
            X_ood_test_scaled[:, corrupt_features] += np.random.normal(
                0,
                noise_level,
                size=X_ood_test_scaled[:, corrupt_features].shape,
            )

        # Weight the features
        X_train_weighted = X_train * feature_weights
        X_id_test_weighted = X_id_test_scaled * feature_weights
        X_ood_test_weighted = X_ood_test_scaled * feature_weights

        try:
            model, is_poor = create_model()
            model.fit(X_train_weighted, y_train_model)

            if (
                is_poor and np.random.random() < 0.5
            ):  # 50% chance to invert poor model predictions
                id_acc = 1 - model.score(X_id_test_weighted, y_test)
                ood_acc = 1 - model.score(X_ood_test_weighted, y_ood_test)
            else:
                id_acc = model.score(X_id_test_weighted, y_test)
                ood_acc = model.score(X_ood_test_weighted, y_ood_test)

            ratio = np.sum(feature_weights[:dg_dim]) / (
                np.sum(feature_weights) + 1e-10
            )

            id_accuracies.append(id_acc)
            ood_accuracies.append(ood_acc)
            weights_ratio.append(ratio)

        except Exception as e:
            continue

    valid_models = len(id_accuracies)
    if valid_models < n_models:
        print(
            f"Note: {n_models - valid_models} models failed to converge and were skipped."
        )

    return (
        np.array(id_accuracies),
        np.array(ood_accuracies),
        np.array(weights_ratio),
    )
