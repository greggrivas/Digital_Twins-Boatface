import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import PartialDependenceDisplay
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor


BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = Path(__file__).resolve().parent
IMAGES_DIR = BASE_DIR.parent / "Images"
TARGETS = ["Compressor_Decay", "Turbine_Decay"]
VALID_TARGETS = set(TARGETS)


def slug_for_target(target):
    return "compressor" if target == "Compressor_Decay" else "turbine"


def sanitize_model_name(name):
    return (
        name.lower()
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("-", "_")
    )


def run_target_analysis(df, target):
    slug = slug_for_target(target)
    target_dir = IMAGES_DIR / slug
    target_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print(f"RUNNING MODEL ANALYSIS FOR: {target}")
    print("=" * 70)

    X = df.drop(columns=["Compressor_Decay", "Turbine_Decay", "T1", "P1"])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("=" * 60)
    print("BASELINE MODELS")
    print("=" * 60)

    dummy = DummyRegressor(strategy="mean")
    dummy.fit(X_train, y_train)
    dummy_preds = dummy.predict(X_test)
    print(
        f"Dummy Regressor (Mean): R² = {r2_score(y_test, dummy_preds):.4f}, "
        f"MAE = {mean_absolute_error(y_test, dummy_preds):.6f}"
    )

    linear = LinearRegression()
    linear.fit(X_train_scaled, y_train)
    linear_preds = linear.predict(X_test_scaled)
    print(
        f"Linear Regression:      R² = {r2_score(y_test, linear_preds):.4f}, "
        f"MAE = {mean_absolute_error(y_test, linear_preds):.6f}"
    )

    models = {
        "Dummy (Mean)": DummyRegressor(strategy="mean"),
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "SVR": SVR(kernel="rbf", epsilon=0.001, C=10),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    }

    perf_metrics = []
    trained_models = {}
    predictions = {}

    for name, model in models.items():
        needs_scaling = name in ["SVR", "Ridge Regression", "Linear Regression"]

        if needs_scaling:
            model.fit(X_train_scaled, y_train)
            preds = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

        trained_models[name] = model
        predictions[name] = preds

        mae = mean_absolute_error(y_test, preds)
        mse = mean_squared_error(y_test, preds)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, preds)
        perf_metrics.append(
            {"Model": name, "MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}
        )

        plt.figure(figsize=(6, 5))
        plt.scatter(y_test, preds, alpha=0.3, s=10)
        plt.plot([y.min(), y.max()], [y.min(), y.max()], "r--", lw=2)
        plt.title(f"{target} - {name}\nActual vs Predicted")
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.tight_layout()
        plt.savefig(target_dir / f"{sanitize_model_name(name)}_scatter.png", dpi=300)
        plt.close()

    perf_df = pd.DataFrame(perf_metrics)
    print("\n" + "=" * 60)
    print(f"PERFORMANCE SUMMARY: {target}")
    print("=" * 60)
    print(perf_df.to_string(index=False))
    perf_df.to_csv(MODELS_DIR / f"{slug}_metrics.csv", index=False)

    print("\n" + "=" * 60)
    print("K-FOLD CROSS-VALIDATION (K=5)")
    print("=" * 60)
    cv_results = []
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    cv_models = {
        "Linear Regression": make_pipeline(StandardScaler(), LinearRegression()),
        "Ridge Regression": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "SVR": make_pipeline(StandardScaler(), SVR(kernel="rbf", epsilon=0.001, C=10)),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    }

    for name, model in cv_models.items():
        scores = cross_val_score(model, X, y, cv=kfold, scoring="r2")
        cv_results.append(
            {
                "Model": name,
                "Mean R²": scores.mean(),
                "Std R²": scores.std(),
                "Min R²": scores.min(),
                "Max R²": scores.max(),
            }
        )
        print(f"{name}: R² = {scores.mean():.4f} ± {scores.std():.4f}")

    cv_df = pd.DataFrame(cv_results)
    cv_df.to_csv(MODELS_DIR / f"{slug}_cv_results.csv", index=False)

    plt.figure(figsize=(10, 6))
    x_pos = range(len(cv_df))
    plt.bar(
        x_pos,
        cv_df["Mean R²"],
        yerr=cv_df["Std R²"],
        capsize=5,
        color="steelblue",
        edgecolor="navy",
    )
    plt.xticks(x_pos, cv_df["Model"], rotation=45, ha="right")
    plt.ylabel("R² Score")
    plt.title(f"5-Fold Cross-Validation Results ({target})")
    plt.ylim(0, 1.1)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(target_dir / "cv_comparison.png", dpi=300)
    plt.close()

    print("\n" + "=" * 60)
    print("SEED SENSITIVITY ANALYSIS")
    print("=" * 60)
    seeds = [42, 123, 456, 789, 1011]
    seed_results = {"Decision Tree": [], "Random Forest": []}

    for seed in seeds:
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=seed)
        dt = DecisionTreeRegressor(random_state=seed)
        rf = RandomForestRegressor(n_estimators=100, random_state=seed)
        dt.fit(X_tr, y_tr)
        rf.fit(X_tr, y_tr)
        seed_results["Decision Tree"].append(r2_score(y_te, dt.predict(X_te)))
        seed_results["Random Forest"].append(r2_score(y_te, rf.predict(X_te)))

    for name, scores in seed_results.items():
        print(f"{name}: R² = {np.mean(scores):.4f} ± {np.std(scores):.4f}")

    plt.figure(figsize=(10, 6))
    x = np.arange(len(seeds))
    width = 0.35
    plt.bar(x - width / 2, seed_results["Decision Tree"], width, label="Decision Tree", color="orange")
    plt.bar(x + width / 2, seed_results["Random Forest"], width, label="Random Forest", color="forestgreen")
    plt.xlabel("Random Seed")
    plt.ylabel("R² Score")
    plt.title(f"Seed Sensitivity Analysis ({target})")
    plt.xticks(x, seeds)
    ymin = min(min(seed_results["Decision Tree"]), min(seed_results["Random Forest"])) - 0.01
    plt.ylim(max(-0.1, ymin), 1.01)
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(target_dir / "seed_sensitivity.png", dpi=300)
    plt.close()

    print("\n" + "=" * 60)
    print("RESIDUAL DIAGNOSTICS")
    print("=" * 60)
    best_preds = predictions["Random Forest"]
    residuals = y_test.values - best_preds
    print("Residual Statistics (Random Forest):")
    print(f"  Mean: {np.mean(residuals):.6f}")
    print(f"  Std:  {np.std(residuals):.6f}")
    print(f"  Min:  {np.min(residuals):.6f}")
    print(f"  Max:  {np.max(residuals):.6f}")

    from scipy import stats

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes[0, 0].scatter(best_preds, residuals, alpha=0.3, s=10)
    axes[0, 0].axhline(y=0, color="r", linestyle="--", lw=2)
    axes[0, 0].set_xlabel("Predicted Values")
    axes[0, 0].set_ylabel("Residuals")
    axes[0, 0].set_title("Residuals vs Predicted")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].hist(residuals, bins=50, edgecolor="black", alpha=0.7)
    axes[0, 1].axvline(x=0, color="r", linestyle="--", lw=2)
    axes[0, 1].set_xlabel("Residual Value")
    axes[0, 1].set_ylabel("Frequency")
    axes[0, 1].set_title("Residual Distribution")
    axes[0, 1].grid(True, alpha=0.3)

    stats.probplot(residuals, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title("Q-Q Plot")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].scatter(y_test, residuals, alpha=0.3, s=10)
    axes[1, 1].axhline(y=0, color="r", linestyle="--", lw=2)
    axes[1, 1].set_xlabel("Actual Values")
    axes[1, 1].set_ylabel("Residuals")
    axes[1, 1].set_title("Residuals vs Actual")
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle(f"Residual Diagnostics - Random Forest ({target})", fontsize=14)
    plt.tight_layout()
    plt.savefig(target_dir / "residual_diagnostics.png", dpi=300)
    plt.close()

    print("\n" + "=" * 60)
    print("REGIME-SPECIFIC ERROR ANALYSIS")
    print("=" * 60)
    regime_df = X_test.copy()
    regime_df["Actual"] = y_test.values
    regime_df["Predicted"] = best_preds
    regime_df["Residual"] = residuals
    regime_df["Abs_Error"] = np.abs(residuals)
    speed_bins = [0, 10, 15, 20, 30]
    speed_labels = ["Low (0-10)", "Medium (10-15)", "High (15-20)", "Very High (20+)"]
    regime_df["Speed_Regime"] = pd.cut(regime_df["Ship_Speed"], bins=speed_bins, labels=speed_labels)
    regime_metrics = regime_df.groupby("Speed_Regime", observed=False).agg(
        {"Abs_Error": ["mean", "std", "count"], "Residual": "mean"}
    ).round(6)
    regime_metrics.columns = ["MAE", "Std_Error", "Count", "Mean_Bias"]
    print(regime_metrics)
    regime_metrics.to_csv(MODELS_DIR / f"{slug}_regime_analysis.csv")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    regime_metrics["MAE"].plot(kind="bar", ax=axes[0], color="steelblue", edgecolor="navy")
    axes[0].set_ylabel("Mean Absolute Error")
    axes[0].set_title(f"MAE by Speed Regime ({target})")
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].grid(axis="y", linestyle="--", alpha=0.7)

    colors = ["green" if x >= 0 else "red" for x in regime_metrics["Mean_Bias"]]
    regime_metrics["Mean_Bias"].plot(kind="bar", ax=axes[1], color=colors, edgecolor="black")
    axes[1].axhline(y=0, color="black", linestyle="-", lw=1)
    axes[1].set_ylabel("Mean Bias (Actual - Predicted)")
    axes[1].set_title(f"Prediction Bias by Speed Regime ({target})")
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(target_dir / "regime_analysis.png", dpi=300)
    plt.close()

    ax = perf_df.set_index("Model")[["MAE", "MSE"]].plot(kind="bar", figsize=(10, 6))
    plt.yscale("log")
    plt.title(f"Error Magnitude Comparison ({target})")
    plt.ylabel("Value (Log Scale)")
    plt.xticks(rotation=45, ha="right")
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.tight_layout()
    plt.savefig(target_dir / "error_comparison.png", dpi=300)
    plt.close(ax.figure)

    plt.figure(figsize=(10, 6))
    colors = ["gray" if r2 < 0.5 else "steelblue" if r2 < 0.9 else "forestgreen" for r2 in perf_df["R2"]]
    plt.bar(perf_df["Model"], perf_df["R2"], color=colors, edgecolor="navy")
    plt.axhline(y=0, color="red", linestyle="--", lw=1, label="Baseline (R²=0)")
    plt.title(f"R² Comparison ({target})")
    plt.ylabel("R² Score")
    plt.ylim(-0.1, 1.1)
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(target_dir / "r2_comparison.png", dpi=300)
    plt.close()

    rf_model = trained_models["Random Forest"]
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)
    plt.figure(figsize=(10, 6))
    plt.title(f"Random Forest Feature Importance ({target})")
    plt.barh(range(len(indices)), importances[indices], align="center", color="forestgreen")
    plt.yticks(range(len(indices)), [X.columns[i] for i in indices])
    plt.xlabel("Relative Importance")
    plt.tight_layout()
    plt.savefig(target_dir / "feature_importance.png", dpi=300)
    plt.close()

    try:
        import shap

        explainer = shap.TreeExplainer(rf_model)
        shap_values = explainer.shap_values(X_test)

        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_test, show=False)
        plt.tight_layout()
        plt.savefig(target_dir / "shap_summary.png", bbox_inches="tight", dpi=300)
        plt.close()

        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
        plt.tight_layout()
        plt.savefig(target_dir / "shap_bar.png", bbox_inches="tight", dpi=300)
        plt.close()
    except Exception as e:
        print(f"SHAP skipped: {e}")

    try:
        top_features = [X.columns[i] for i in indices[-4:]]
        fig, ax = plt.subplots(figsize=(12, 8))
        PartialDependenceDisplay.from_estimator(rf_model, X_test, top_features, kind="average", ax=ax)
        plt.suptitle(f"Partial Dependence Plots ({target})", fontsize=14)
        plt.tight_layout()
        plt.savefig(target_dir / "partial_dependence.png", dpi=300)
        plt.close()
    except Exception as e:
        print(f"PDP skipped: {e}")

    print("\n" + "=" * 60)
    print(f"ANALYSIS COMPLETE FOR: {target}")
    print("=" * 60)
    print(f"Metric tables saved to: {MODELS_DIR}/")
    print(f"Figures saved to: {target_dir}/")


def main():
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(BASE_DIR / "cleaned_data.csv")

    selected = os.getenv("TARGET")
    if selected:
        if selected not in VALID_TARGETS:
            raise ValueError("TARGET must be 'Compressor_Decay' or 'Turbine_Decay'")
        targets = [selected]
    else:
        targets = TARGETS

    for target in targets:
        run_target_analysis(df, target)


if __name__ == "__main__":
    main()
