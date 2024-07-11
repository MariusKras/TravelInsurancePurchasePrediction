import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import mannwhitneyu, chi2_contingency
from sklearn.metrics import (
    f1_score,
    classification_report,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    confusion_matrix,
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from itertools import product
from sklearn.model_selection import KFold


def insert_space_before_capitals(text: str) -> str:
    """
    Insert a space before each uppercase letter in the given text.

    Args:
    - text (str): The input text where spaces will be inserted before uppercase letters.

    Returns:
    - str: The modified text with spaces inserted before uppercase letters.

    Example:
    >>> insert_space_before_capitals("ThisIsAString")
    'This Is A String'
    """
    new_text = ""
    for char in text:
        if char.isupper() and new_text:
            new_text += " "
        new_text += char
    return new_text


def numeric_relationships(dataframe: pd.DataFrame) -> None:
    """
    Analyzes the relationship between numeric predictors and travel insurance status.

    Args:
        dataframe (pd.DataFrame): The input DataFrame containing the data.

    The function will:
    1. Select numeric columns and the 'TravelInsurance' column.
    2. Generate distribution and boxplot visuals for each numeric predictor by travel insurance status.
    3. Perform Mann-Whitney U test and bootstrap analysis for differences in medians.

    Returns:
        None
    """
    selected_columns = dataframe.select_dtypes(include=["number"]).columns.tolist()
    selected_columns.append("TravelInsurance")
    temp_dataframe = dataframe[selected_columns].copy()
    temp_dataframe.columns = [
        insert_space_before_capitals(col) for col in temp_dataframe.columns
    ]
    for predictor_name in temp_dataframe.columns:
        if predictor_name != "Travel Insurance":
            fig, axs = plt.subplots(1, 2, figsize=(12, 4))
            fig.suptitle(
                f"Distribution of {predictor_name} by Travel Insurance Status",
                fontsize=13,
            )
            sns.histplot(
                data=temp_dataframe,
                x=predictor_name,
                hue="Travel Insurance",
                multiple="stack",
                edgecolor="white",
                ax=axs[0],
                bins=21,
                zorder=2,
                linewidth=0.5,
                alpha=0.92,
            )
            axs[0].set_xlabel(predictor_name)
            axs[0].set_ylabel("Frequency")
            axs[0].tick_params(axis="both", labelsize=9, length=0)
            axs[0].grid(alpha=0.2, axis="y", zorder=0)
            sns.despine(ax=axs[0], left=True, bottom=True)
            sns.boxplot(
                data=temp_dataframe,
                y=predictor_name,
                x="Travel Insurance",
                ax=axs[1],
                zorder=2,
            )
            axs[1].set_ylabel(predictor_name)
            axs[1].set_xlabel("Travel Insurance")
            axs[1].tick_params(axis="both", labelsize=9, length=0)
            axs[1].grid(alpha=0.2, axis="y", zorder=0)
            sns.despine(ax=axs[1], left=True, bottom=True)
            plt.tight_layout()
            plt.show()

            data_yes = temp_dataframe[temp_dataframe["Travel Insurance"] == "Yes"][
                predictor_name
            ]
            data_no = temp_dataframe[temp_dataframe["Travel Insurance"] == "No"][
                predictor_name
            ]
            stat, p_value = mannwhitneyu(data_yes, data_no)
            print(f"{predictor_name} - Travel Insurance:")
            print(f"Mann-Whitney U test p-value: {p_value:.2f}")

            data = temp_dataframe[predictor_name].values.flatten()
            n_iterations = 1000
            medians_yes = np.zeros(n_iterations)
            medians_no = np.zeros(n_iterations)
            size_yes = len(temp_dataframe[temp_dataframe["Travel Insurance"] == "Yes"])
            for i in range(n_iterations):
                sample_combined = np.random.choice(data, size=len(data), replace=True)
                sample_yes = sample_combined[:size_yes]
                sample_no = sample_combined[size_yes:]
                medians_yes[i] = np.median(sample_yes)
                medians_no[i] = np.median(sample_no)
            diff_medians = medians_yes - medians_no
            data_yes = temp_dataframe[temp_dataframe["Travel Insurance"] == "Yes"][
                predictor_name
            ].values.flatten()
            data_no = temp_dataframe[temp_dataframe["Travel Insurance"] == "No"][
                predictor_name
            ].values.flatten()
            observed_diff = np.median(data_yes) - np.median(data_no)
            p_value = np.mean(np.abs(diff_medians) >= np.abs(observed_diff))
            print(f"Bootstrap p-value for difference in medians: {p_value:.2f}\n")


def categorical_relationships(dataframe: pd.DataFrame) -> None:
    """
    Analyzes the relationship between categorical predictors and travel insurance status.

    Args:
        dataframe (pd.DataFrame): The input DataFrame containing the data.

    The function will:
    1. Select categorical columns.
    2. Generate distribution and proportion bar plots for each categorical predictor by travel insurance status.
    3. Perform Chi-Square test for association between predictors and travel insurance status.

    Returns:
        None
    """
    selected_columns = dataframe.select_dtypes(include=["category"]).columns.tolist()
    temp_dataframe = dataframe[selected_columns].copy()
    temp_dataframe.columns = [
        insert_space_before_capitals(col) if col != "Employment Type" else col
        for col in temp_dataframe.columns
    ]
    for predictor_name in temp_dataframe.columns:
        if predictor_name != "Travel Insurance":
            fig, axs = plt.subplots(1, 2, figsize=(12, 4))
            fig.suptitle(
                f"Distribution of {predictor_name} by Travel Insurance Status",
                fontsize=13,
            )
            crosstab = pd.crosstab(
                temp_dataframe[predictor_name], temp_dataframe["Travel Insurance"]
            )
            crosstab.plot(
                kind="bar", stacked=True, ax=axs[0], edgecolor="white", linewidth=0.5
            )
            axs[0].set_xlabel(predictor_name)
            axs[0].set_ylabel("Count")
            axs[0].tick_params(axis="both", labelsize=9, length=0)
            axs[0].grid(alpha=0.2, axis="y", zorder=0)
            sns.despine(ax=axs[0], left=True, bottom=True)
            axs[0].set_xticklabels(axs[0].get_xticklabels(), rotation=0)
            for patch in axs[0].patches:
                patch.set_zorder(2)
            crosstab_normalized = crosstab.div(crosstab.sum(axis=1), axis=0)
            crosstab_normalized.plot(
                kind="bar", stacked=True, ax=axs[1], edgecolor="white", linewidth=0.5
            )
            axs[1].set_ylabel("Proportion")
            axs[1].set_xlabel(predictor_name)
            axs[1].tick_params(axis="both", labelsize=9, length=0)
            axs[1].grid(alpha=0.2, axis="y", zorder=0)
            sns.despine(ax=axs[1], left=True, bottom=True)
            axs[1].set_xticklabels(axs[1].get_xticklabels(), rotation=0)
            axs[1].legend().set_visible(False)
            for patch in axs[1].patches:
                patch.set_zorder(2)
            plt.tight_layout()
            plt.show()

            _, p_value, _, _ = chi2_contingency(crosstab)
            print(
                f"{predictor_name} - Travel Insurance:\nChi-Square test p-value: {p_value:.2f}\n"
            )


def tune_hyperparameters(
    preprocessor: ColumnTransformer,
    param_grid: dict,
    model_class: type,
    customers: pd.DataFrame,
) -> dict:
    """
    Tunes the hyperparameters of a given model, by guaranteeing that duplicate rows do not appear in both training and test sets.

    Parameters:
    preprocessor (ColumnTransformer): A preprocessor object to transform the data.
    param_grid (dict): A dictionary containing the hyperparameters and their possible values.
    model_class (type): The class of the model to be tuned.
    customers (pd.DataFrame): The input features.
    y (pd.Series): The target variable.

    Returns:
    dict: A dictionary containing the best hyperparameters.
    """
    X = customers.drop(columns="TravelInsurance")
    y = customers["TravelInsurance"]
    X.reset_index(drop=True, inplace=True)
    y.reset_index(drop=True, inplace=True)
    label_encoder = LabelEncoder()
    y = pd.Series(label_encoder.fit_transform(y), index=y.index)
    param_combinations = list(product(*param_grid.values()))
    param_names = list(param_grid.keys())
    kf = KFold(n_splits=5, shuffle=True, random_state=5)
    best_params = None
    best_score = -np.inf
    for params in param_combinations:
        scores = []
        unique_customers = customers.drop_duplicates().reset_index(drop=True)
        for train_index, test_index in kf.split(unique_customers):
            train_customers = unique_customers.iloc[train_index]
            test_customers = unique_customers.iloc[test_index]
            X_train = X[X.index.isin(train_customers.index)]
            y_train = y[X_train.index]
            X_test = X[X.index.isin(test_customers.index)]
            y_test = y[X_test.index]
            model_params = {param_names[i]: params[i] for i in range(len(param_names))}
            if "random_state" in model_class().get_params().keys():
                model_params["random_state"] = 5
            model = model_class(**model_params)
            pipeline = Pipeline(
                steps=[("preprocessor", preprocessor), ("classifier", model)]
            )
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            score = f1_score(y_test, y_pred, pos_label=1)
            scores.append(score)
        average_score = np.mean(scores)
        if average_score > best_score:
            best_score = average_score
            best_params = params
    best_params_dict = {param_names[i]: best_params[i] for i in range(len(param_names))}
    print("Best parameters found:", best_params_dict)
    print("Best cross-validated F1 for Yes category:", round(best_score, 2))
    return best_params_dict


def tune_threshold(
    preprocessor: ColumnTransformer,
    best_params: dict,
    model_class: type,
    X: pd.DataFrame,
    y: pd.Series,
) -> float:
    """
    Tunes the decision threshold of a given model, by guaranteeing that duplicate rows do not appear in both training and test sets.

    Parameters:
    preprocessor (ColumnTransformer): A preprocessor object to transform the data.
    best_params (dict): A dictionary containing the best hyperparameters found for the model.
    model_class (type): The class of the model to be tuned.
    X (pd.DataFrame): The input features.
    y (pd.Series): The target variable.

    Returns:
    float: The best threshold for F1 score.
    """
    kf = KFold(n_splits=5, shuffle=True, random_state=5)
    y_probs_all = np.zeros_like(y, dtype=float)
    y_true_all = np.zeros_like(y, dtype=int)
    model_params = best_params.copy()
    if "random_state" in model_class().get_params().keys():
        model_params["random_state"] = 5
    model = model_class(**model_params)
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", model)])
    for train_index, test_index in kf.split(X):
        X_temp_train, X_temp_test = X.iloc[train_index], X.iloc[test_index]
        y_temp_train, y_temp_test = y.iloc[train_index], y.iloc[test_index]
        X_temp_train_unique = X_temp_train.drop_duplicates()
        y_temp_train_unique = y_temp_train.loc[X_temp_train_unique.index]
        pipeline.fit(X_temp_train_unique, y_temp_train_unique)
        y_probs_all[test_index] = pipeline.predict_proba(X_temp_test)[:, 1]
        y_true_all[test_index] = y_temp_test
    fpr, tpr, thresholds = roc_curve(y_true_all, y_probs_all)
    roc_auc = roc_auc_score(y_true_all, y_probs_all)
    plt.figure(figsize=(6, 4))
    plt.plot(
        fpr,
        tpr,
        lw=2,
        color="#DD8452",
        label=f"ROC curve (AUC = {roc_auc:.2f})",
        zorder=2,
    )
    plt.plot([0, 1], [0, 1], color="#4C72B0", lw=2, linestyle="--", zorder=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve")
    plt.legend(loc="lower right")
    plt.tick_params(axis="both", labelsize=9, length=0)
    plt.grid(alpha=0.2, axis="y", zorder=0)
    sns.despine(left=True, bottom=True)
    plt.show()
    precision, recall, thresholds_pr = precision_recall_curve(y_true_all, y_probs_all)
    f1_scores = 2 * (precision * recall) / (precision + recall)
    best_threshold_index = np.argmax(f1_scores)
    best_threshold = thresholds_pr[best_threshold_index]
    y_pred_best_threshold = (y_probs_all >= best_threshold).astype(int)
    best_f1_score = f1_score(y_true_all, y_pred_best_threshold)
    print(f"Best Threshold for F1 Score: {best_threshold:.2f}")
    print(f"F1 Score at Best Threshold: {best_f1_score:.2f}")
    return best_threshold


def plot_confusion(
    y_test: pd.Series,
    y_test_pred: np.ndarray,
) -> None:
    """
    Plots the confusion matrices for the training and test datasets.

    Args:
        y_test (pd.Series): Actual test data.
        y_test_pred (np.ndarray): Predicted values for testing set.

    Returns:
        None
    """
    conf_matrix_test = confusion_matrix(y_test, y_test_pred)
    plt.figure(figsize=(4, 3))
    sns.heatmap(
        conf_matrix_test,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        linewidths=0.7,
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")


def evaluate_model_with_threshold(
    pipeline: Pipeline, threshold: float, X: pd.DataFrame, y: pd.Series
):
    """
    Evaluates a model with a given decision threshold, by guaranteeing that duplicate rows do not appear in both training and test sets.

    Parameters:
    pipeline (Pipeline): A scikit-learn pipeline object containing the preprocessing and model.
    threshold (float): The decision threshold for classifying positive cases.
    X (pd.DataFrame): The input features.
    y (pd.Series): The target variable.

    Returns:
    None
    """
    y_true_all = np.array([])
    y_pred_all = np.array([])
    kf = KFold(n_splits=5, shuffle=True, random_state=5)
    for train_index, test_index in kf.split(X):
        X_temp_train, X_temp_test = X.iloc[train_index], X.iloc[test_index]
        y_temp_train, y_temp_test = y.iloc[train_index], y.iloc[test_index]
        X_temp_train_unique = X_temp_train.drop_duplicates()
        y_temp_train_unique = y_temp_train.loc[X_temp_train_unique.index]
        pipeline.fit(X_temp_train_unique, y_temp_train_unique)
        y_probs_temp = pipeline.predict_proba(X_temp_test)[:, 1]
        y_pred_temp = (y_probs_temp >= threshold).astype(int)
        y_true_all = np.concatenate([y_true_all, y_temp_test])
        y_pred_all = np.concatenate([y_pred_all, y_pred_temp])
    print(f"Classification Report (Threshold = {threshold:.2f}):")
    print(classification_report(y_true_all, y_pred_all))
    plot_confusion(y_true_all, y_pred_all)
