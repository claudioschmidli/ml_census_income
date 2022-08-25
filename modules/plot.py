import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def missing_values(input_data: pd.DataFrame, features: list, title_prefix: str):
    """Plot a pie chart showing the number of rows=' ?' of the given features.

    Args:
        input_data (pd.DataFrame): Dataframe containg the data of interest.
        features (list): Feature names containing of features with missing data
        title_prefix (str): Prefix for the title of the subplots
    """
    plt.figure(facecolor="white", figsize=(10, 30))
    plt.subplots_adjust(
        left=None, bottom=None, right=None, top=None, wspace=0.65, hspace=0
    )
    for i, feature in enumerate(features, 1):
        plt.subplot(1, len(features), i)
        labels = "missing", "complete"
        length = len(input_data)
        missing = len(input_data[input_data[feature] == " ?"])
        fracs = missing, length - missing
        plt.pie(fracs, labels=labels, autopct="%1.1f%%", shadow=True, explode=(0, 0.3))
        plt.title(f"{title_prefix} {feature}")
    plt.show()


def value_occurence(train, test, value, columns):
    counts = []
    for column in columns:
        current_counts = len(train.loc[train[column] == value])
        # print(f'{column}:\t{current_counts}')
        counts.append(current_counts)
    y_pos = np.arange(len(counts))

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.tight_layout()
    ax1.bar(y_pos, counts)
    ax1.set_xticks(y_pos, columns)
    ax1.set_title("Train set")

    counts = []
    for column in columns:
        current_counts = len(test.loc[test[column] == value])
        # print(f'{column}:\t{current_counts}')
        counts.append(current_counts)
    y_pos = np.arange(len(counts))

    ax2.bar(y_pos, counts)
    ax2.set_xticks(y_pos, columns)
    ax2.set_title("Test set")


def autopct_more_than_1(pct):
    return ("%1.f%%" % pct) if pct >= 3 else ""


def column_pie_chart(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    column: str,
    explode: None = False,
):
    plt.figure(facecolor="white", figsize=(7, 20))
    plt.subplots_adjust(
        left=None, bottom=None, right=None, top=None, wspace=0.45, hspace=0
    )
    plt.subplot(1, 2, 1)
    labels = train_data[column].value_counts().index.tolist()
    fracs = train_data[column].value_counts()
    for i, fract in enumerate(fracs):
        if fract / len(train_data) * 100 < 3:
            labels[i] = ""
    if explode:
        plt.pie(
            fracs,
            labels=labels,
            autopct=autopct_more_than_1,
            shadow=True,
            explode=(0, 0.1),
        )
    else:
        plt.pie(fracs, labels=labels, autopct=autopct_more_than_1, shadow=True)
    plt.title("Training set")
    plt.subplot(1, 2, 2)
    labels = test_data[column].value_counts().index.tolist()
    fracs = test_data[column].value_counts()
    for i, fract in enumerate(fracs):
        if fract / len(test_data) * 100 < 3:
            labels[i] = ""
    if explode:
        plt.pie(
            fracs,
            labels=labels,
            autopct=autopct_more_than_1,
            shadow=True,
            explode=(0, 0.1),
        )
    else:
        plt.pie(fracs, labels=labels, autopct=autopct_more_than_1, shadow=True)
    plt.title("Test set")
    plt.show


def column_hist(train_data: pd.DataFrame, test_data: pd.DataFrame, column: str):
    plt.figure(facecolor="white")
    fig, axes = plt.subplots(1, 2, figsize=(16, 4))
    train_plot = (
        train_data[column]
        .hist(ax=axes[0], bins=120, grid=False)
        .set_title(f"{column} (training set)")
    )
    test_plot = (
        test_data[column]
        .hist(ax=axes[1], bins=120, grid=False)
        .set_title(f"{column} (test set)")
    )
    plt.show


def countplot(train_data: pd.DataFrame, test_data: pd.DataFrame, column: str):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    sns.countplot(x=train_data[column], ax=axes[0], label="Count")
    axes[0].set_title("Training set")

    sns.countplot(x=test_data[column], ax=axes[1], label="Count")
    axes[1].set_title("Training set")
    plt.show()


def age_group(x):
    x = int(x)
    x = abs(x)
    if 18 < x < 31:
        return "19-30"
    if 30 < x < 41:
        return "31-40"
    if 40 < x < 51:
        return "41-50"
    if 50 < x < 61:
        return "51-60"
    if 60 < x < 71:
        return "61-70"
    else:
        return "Greater than 70"


def income_by_age_group(data):
    data["age_group"] = data["age"].apply(age_group)
    plt.figure(figsize=(12, 6))
    order_list = ["19-30", "31-40", "41-50", "51-60", "61-70", "Greater than 70"]
    sns.countplot(x=data["age_group"], hue=data["income"], order=order_list)
    plt.title(
        "Income of Individuals of Different Age Groups", fontsize=18, fontweight="bold"
    )
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=16)
    data.drop("age_group", axis=1, inplace=True)


def income_by_working_class(data):
    plt.figure(figsize=(12, 6))
    sns.countplot(x=data["workclass"], hue=data["income"])
    plt.title(
        "Income of Individuals of Different Working CLasses",
        fontsize=18,
        fontweight="bold",
    )
    plt.xticks(fontsize=16, rotation=90)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=16)


def income_by_education(data):
    data["education"] = data["education"].str.strip(" .")
    plt.figure(figsize=(15, 6))
    order_list = [
        "Preschool",
        "1st-4th",
        "5th-6th",
        "7th-8th",
        "9th",
        "10th",
        "11th",
        "12th",
        "HS-grad ",
        "Some-college",
        "Bachelors",
        "Masters",
        "Doctorate",
        "Prof-school",
        "Assoc-acdm",
        "Assoc-voc",
    ]
    sns.countplot(x=data["education"], hue=data["income"], order=order_list)
    plt.title(
        "Income of Individuals of Different Education Levels",
        fontsize=18,
        fontweight="bold",
    )
    plt.xticks(fontsize=16, rotation=90)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=16)


def income_by_marital_status(data):
    plt.figure(figsize=(12, 6))
    sns.countplot(x=data["marital"], hue=data["income"])
    plt.title(
        "Income of Individuals of Different Marital Status",
        fontsize=18,
        fontweight="bold",
    )
    plt.xticks(fontsize=16, rotation=90)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=16)


def income_by_occupation(data):
    plt.figure(figsize=(18, 6))
    sns.countplot(x=data["occupation"], hue=data["income"])
    plt.title(
        "Income of Individuals of Different Occupations", fontsize=18, fontweight="bold"
    )
    plt.xticks(fontsize=16, rotation=90)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=16)


def income_by_relationship(data):
    plt.figure(figsize=(12, 6))
    sns.countplot(x=data["relationship"], hue=data["income"])
    plt.title(
        "Income of Individuals of Different Relationship",
        fontsize=18,
        fontweight="bold",
    )
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=16)


def income_by_race(data):
    plt.figure(figsize=(12, 6))
    sns.countplot(x=data["race"], hue=data["income"])
    plt.title(
        "Income of Individuals of Different Races", fontsize=18, fontweight="bold"
    )
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=16)


def income_by_gender(data):
    plt.figure(figsize=(12, 6))
    sns.countplot(x=data["gender"], hue=data["income"])
    plt.title(
        "Income of Individuals of Different Genders", fontsize=18, fontweight="bold"
    )
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=16)


def feature_correlation(data):
    plt.figure(figsize=(12, 10))
    plt.title(
        "Correlation between different features of the dataset",
        fontsize=18,
        fontweight="bold",
    )
    sns.heatmap(data=data.corr(), annot=True)
    plt.xticks(fontsize=12, rotation=90)
    plt.yticks(fontsize=12, rotation=0)
    plt.legend(fontsize=12)
