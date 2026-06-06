import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ===========================
# STEP 1 LOAD DATASET
# ===========================

df = pd.read_csv("creditcard.csv")

print("\nDataset Loaded Successfully")

print("\nFirst 5 Rows:\n")

print(df.head())

print("\nClass Distribution:\n")

print(df["Class"].value_counts())


# ===========================
# STEP 2 PREPROCESS
# ===========================

X = df.drop("Class", axis=1)

y = df["Class"]

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


# ===========================
# STEP 3 TRAIN TEST SPLIT
# ===========================

X_train, X_test, y_train, y_test = train_test_split(

    X_scaled,

    y,

    test_size=0.2,

    random_state=42,

    stratify=y

)


# ===========================
# STEP 4 MODELS
# ===========================

models = {

"Random Forest":

RandomForestClassifier(

n_estimators=100,

random_state=42,

n_jobs=-1

),

"XGBoost":

XGBClassifier(

eval_metric="logloss",

random_state=42,

n_estimators=100

)

}


# ===========================
# STEP 5 TRAIN
# ===========================

for name, model in models.items():

    print("\n")

    print("="*50)

    print(name)

    print("="*50)

    model.fit(

        X_train,

        y_train

    )

    y_pred = model.predict(

        X_test

    )

    accuracy = accuracy_score(

        y_test,

        y_pred

    )

    print(

        f"\nAccuracy : {accuracy*100:.2f}%"

    )

    print(

        "\nClassification Report:\n"

    )

    print(

        classification_report(

            y_test,

            y_pred

        )

    )

    cm = confusion_matrix(

        y_test,

        y_pred

    )

    plt.figure(

        figsize=(5,4)

    )

    sns.heatmap(

        cm,

        annot=True,

        fmt="d",

        cmap="Blues"

    )

    plt.title(

        f"{name} Confusion Matrix"

    )

    plt.xlabel(

        "Predicted"

    )

    plt.ylabel(

        "Actual"

    )

    plt.savefig(

        f"{name}_confusion_matrix.png"

    )

    plt.close()


# ===========================
# STEP 6 VISUALIZATION
# ===========================

plt.figure(

figsize=(6,4)

)

sns.countplot(

x="Class",

data=df,

palette="coolwarm"

)

plt.title(

"Fraud vs Non Fraud Transactions"

)

plt.savefig(

"fraud_distribution.png"

)

plt.close()


print("\nSaved Images:")

print("fraud_distribution.png")

print("Random Forest_confusion_matrix.png")

print("XGBoost_confusion_matrix.png")

print("\nProject Completed Successfully")
