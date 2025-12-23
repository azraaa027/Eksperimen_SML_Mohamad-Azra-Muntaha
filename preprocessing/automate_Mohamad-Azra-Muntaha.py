import pandas as pd

def load_and_preprocess(input_path: str, output_path: str):
    df = pd.read_csv(input_path)
    
    df["Age"] = df.groupby(["Pclass", "Sex"])["Age"].transform(lambda x: x.fillna(x.median()))
    df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)
    df = df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])
    df = pd.get_dummies(df, columns=["Sex", "Embarked"], drop_first=True)
    
    df.to_csv(output_path, index=False)
    print(f"✅ Preprocessed data saved to {output_path}")

if __name__ == "__main__":
    load_and_preprocess(
        input_path="../titanic_raw/train.csv",
        output_path="titanic_preprocessed.csv"
    )