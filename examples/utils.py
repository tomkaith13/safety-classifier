import pandas as pd
import dspy
from sklearn.model_selection import train_test_split
import os


def load_data(file_path: str) -> pd.DataFrame:
    _, ext = os.path.splitext(file_path)
    print(f"Loading data from {file_path} with extension {ext}")
    if ext == ".csv":
        df = pd.read_csv(file_path)
        return df
    elif ext == ".json":
        df = pd.read_json(file_path)
        return df
    
    return None

def create_training_and_test_examples(df: pd.DataFrame) -> (tuple[list[dspy.Example], list[dspy.Example]]):
    # Implement your example generation logic here

    train_df, test_df = train_test_split(df, shuffle=True, random_state=42, test_size=0.8)

    examples = []

    for _, row in train_df.iterrows():
        example = dspy.Example(
            user_query=row['text'],
            is_safe=True if row['Safe'] == 1 else False
        ).with_inputs("user_query")
        examples.append(example)

    test_examples = []
    for _, row in test_df.iterrows():
        example = dspy.Example(
            user_query=row['text'],
            is_safe=True if row['Safe'] == 1 else False
        ).with_inputs("user_query")
        test_examples.append(example)

    return examples, test_examples

def generate_full_evaluation_set(df: pd.DataFrame) -> list[dspy.Example]:
    examples = []

    for _, row in df.iterrows():
        example = dspy.Example(
            user_query=row['text'],
            is_safe=True if row['Safe'] == 1 else False
        ).with_inputs("user_query")
        examples.append(example)

    return examples

def validate_result(example, predicted_example, trace=None):
    return example.is_safe == predicted_example.is_safe

def transform_aegis_json_to_jsonl(filepath: str):

    aegis_df = load_data(filepath)
    aegis_df = aegis_df[["prompt", "prompt_label"]]

    # Remove all rows where prompt is 'REDACTED'
    aegis_df = aegis_df[aegis_df["prompt"] != "REDACTED"]
    
    filepath, _ = os.path.splitext(filepath)
    aegis_df.to_json(f"{filepath}.jsonl", orient="records", lines=True)
    print(aegis_df.head())