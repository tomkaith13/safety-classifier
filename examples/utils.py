import pandas as pd
import dspy
from sklearn.model_selection import train_test_split


def load_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    return df

def generate_training_examples(df: pd.DataFrame) -> list[dspy.Example]:
    # Implement your example generation logic here

    train_df, _ = train_test_split(df, shuffle=True, random_state=42)

    examples = []

    for _, row in train_df.iterrows():
        example = dspy.Example(
            user_query=row['text'],
            is_safe=True if row['Safe'] == 1 else False
        )
        examples.append(example)

    return examples

def validate_result(example, predicted_example, trace=None):
    return example.is_safe == predicted_example.is_safe
