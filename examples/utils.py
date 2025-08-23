import pandas as pd
import dspy
from sklearn.model_selection import train_test_split


def load_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    return df

def generate_training_examples(df: pd.DataFrame) -> (tuple[list[dspy.Example], list[dspy.Example]]):
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

def validate_result(example, predicted_example, trace=None):
    return example.is_safe == predicted_example.is_safe
