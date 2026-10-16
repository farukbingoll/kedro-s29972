from kedro.pipeline import Pipeline, node, pipeline
from .nodes import validate_data, clean_data, split_data, train_model, evaluate_model


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(validate_data, ["iris_raw", "params:features", "params:target"],
             "iris_validated", name="validate_raw_data"),
        node(clean_data, ["iris_validated", "params:features", "params:target"],
             "iris_clean", name="clean_iris_data"),
        node(split_data, ["iris_clean", "params:features", "params:target",
                          "params:test_size", "params:random_seed"],
             ["x_train", "x_test", "y_train", "y_test"], name="split_iris_data"),
        node(train_model, ["x_train", "y_train", "params:max_depth", "params:random_seed"],
             "iris_model", name="train_decision_tree"),
        node(evaluate_model, ["iris_model", "x_test", "y_test"],
             "iris_metrics", name="evaluate_decision_tree")
    ])
