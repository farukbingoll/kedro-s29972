from kedro.pipeline import Pipeline
from iris_kedro.pipelines.classification import create_pipeline


def register_pipelines() -> dict[str, Pipeline]:
    pipeline = create_pipeline()
    return {"__default__": pipeline, "classification": pipeline}
