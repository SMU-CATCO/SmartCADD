import pytest
from pipeline import Pipeline
from filters import SimpleFilter  # Assuming SimpleFilter is as defined above


@pytest.fixture
def pipeline():
    return Pipeline(filters=[SimpleFilter()])


def test_pipeline_processing(pipeline, compounds):
    processed = pipeline(compounds)
    assert (
        len(processed) == 2
    ), "Pipeline should process and filter compounds accordingly"
