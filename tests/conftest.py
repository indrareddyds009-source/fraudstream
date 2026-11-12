import pytest

from fraudstream import pipeline


@pytest.fixture(scope="session")
def trained(tmp_path_factory):
    out = tmp_path_factory.mktemp("artifacts")
    docs = tmp_path_factory.mktemp("docs")
    results = pipeline.run(str(out), str(docs), n_cards=600, seed=3)
    return out, docs, results
