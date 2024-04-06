import pytest

@pytest.fixture
def example_data():
    return [1, 2, 3, 4, 5, 6]

def test_sum_of_list(example_data):
    assert sum(example_data) == 21