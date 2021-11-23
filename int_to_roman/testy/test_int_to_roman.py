from int_to_roman import int_to_Roman
import pytest
def test_int_argument():
    int_to_Roman(1)

def test_only_int_argument():
    with pytest.raises(AttributeError):
        int_to_Roman("123")

def test_return_string():
    assert isinstance(int_to_Roman(999), str)
