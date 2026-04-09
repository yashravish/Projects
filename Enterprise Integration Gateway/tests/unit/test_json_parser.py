"""
Unit tests for JSON parsing utilities.
"""
import pytest

from app.utils.json_parser import extract_list, get_nested, safe_parse_json


class TestSafeParseJson:
    def test_valid_json_object(self):
        result = safe_parse_json('{"key": "value"}')
        assert result == {"key": "value"}

    def test_valid_json_array(self):
        result = safe_parse_json('[1, 2, 3]')
        assert result == [1, 2, 3]

    def test_invalid_json_returns_none(self):
        result = safe_parse_json("{not valid json}")
        assert result is None

    def test_empty_string_returns_none(self):
        result = safe_parse_json("")
        assert result is None

    def test_bytes_input(self):
        result = safe_parse_json(b'{"status": "ok"}')
        assert result == {"status": "ok"}


class TestExtractList:
    def test_dict_with_key(self):
        payload = {"customers": [{"id": 1}, {"id": 2}]}
        result = extract_list(payload, "customers")
        assert len(result) == 2

    def test_list_input_returned_as_is(self):
        data = [{"id": 1}]
        result = extract_list(data, "anything")
        assert result == data

    def test_missing_key_returns_empty(self):
        result = extract_list({"other": []}, "customers")
        assert result == []

    def test_non_list_value_returns_empty(self):
        result = extract_list({"customers": "not a list"}, "customers")
        assert result == []


class TestGetNested:
    def test_single_level(self):
        assert get_nested({"a": 1}, "a") == 1

    def test_nested_two_levels(self):
        data = {"address": {"city": "New York"}}
        assert get_nested(data, "address", "city") == "New York"

    def test_missing_key_returns_default(self):
        assert get_nested({"a": 1}, "b") is None
        assert get_nested({"a": 1}, "b", default="N/A") == "N/A"

    def test_intermediate_none_returns_default(self):
        data = {"address": None}
        assert get_nested(data, "address", "city") is None

    def test_empty_dict(self):
        assert get_nested({}, "any") is None
