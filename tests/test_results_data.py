import os
import json
import pytest
from datetime import datetime

from helpers import results_data, utils

KEY1_VALUE = "d90"
TEST_PATTERN = ["{key1}", "{run}", "file.json"]
TEST_RESULTS = {"summary": "Original results", "stats": {"value1": 1, "value2": 2}}

utils.setup_logging(level="ERROR")


def test_path_formation(tmp_path):
    cache = results_data.ResultCache(TEST_PATTERN, tmp_path)
    assert cache._get_wildcard_pattern() == os.path.join("*", "*", "file.json")

    cache.set(key1=KEY1_VALUE)
    assert cache._get_wildcard_pattern() == os.path.join(KEY1_VALUE, "*", "file.json")

    with pytest.raises(ValueError):
        cache.filename()

    cache.set(run=3)
    assert cache.filename() == os.path.join(tmp_path, KEY1_VALUE, "3", "file.json")
    assert not cache.exists()


def test_save(tmp_path):
    cache = results_data.ResultCache(TEST_PATTERN, tmp_path, key1=KEY1_VALUE)
    cache.save(TEST_RESULTS, run=0)

    assert os.path.isfile(os.path.join(tmp_path, KEY1_VALUE, "0", "file.json"))


def test_find(tmp_path):
    cache = results_data.ResultCache(TEST_PATTERN, tmp_path, key1=KEY1_VALUE)
    cache.save(TEST_RESULTS, run=0)
    cache.save(TEST_RESULTS, run=1)

    assert os.path.isfile(os.path.join(tmp_path, KEY1_VALUE, "0", "file.json"))
    assert os.path.isfile(os.path.join(tmp_path, KEY1_VALUE, "1", "file.json"))
    assert cache._get_wildcard_pattern() == os.path.join(KEY1_VALUE, "*", "file.json")
    assert len(cache.find()) == 2


def test_load(tmp_path):
    cache = results_data.ResultCache(TEST_PATTERN, tmp_path, key1=KEY1_VALUE)
    cache.save(TEST_RESULTS, run=0)
    cache.save(TEST_RESULTS, run=1)

    with pytest.raises(ValueError):
        cache.load()

    r1 = cache.load(run=0)

    assert r1["stats"]["value1"] == TEST_RESULTS["stats"]["value1"]

    ra = cache.load_all()
    assert len(ra) == 2
    assert all(x in ra.keys() for x in "01")


def test_overwrite(tmp_path):
    cache = results_data.ResultCache(TEST_PATTERN, tmp_path, key1=KEY1_VALUE, run=4)
    cache.save(TEST_RESULTS)
    assert os.path.isfile(os.path.join(tmp_path, KEY1_VALUE, "4", "file.json"))

    # Test the overwrite mode
    assert results_data.get_overwrite_mode() == "overwrite"

    new_results = {**TEST_RESULTS}
    new_results["summary"] = "Updated results"
    new_results["stats"]["value2"] = -1

    cache.save(new_results)
    r1 = cache.load()
    assert r1["stats"]["value2"] == -1, "Error in value saved in the `overwrite` mode."

    # Test the exception mode
    results_data.set_overwrite_mode("exception")
    new_results["summary"] = "Results that should not have been saved!"
    new_results["stats"]["value2"] = -2

    with pytest.raises(FileExistsError):
        cache.save(new_results)

    r1 = cache.load()
    assert r1["stats"]["value2"] == -1, "Error in value saved in the `exception` mode."

    # Test the backup mode
    results_data.set_overwrite_mode("backup")
    new_results["summary"] = "Final results!"
    new_results["stats"]["value2"] = -3
    backup_filename = f"file.json.{datetime.now():%Y%m%d%H%M}"

    cache.save(new_results)

    r1 = cache.load()
    assert r1["stats"]["value2"] == -3, "Error in value saved in the `backup` mode."
    assert os.path.isfile(os.path.join(tmp_path, KEY1_VALUE, "4", "file.json"))
    assert os.path.isfile(os.path.join(tmp_path, KEY1_VALUE, "4", backup_filename))

    with open(os.path.join(tmp_path, KEY1_VALUE, "4", backup_filename)) as f:
        rb = json.load(f)

    assert rb["stats"]["value2"] == -1, "Error in value recovered from the backup."
