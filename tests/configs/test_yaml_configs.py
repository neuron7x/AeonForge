import os, yaml

def test_default_yaml_exists_and_has_minimum_keys():
    path = os.path.join('configs', 'default.yaml')
    assert os.path.isfile(path)
    data = yaml.safe_load(open(path, 'r', encoding='utf-8'))
    assert isinstance(data, dict)
    assert 'seed' in data and 'cbc' in data

def test_human_loop_yaml_roles():
    path = os.path.join('configs', 'human_loop.yaml')
    assert os.path.isfile(path)
    data = yaml.safe_load(open(path, 'r', encoding='utf-8'))
    assert isinstance(data, dict)
    assert 'agents' in data and isinstance(data['agents'], list)
