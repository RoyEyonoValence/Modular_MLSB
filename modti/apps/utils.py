import os
import re
import ast
import yaml
import fsspec
import collections
from typing import List

def load_hp(conf_path):
    with fsspec.open(conf_path) as IN:
        config = yaml.load(IN, Loader=get_config_loader(os.path.dirname(conf_path)))
    return config


def get_config_loader(config_path: str):
    """Add constructors and loader for custom !load_file tag.

    Makes it possible to defines YAML files hierarchically"""

    def _load_file_constructor(
            loader: yaml.SafeLoader, node: yaml.nodes.ScalarNode
    ):
        """Constructor for !load_file tag"""
        with fsspec.open(os.path.join(config_path, loader.construct_scalar(node))) as IN:
            node_content = yaml.safe_load(IN)
        return node_content

    loader = yaml.SafeLoader
    loader.add_constructor("!load_file", _load_file_constructor)
    loader.add_implicit_resolver(
        "tag:yaml.org,2002:float",
        re.compile(
            """^(?:
        [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
        |[-+]?\\.(?:inf|Inf|INF)
        |\\.(?:nan|NaN|NAN))$""",
            re.X,
        ),
        list("-+0123456789."),
    )
    return loader


def nested_dict_update(original, overrides):
    """Updates nested keys without overriding the entire dictionary"""
    ret = original.copy()
    for k, v in overrides.items():
        if isinstance(v, collections.abc.Mapping):
            ret[k] = nested_dict_update(ret.get(k, {}), v)
        else:
            ret[k] = v
    return ret


def parse_overrides(overrides: List[str]):
    """Parses the `a.b.c=z` syntax in a hierarchical dict"""

    parse = {}
    for override in overrides:
        split = override.split("=")
        key = split[0]
        value = "=".join(split[1:])
        keys = key.split(".")

        data = parse
        for k in keys[:-1]:
            if k not in data:
                data[k] = {}
            data = data[k]

        try:
            value = ast.literal_eval(value)
        except ValueError:
            pass
        except SyntaxError:
            pass

        data[keys[-1]] = value
    return parse
