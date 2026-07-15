# -----------------------------------------------------------------------------
# Functions for parsing args
# -----------------------------------------------------------------------------
import copy
import os
from ast import literal_eval
from collections.abc import Mapping
from pathlib import Path

import yaml


class CfgNode(dict):
    """
    CfgNode represents an internal node in the configuration tree. It's a simple
    dict-like container that allows for attribute-based access to keys.
    """
    def __init__(self, init_dict=None, key_list=None, new_allowed=False):
        # Recursively convert nested dictionaries in init_dict into CfgNodes
        init_dict = {} if init_dict is None else init_dict
        key_list = [] if key_list is None else key_list
        for k, v in init_dict.items():
            if type(v) is dict:
                # Convert dict to CfgNode
                init_dict[k] = CfgNode(v, key_list=key_list + [k])
        super(CfgNode, self).__init__(init_dict)

    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        self[name] = value

    def __str__(self):
        def _indent(s_, num_spaces):
            s = s_.split("\n")
            if len(s) == 1:
                return s_
            first = s.pop(0)
            s = [(num_spaces * " ") + line for line in s]
            s = "\n".join(s)
            s = first + "\n" + s
            return s

        r = ""
        s = []
        for k, v in sorted(self.items()):
            seperator = "\n" if isinstance(v, CfgNode) else " "
            attr_str = "{}:{}{}".format(str(k), seperator, str(v))
            attr_str = _indent(attr_str, 2)
            s.append(attr_str)
        r += "\n".join(s)
        return r

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__,
                               super(CfgNode, self).__repr__())

    @property
    def sections(self):
        """Hierarchical config before the legacy flat compatibility view."""
        return self.__dict__.get("_sections", CfgNode())

    def to_dict(self):
        def convert(value):
            if isinstance(value, Mapping):
                return {key: convert(item) for key, item in value.items()}
            if isinstance(value, list):
                return [convert(item) for item in value]
            return copy.deepcopy(value)

        return convert(self)


def _deep_merge(base, override):
    """Recursively merge mappings with MMEngine-style ``_delete_`` support."""
    if not isinstance(base, Mapping) or not isinstance(override, Mapping):
        return copy.deepcopy(override)
    if bool(override.get("_delete_", False)):
        return {
            key: copy.deepcopy(value)
            for key, value in override.items()
            if key != "_delete_"
        }
    merged = copy.deepcopy(dict(base))
    for key, value in override.items():
        if key == "_delete_":
            continue
        if key in merged and isinstance(merged[key], Mapping) and isinstance(value, Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _load_with_bases(path, stack=()):
    path = Path(path).expanduser().resolve()
    if path in stack:
        chain = " -> ".join(str(item) for item in (*stack, path))
        raise ValueError(f"Circular config inheritance detected: {chain}")
    if not path.is_file() or path.suffix.lower() not in {".yaml", ".yml"}:
        raise FileNotFoundError(f"Config YAML not found: {path}")
    with path.open("r", encoding="utf-8-sig") as stream:
        current = yaml.safe_load(stream) or {}
    if not isinstance(current, Mapping):
        raise TypeError(f"Top-level config must be a mapping: {path}")
    bases = current.pop("_base_", [])
    if isinstance(bases, (str, os.PathLike)):
        bases = [bases]
    if not isinstance(bases, (list, tuple)):
        raise TypeError(f"_base_ must be a path or list of paths: {path}")
    resolved = {}
    for base in bases:
        base_path = Path(base)
        if not base_path.is_absolute():
            base_path = path.parent / base_path
        resolved = _deep_merge(resolved, _load_with_bases(base_path, (*stack, path)))
    return _deep_merge(resolved, current)


def _flatten_sections(sections):
    """Expose historical ``cfg.key`` access without losing section structure."""
    flat = {}
    legacy_sections = {"DATA", "MODEL", "TRAIN", "RUNTIME", "Distributed", "TEST"}
    for section_name, values in sections.items():
        if section_name not in legacy_sections or not isinstance(values, Mapping):
            flat[section_name] = copy.deepcopy(values)
            continue
        for key, value in values.items():
            if key in flat and flat[key] != value:
                raise KeyError(
                    f"Duplicate flattened config key {key!r} has conflicting "
                    f"values; found again in section {section_name!r}"
                )
            flat[key] = copy.deepcopy(value)
    return flat


def load_cfg_from_cfg_file(file):
    sections = _load_with_bases(file)
    cfg = CfgNode(_flatten_sections(sections))
    cfg.__dict__["_sections"] = CfgNode(copy.deepcopy(sections))
    cfg.__dict__["filename"] = str(Path(file).expanduser().resolve())
    return cfg


def merge_cfg_from_list(cfg, cfg_list):
    new_cfg = copy.deepcopy(cfg)
    assert len(cfg_list) % 2 == 0
    for full_key, v in zip(cfg_list[0::2], cfg_list[1::2]):
        path = full_key.split(".")
        sections = new_cfg.sections
        if len(path) > 1 and path[0] in sections:
            cursor = sections[path[0]]
            for part in path[1:-1]:
                if part not in cursor or not isinstance(cursor[part], Mapping):
                    raise KeyError(f"Non-existent config path: {full_key}")
                cursor = cursor[part]
            if path[-1] not in cursor:
                raise KeyError(f"Non-existent config path: {full_key}")
            original = cursor[path[-1]]
            value = _check_and_coerce_cfg_value_type(
                _decode_cfg_value(v), original, path[-1], full_key
            )
            cursor[path[-1]] = value
            section_name = path[0]
            if section_name in {
                "DATA", "MODEL", "TRAIN", "RUNTIME", "Distributed", "TEST"
            }:
                root_key = path[1]
                setattr(new_cfg, root_key, copy.deepcopy(sections[section_name][root_key]))
        else:
            subkey = path[-1]
            if subkey not in new_cfg:
                raise KeyError(f"Non-existent key: {full_key}")
            value = _check_and_coerce_cfg_value_type(
                _decode_cfg_value(v), new_cfg[subkey], subkey, full_key
            )
            setattr(new_cfg, subkey, value)
            for values in sections.values():
                if isinstance(values, Mapping) and subkey in values:
                    values[subkey] = value
                    break

    return new_cfg


def _decode_cfg_value(v):
    """Decodes a raw config value (e.g., from a yaml config files or command
    line argument) into a Python object.
    """
    # All remaining processing is only applied to strings
    if not isinstance(v, str):
        return v
    # Try to interpret `v` as a:
    #   string, number, tuple, list, dict, boolean, or None
    try:
        v = literal_eval(v)
    # The following two excepts allow v to pass through when it represents a
    # string.
    #
    # Longer explanation:
    # The type of v is always a string (before calling literal_eval), but
    # sometimes it *represents* a string and other times a data structure, like
    # a list. In the case that v represents a string, what we got back from the
    # yaml parser is 'foo' *without quotes* (so, not '"foo"'). literal_eval is
    # ok with '"foo"', but will raise a ValueError if given 'foo'. In other
    # cases, like paths (v = 'foo/bar' and not v = '"foo/bar"'), literal_eval
    # will raise a SyntaxError.
    except ValueError:
        pass
    except SyntaxError:
        pass
    return v


def _check_and_coerce_cfg_value_type(replacement, original, key, full_key):
    """Checks that `replacement`, which is intended to replace `original` is of
    the right type. The type is correct if it matches exactly or is one of a few
    cases in which the type can be easily coerced.
    """
    original_type = type(original)
    replacement_type = type(replacement)

    if original is None:
        return replacement
    if isinstance(original, Mapping) and isinstance(replacement, Mapping):
        return CfgNode(copy.deepcopy(dict(replacement)))

    # The types must match (with some exceptions)
    if replacement_type == original_type:
        return replacement

    # Cast replacement from from_type to to_type if the replacement and original
    # types match from_type and to_type
    def conditional_cast(from_type, to_type):
        if replacement_type == from_type and original_type == to_type:
            return True, to_type(replacement)
        else:
            return False, None

    # Conditionally casts
    # list <-> tuple
    casts = [(tuple, list), (list, tuple)]
    # For py2: allow converting from str (bytes) to a unicode string
    try:
        casts.append((str, unicode))  # noqa: F821
    except Exception:
        pass

    for (from_type, to_type) in casts:
        converted, converted_value = conditional_cast(from_type, to_type)
        if converted:
            return converted_value

    raise ValueError(
        "Type mismatch ({} vs. {}) with values ({} vs. {}) for config "
        "key: {}".format(original_type, replacement_type, original,
                         replacement, full_key))
