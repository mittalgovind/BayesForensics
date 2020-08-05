# -*- coding: utf-8 -*-
"""
General helper functions.

Example functionality:
----------------------
- check if working in an interactive environment (ipython / jupyter notebook)
- configure toolbox-wide logging
- test and print functions for numbers
- printing a concise dict summary - omits tensor values (only shapes are shown)
- fuzzy-match a string to a list of options
- recursively find a key in a dictionary

"""
import os
import re
import subprocess
import sys
from functools import reduce

from tqdm import tqdm

import Levenshtein
import numpy as np

from loguru import logger

_numeric_types = {int, float, bool, np.bool, np.float, np.float16, np.float32, np.float64,
                           np.int, np.int8, np.int32, np.int16, np.int64,
                           np.uint, np.uint8, np.uint32, np.uint16, np.uint64}


def setup_logging(filename=None, long_date=False, level='INFO'):
    """
    Configure the logger to a compact format.
    :param filename: add an additional sink to the given file
    :param long_date: flag to choose a full or compact date format
    """

    if long_date:
        # If needed, can add "<cyan>{name}</cyan>:" to show file which triggered the message
        log_format = '<green>{time:YYYY-MM-DD HH:mm:ss}</> | <lvl>{level}</lvl> | <lvl>{message}</>'
    else:
        log_format = '<green>{time:HH:mm:ss}</> | <lvl>{level}</> | <lvl>{message}</lvl>'

    config = {
        "handlers": [
            {"sink": sys.stderr, "format": log_format, "level": level, "colorize": True}
        ],
    }

    if filename is not None:
        if '/' not in filename:
            filename = os.path.join('logs', filename)
        config['handlers'].append({"sink": filename, "serialize": False, "format": log_format, "level": level})

    logger.configure(**config)


def is_number(value):
    return type(value) in _numeric_types


def is_numeric_type(t):
    return t in _numeric_types


def is_nan(value):
    if value is None:
        return True

    if is_number(value):
        return np.isnan(value)

    return False


def is_vector(data):

    if isinstance(data, list) and all(is_number(x) for x in data):
        return True
    elif isinstance(data, np.ndarray) and (data.ndim == 1 or (data.ndim == 2 and 1 in data.shape)):
        return True
    else:
        return False


def format_sequence_rle(numbers, sep='-'):
    """
    Formats a sequence of numbers (int) using RLE, e.g., (32,32,32,64,64,1) -> '32x3-64x2-1'
    :param numbers: list of numbers
    :param sep: separator
    :return: 
    """
    items = []
    counter = 1
    for i in range(1, len(numbers)):
        if numbers[i - 1] != numbers[i]:
            items.append(f'{numbers[i - 1]}x{counter}' if counter > 1 else f'{numbers[i - 1]}')
            counter = 1
        else:
            counter += 1
        if i == len(numbers) - 1:
            items.append(f'{numbers[i]}x{counter}' if counter > 1 else f'{numbers[i]}')

    return sep.join(items)


def format_number_order(n):
    n = float(n)
    suffix = ('', 'k', 'M', 'B', 'T')
    idx = max(0, min(len(suffix)-1, int(np.floor(0 if n == 0 else np.log10(abs(n))/3))))
    return f'{n / 10**(3 * idx):.0f}{suffix[idx]}'


def format_number(x, digits=3):
    if np.isnan(x):
        return 'nan'

    if np.isinf(x):
        return '∞'

    try:
        if isinstance(x, float) and x != 0:
            w = max(0, int(np.floor(np.log10(np.abs(x))))) + (digits - 1)
            p = max(0, - np.int(np.floor(np.log10(np.abs(x))))) + (digits - 1)
            return f'{x:{w}.{p}f}'
        else:
            return f'{x}'
    except:
        return '?'


def match_option(x, options, regexp=False):

    if regexp:
        matches = [y for y in options if re.match(x, y)]
        if len(matches) == 1:
            return matches[0]
        else:
            raise ValueError(f'No regexp match: "{x}" to any of {options}!')

    else:
        start_match = [y.startswith(x) or x.startswith(y) for y in options]

        if sum(start_match) == 1:
            return options[start_match.index(True)]
        else:
            distances = [Levenshtein.distance(x, y) for y in options]
            return options[distances.index(min(distances))]


def logCall(func):
    """
    Decorator to print function call details - parameters names and effective values
    """
    def wrapper(*func_args, **func_kwargs):
        arg_names = func.__code__.co_varnames[:func.__code__.co_argcount]
        args = func_args[:len(arg_names)]
        defaults = func.__defaults__ or ()
        args = args + defaults[len(defaults) - (func.__code__.co_argcount - len(args)):]
        params = list(zip(arg_names, args))
        args = func_args[len(arg_names):]

        if args:
            params.append(('args', args))

        if func_kwargs:
            params.append(('kwargs', func_kwargs))

        # Print function call
        print('@> ' + func.__name__ + ' (' + ', '.join('%s = %r' % p for p in params) + ' )')

        # Actual function call
        return func(*func_args, **func_kwargs)

    return wrapper


def mockCall(func):
    """
    Decorator to print function call details but skip the actual call.
    """
    def wrapper(*func_args, **func_kwargs):
        arg_names = func.__code__.co_varnames[:func.__code__.co_argcount]
        args = func_args[:len(arg_names)]
        defaults = func.__defaults__ or ()
        args = args + defaults[len(defaults) - (func.__code__.co_argcount - len(args)):]
        params = list(zip(arg_names, args))
        args = func_args[len(arg_names):]

        if args:
            params.append(('args', args))

        if func_kwargs:
            params.append(('kwargs', func_kwargs))

        # Print function call
        print('@! ' + func.__name__ + ' (' + ', '.join('%s = %r' % p for p in params) + ' )')

    return wrapper


def is_interactive():
    """
    Checks whether you're working in an interactive terminal (e.g., Jupyter notebook)
    """
    try:
        __IPYTHON__
        return True
    except:
        import __main__ as main
        return not hasattr(main, '__file__')


def get(data, key, default=None, sep='.'):
    try:
        return reduce(lambda c, k: c.get(k, default), key.split(sep), data)
    except KeyError:
        return default


def join_args(args, sep=',', prefix=False):
    out = sep.join('{}={}'.format(k, '"{}"'.format(v) if isinstance(v, str) else v) for k, v in args.items())
    if prefix and len(args) > 0:
        return f'{sep}{out}'
    else:
        return out


def printd(d, filters=None, indent=2, level=1):
    """
    Prints a concise summary of a dict-like object (arrays/tensors are not displayed - only their shape)

    :param d: input dictionary (or dict-like object)
    :param filters: a collection of regexp filters (for each recursion level) to select keys for printing
    :param indent: indentation size (#spaces)
    :param level: current recursion level (should be 1 in all direct calls)
    """
    if len(d) == 0:
        print('{}')
        return

    print('{')

    if isinstance(filters, str):
        raise ValueError('The filters should be a collection of strings (for each recursion level)')

    width = max([len(f'{k}') for k in d.keys()])
    has_dicts = any([isinstance(d[k], dict) for k in d.keys()])

    for k, v in d.items():

        if filters is not None and level <= len(filters) and not re.match(filters[level-1], k):
            continue

        # Print the key (align to the left if there are nested dicts, otherwise to the right)
        print((indent*level)*' ', end='')
        if has_dicts:
            print(f'{k:<{width}}: ', end='')
        else:
            print(f'{k:>{width}}: ', end='')

        # Print the values, depending on their type
        if isinstance(v, dict):
            printd(v, filters, indent, level + 1)

        elif hasattr(v, 'shape'):
            if v.ndim == 0:
                print(f'{format_number(v.item())} (0-d array)')
            elif len(v) == 0:
                print(f'empty array')
            else:
                print(f'array {v.shape} ∈ [{format_number(v.min())}, {format_number(v.max())}]')

        elif isinstance(v, str):
            print('"{}"'.format(v))

        elif isinstance(v, list):
            if len(v) == 0:
                print('[]')
            elif len(v) < 5 and is_number(v[0]):
                print(v)
            elif len(v) > 5 and is_number(v[0]):
                print(f'list of {len(v)} items: [{format_number(v[0])}, ..., {format_number(v[-1])}]')
            else:
                print(f'list of {len(v)} items of type <{type(v[0]).__name__}>')

        elif isinstance(v, tuple):
            if len(v) < 5:
                print(v)
            else:
                print(f'tuple of {len(v)} items: ({format_number(v[0])}, ..., {format_number(v[-1])})')

        else:
            if isinstance(v, float) or isinstance(v, int):
                print(format_number(v))
            else:
                print(v)

    print((indent*(level-1))*' ', end='')
    print('}')


def format_patch_shape(patch_size):
    if patch_size is None:
        return '?'
    elif any(x is None for x in patch_size):
        return '(rgb)' if patch_size[-1] == 3 else '(raw)'
    else:
        return '×'.join(str(x) for x in patch_size)


def shell(command, log=None, verbosity=2):

    if verbosity == 2:
        logger.info(f'>> {command}')
    elif verbosity == 1:
        if log is not None:
            print(f'\n>> {command} \\')
            print(f'   1> {log}.stdout 2> {log}.stderr')
        else:
            print(f'\n>> {command}\n')

    if log is None:
        p = subprocess.Popen(command, shell=True)
    else:
        p = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    outs, errs = p.communicate()

    if log is not None:

        assert outs is not None, 'The output from the sub-process is null'

        with open(f'{log}.stdout', 'w') as fo:
            for line in outs.decode('utf-8').splitlines():
                fo.write(f'{line}\n')

        with open(f'{log}.stderr', 'w') as fe:
            for line in errs.decode('utf-8').replace('\r', '\n').splitlines():
                fe.write(f'{line}\n')

    p.wait()
    return p.returncode


def tqdm_width():
    try:
        return os.get_terminal_size()[0]
    except OSError:
        return 200


def progress_bar(iter_total, desc=None):
    disabled = os.environ.get('DISABLE_TQDM', False)
    if is_number(iter_total):
        return tqdm(desc=desc, total=int(iter_total), ncols=tqdm_width(), disable=disabled, position=0, leave=True)
    else:
        return tqdm(iter_total, desc=desc, ncols=tqdm_width(), disable=disabled, position=0, leave=True)


def set_progressbar_status(status=False):
    os.environ['DISABLE_TQDM'] = str(status)


def factory(spec):
    if spec is None:
        return None

    import models

    if isinstance(spec, str):
        if spec[0] == '(' and spec[-1] == ')':
            instance = eval(spec)
        elif spec == 'None':
            return None
        else:
            instance = eval(f'models.{spec}')

    elif isinstance(spec, dict):
        cls = getattr(models, spec['class'])

        if 'args' in spec:
            args = get(spec, 'args', {})
            for k in args.keys():
                if isinstance(args[k], str) and args[k][0] == '(' and args[k][-1] == ')':
                    args[k] = eval(args[k])
            instance = cls(**args)

        elif 'checkpoint' in spec:
            instance = cls.restore(get(spec, 'checkpoint'))
        else:
            raise ValueError('Invalid model definition! Was expecting either "args" or "checkpoint" keys!')

    elif isinstance(spec, int) or isinstance(spec, float):
        return spec

    else:
        raise ValueError('Model definition not supported!')

    return instance
