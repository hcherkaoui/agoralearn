""" Utility functions."""

# Authors: Hamza Cherkaoui

import cProfile
import numpy as np


def format_duration(seconds):
    """Converts a duration in seconds to HH:MM:SS format."""
    s, ns = divmod(seconds, 1)
    m, s = divmod(int(s), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}h {m:02d}m {s:02d}s {int(1000.0 * ns):03d}ms"


def profile_me(func):  # pragma: no cover
    """Profiling decorator, produce a report <func-name>.profile to be open as
    Place @profile_me on top of the desired function, then:
    'python -m snakeviz <func-name>.profile'

    Parameters
    ----------
    func : func, function to profile
    """

    def profiled_func(*args, **kwargs):
        filename = func.__name__ + ".profile"
        prof = cProfile.Profile()
        ret = prof.runcall(func, *args, **kwargs)
        prof.dump_stats(filename)
        return ret

    return profiled_func


def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance.

    Parameters
    ----------
    seed : None, int, random-instance, (default=None), random-instance
        or random-seed used to initialize the random-instance

    Return
    ------
    random_instance : random-instance used to initialize the analysis
    """
    if seed is None or seed is np.random:
        return np.random.default_rng(seed=None)
    if isinstance(seed, (int, np.integer)):
        return np.random.default_rng(seed=seed)
    if isinstance(seed, np.random.Generator):
        return seed
    raise ValueError(
        f"{seed} cannot be used to seed a " f"numpy.random.Generator instance"
    )


def experiment_suffix(args, keys=None, prefix='____'):
    """
    Create a stats string from selected variables in an args-like object.

    Parameters
    ----------
    args : Namespace or dict
        Object containing configuration attributes.
    keys : list of str, optional
        Subset of keys to include. If None, all keys are included.
    prefix : str
        Prefix for the resulting string.

    Returns
    -------
    str
        Formatted string like '____lbda_1.75e-3__seed_0'.
    """
    if isinstance(args, dict):
        args_dict = args
    else:
        args_dict = vars(args)

    if keys is None:
        keys = sorted(args_dict.keys())

    parts = []
    for k in keys:
        val = args_dict[k]
        if isinstance(val, float):
            parts.append(f"{k}_{val:.2e}")
        else:
            parts.append(f"{k}_{val}")
    return prefix + "__".join(parts)
