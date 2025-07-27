"""Utility functions."""

# Authors: Hamza Cherkaoui

import numpy as np
import cProfile


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

