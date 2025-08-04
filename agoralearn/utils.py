"""Utility functions."""

# Authors: Hamza Cherkaoui

import os
import subprocess
import numpy as np
import cProfile


def experiment_filename_suffix(args, keys=None, suffix='____'):
    """
    Generate a filename suffix from argparse arguments.

    Parameters:
    -----------
    args (argparse.Namespace): Parsed arguments from argparse.
    keys (list or None): List of keys to include. If None, include all.

    Returns:
    --------
    suffix (str): Filename suffix like '__n_epochs_5__device_cpu'.
    """
    if keys is None:
        keys = sorted(vars(args).keys())

    suffix_parts = []
    for key in keys:
        value = getattr(args, key, None)
        if isinstance(value, float):
            value_str = f"{value:.0e}" if value < 1e-2 else str(value)
        else:
            value_str = str(value)
        suffix_parts.append(f"{key}_{value_str}")

    return suffix + "__".join(suffix_parts)


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


def profile_me(func):
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


def crop_pdf(input_pdf, output_pdf=None, verbose=True):
    """
    Crop a PDF file using pdfcrop (requires it to be installed).

    Parameters:
    -----------
    input_pdf (str): Path to the input PDF.
    output_pdf (str or None): Path to save the cropped PDF. If None, overwrites input.
    verbose (bool): Whether to print status messages.

    Returns:
    --------
    success (bool): True if pdfcrop ran successfully, False otherwise.
    """
    if output_pdf is None:
        output_pdf = input_pdf

    if not os.path.exists(input_pdf):
        if verbose:
            print(f"[ERROR] Input PDF not found: {input_pdf}")
        return False

    if verbose:
        print(f"[INFO] Cropping pdf file: {input_pdf}")

    try:
        _ = subprocess.run(
            ['pdfcrop', input_pdf, output_pdf],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            check=True,
            text=True
        )
        return True
    except subprocess.CalledProcessError as e:
        if verbose:
            print(f"[ERROR] pdfcrop failed: {e.stderr.strip()}")
        return False
    except FileNotFoundError:
        if verbose:
            print("[ERROR] 'pdfcrop' not found. Make sure TeX Live is installed.")
        return False

