import string
import re
import numpy as np
import time
from colorama import Fore, Back, Style, init
import functools
import streamlit as st
from loguru import logger

init(autoreset=True)


# = Decorators =================================================================


def time_it(func):
    def wrapper(*args, **kwargs):
        func_name = func.__name__
        logger.info(f"🖥️    Started: '{func_name}'")
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        func_name = func.__name__
        logger.info(f"✅ Completed: '{func_name}' ⚡️{elapsed_time:.6f} sec")
        return result

    return wrapper


def debug_info(func):
    """Decorator that prints function execution information."""

    @functools.wraps(func)
    def wrapper_debug_info(*args, **kwargs):
        args_repr = [repr(a) for a in args]
        kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]
        signature = ", ".join(args_repr + kwargs_repr)
        print(f"Calling {func.__name__}({signature})")

        # Get the start time
        start_time = time.perf_counter()

        value = func(*args, **kwargs)

        # Get the end time
        end_time = time.perf_counter()

        # Calculate the execution time
        run_time = end_time - start_time

        print(
            f"Finished {func.__name__!r} in {run_time:.4f} secs with result {value!r}"
        )
        return value

    return wrapper_debug_info


# end of file
