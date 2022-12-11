import time


def time_it(func):
    """ Decorator to measure time of function execution """
    def wrapper(*args, **kwargs):
        tic = time.perf_counter()
        result = func(*args, **kwargs)
        toc = time.perf_counter()
        result.time = toc-tic
        return result
    return wrapper
