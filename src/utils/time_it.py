import time


def time_it(func):
    def wrapper(*args, **kwargs):
        tic = time.perf_counter()
        result = func(*args, **kwargs)
        toc = time.perf_counter()
        result.time = toc-tic
        return result
    return wrapper
