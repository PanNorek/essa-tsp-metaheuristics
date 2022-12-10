import time


def time_it(func):
    def wrapper(*args, **kwargs):
        tic = time.time()
        result = func(*args, **kwargs)
        toc = time.time()
        result.solving_time = toc-tic
        return result
    return wrapper
