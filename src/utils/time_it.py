import time


def time_it(func):
    def wrapper(*args, **kwargs):
        tic = time.time()
        result = func(*args, **kwargs)
        toc = time.time()
        print(f"Time elapsed: {toc-tic}")
        return result
    return wrapper
