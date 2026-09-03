
import json, math
def _raises(fn, exc):
    try:
        fn()
    except exc:
        return True
    except Exception:
        return False
    return False

def _retry_ok():
    calls = []
    def fn():
        calls.append(1)
        if len(calls) < 3:
            raise RuntimeError("transient")
        return "ok"
    return (len(calls), bounded_retry(fn, 5, lambda e: True)) if False else (
        (lambda r: (len(calls), r))(bounded_retry(fn, 5, lambda e: True)))

def _retry_exhausted():
    calls = []
    def fn():
        calls.append(1)
        raise RuntimeError("always")
    try:
        bounded_retry(fn, 2, lambda e: True)
    except Exception:
        pass
    return len(calls)

def _retry_nonretryable():
    calls = []
    def fn():
        calls.append(1)
        raise ValueError("fatal")
    try:
        bounded_retry(fn, 5, lambda e: not isinstance(e, ValueError))
    except Exception:
        pass
    return len(calls)
