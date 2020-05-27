def bool_arg(x):
    try:
        return bool(int(x))
    except ValueError:
        pass
    x = x.lower()
    if x.startswith('f'):
        return False
    if x.startswith('t'):
        return True
    raise ValueError('Invalid: %s' % x)
