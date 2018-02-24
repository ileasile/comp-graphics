def __convex_win(args: str, x):
    from subprocess import Popen, PIPE
    qconvex_exe = 'bin_win/qconvex.exe'
    add_args = args.split(' ')

    dim = len(x[0])
    sz = len(x)
    data = [str.encode(str(dim)), str.encode(str(sz))]
    data.extend([str.encode(' '.join(map(repr, row))) for row in x])
    data_str = b'\n'.join(data)

    p = Popen([qconvex_exe] + add_args, stdout=PIPE, stdin=PIPE, stderr=PIPE)

    stdout_data = str(p.communicate(input=data_str)[0])
    result = list(map(str.strip, stdout_data.strip().split('\\r\\n')))
    n = len(result) - 2
    result[0] = str(n)
    del result[-1]
    return result


def __get_qconvex():
    try:
        from pyhull.convex_hull import qconvex as qh
        return qh
    except ImportError:
        return __convex_win


qconvex = __get_qconvex()
