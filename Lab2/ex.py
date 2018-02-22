def gen(n: int):
    for k in range(n):
        yield k*k


def ex():
    lst = [5, 4, 3, 5, 3]
    t = map(lambda x: x**2, lst)
    for u in t:
        print(u)


if __name__ == "__main__":
    g = gen(10)
