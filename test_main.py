from testing import testobj




if __name__ == "__main__":
    a = testobj(10)
    b = testobj(20)
    c = a + b
    print(c.data)
    print(a.func)
    print(b.func)