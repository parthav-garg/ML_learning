
class testobj():
    def __init__(self, data):
        self.data = data
        self.func = ""
    def __add__(self, other):
        if isinstance(other, testobj):
            pass
        else:
            testobj(other)
        self.func = "Hello"
        return testobj(self.data + other.data) 
    def __radd__(self, other):
        return self + other
    