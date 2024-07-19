class Scalar:

    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0.0
        self._op = _op
        self._prev = set(_children)
        self._backward = lambda: None

    def __repr__(self):
        return f"Scalar(data={self.data})"

    def __add__(self, other):
        other = other if isinstance(other, Scalar) else Scalar(other)
        out = Scalar(self.data + other.data, (self, other), '+')
        
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        
        return out
        
    def __mul__(self, other):
        other = other if isinstance(other, Scalar) else Scalar(other)
        out = Scalar(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad 
        out._backward = _backward
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Scalar(self.data**other, (self,), f'**{other}')

        def _backward():
            self.grad += (other * self.data**(other-1)) * out.grad
        out._backward = _backward
        return out


    def sigmoid(self):
        ## TODO: implement sigmoid func
        return 0.0

    def ReLU(self):
        ## TODO: implement ReLU func
        return 0.0

    def tanh(self):
        x = (math.exp(2*self.data) - 1) / (math.exp(2*self.data) + 1)
        out = Scalar(x, (self,), 'tanh')

        def _backward():
            self.grad += (1-x**2) * out.grad
        out._backward = _backward
        
        return out

    def backward_prop(self):
        
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad=1.0
        for node in reversed(topo):
            node._backward()


    def __neg__(self): # -self
        return self * -1

    def __radd__(self, other): # other + self
        return self + other

    def __sub__(self, other): # self - other
        return self + (-other)

    def __rsub__(self, other): # other - self
        return other + (-self)

    def __rmul__(self, other): # other * self
        return self * other

    def __truediv__(self, other): # self / other
        return self * other**-1

    def __rtruediv__(self, other): # other / self
        return other * self**-1

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"