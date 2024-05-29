function Base.tanh(a::Value)
    out = Value(tanh(a.data), 0.0, (a,), "tanh")
    function _backward()
        a.grad += (1-out.data^2) * out.grad
    end
    out._backward = _backward
    return out
end

function Base.identity(a::Value)
    return a
end

function relu(a::Value)
    out = Value(max(0.0, a.data), 0.0, (a,), "relu")
    function _backward()
        a.grad += (a.data > 0.0 ? 1.0 : 0) * out.grad
    end
    out._backward = _backward
    return out
end

export relu
