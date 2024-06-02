# Tanh Activation
function Base.tanh(a::Axion)
    T = typeof(a.data)
    out = Axion{T}(tanh(a.data)) 
    out._children = (a,)
    out._op = "tanh"

    out._backward = function _backward()
        a.grad += (1 - out.data^2) * out.grad
    end

    return out
end

# Identity Activation (No change needed)
Base.identity(a::Axion) = a

# ReLU Activation
function relu(x::Axion)
    T = typeof(x.data)
    out = Axion{T}(max(zero(x.data), x.data))
    out._children = (x,)
    out._op = "relu"
    out._backward = function _backward()
        x.grad += (x.data > 0 ? one(x.data) : zero(x.data)) * out.grad  # Type-stable 0 and 1
    end

    return out
end

# Leaky ReLU Activation
function leakyrelu(x::Axion, alpha::FP = 0.01)
    T = typeof(x.data)
    out = Axion{T}(max(alpha * x.data, x.data))
    out._children = (x,)
    out._op = "leakyrelu"
    out._backward = function _backward()
        x.grad += (out.data > 0 ? one(x.data) : alpha) * out.grad  # Type-stable 0 and 1
    end

    return out
end

export relu, leakyrelu
