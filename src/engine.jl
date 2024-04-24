using LinearAlgebra

mutable struct Value
    data::Union{Float64, Int}
    grad::Float64
    _children::Union{Tuple{Value}, Tuple{Value, Value}, Set{Value}}
    _op::String
    _backward::Union{Function, Nothing}

    function Value(data::Union{Float64, Int}, grad::Float64 = 0.0, _children::Union{Tuple{Value}, Tuple{Value, Value}, Set{Value}} = Set{Value}(), _op::String = "", _backward::Union{Function, Nothing} = nothing)
        new(data, grad, _children, _op, _backward)
    end
end

function Base.show(io::IO, val::Value)
    print(io, "Value(data=$(val.data), grad=$(val.grad))")
end


function Base.:+(a::Value, b::Union{Value, Int, Float64})
    if b isa Integer || b isa Float64
        b = Value(b)
    end
    out = Value(a.data + b.data, 0.0, (a , b), "+")
    function _backward()
        a.grad += 1.0 * out.grad
        b.grad += 1.0 * out.grad
    end
    out._backward = _backward
    return out
end

function Base.:+(a::Union{Value, Int, Float64}, b::Value)
    if a isa Integer || a isa Float64
        a = Value(a)
    end
    out = Value(a.data + b.data, 0.0, (a , b), "+")
    function _backward()
        a.grad += 1.0 * out.grad
        b.grad += 1.0 * out.grad
    end
    out._backward = _backward
    return out
end

function Base.:+(a::Value, b::Value)
    out = Value(a.data + b.data, 0.0, (a , b), "+")
    function _backward()
        a.grad += 1.0 * out.grad
        b.grad += 1.0 * out.grad
    end
    out._backward = _backward
    return out
end

function Base.:*(a::Value, b::Union{Value, Int, Float64})
    if b isa Integer || b isa Float64
        b = Value(b)
    end
    out = Value(a.data * b.data, 0.0, (a, b), "*")
    function _backward()
        a.grad += b.data * out.grad
        b.grad += a.data * out.grad
    end
    out._backward = _backward
    return out
end

function Base.:*(a::Union{Value, Int, Float64}, b::Value)
    if a isa Integer || a isa Float64
        a = Value(a)
    end
    out = Value(a.data * b.data, 0.0, (a, b), "*")
    function _backward()
        a.grad += b.data * out.grad
        b.grad += a.data * out.grad
    end
    out._backward = _backward
    return out
end

function Base.:*(a::Value, b::Value)
    out = Value(a.data * b.data, 0.0, (a, b), "*")
    function _backward()
        a.grad += b.data * out.grad
        b.grad += a.data * out.grad
    end
    out._backward = _backward
    return out
end

function Base.:/(a::Value, b::Union{Value, Int, Float64})
    if b isa Integer || b isa Float64
        b = Value(b)
    end
    out = Value(a.data * (b.data^-1), 0.0, (a, b), "/")
    function _backward()
        a.grad += 1 * b.data^-1 * out.grad
        b.grad += -a.data*b.data^-2 * out.grad
    end
    out._backward = _backward
    return out
end

function Base.:/(a::Union{Value, Int, Float64}, b::Value)
    if a isa Integer || a isa Float64
        a = Value(a)
    end
    out = Value(a.data * (b.data^-1), 0.0, (a, b), "/")
     function _backward()
        a.grad += 1 * b.data^-1
        b.grad += -a.data*b.data^-2
    end
    out._backward = _backward
    return out
end

function Base.:/(a::Value, b::Value)
    out = Value(a.data * (b.data^-1), 0.0, (a, b), "/")
     function _backward()
        a.grad += 1 * b.data^-1
        b.grad += -a.data*b.data^-2
    end
    out._backward = _backward
    return out
end

function Base.:exp(a::Value)
    out = Value(exp(a.data), 0.0, (a, b), "exp")
    function _backward()
        a.grad += out.grad * out.data
    end
    out._backward = _backward
    return out
end

function Base.:^(a::Value, b::Union{Int, Float64})
    @assert b isa Union{Int, Float64} "Right now other is only supported as an Integer or Float."
    out = Value(a.data ^ b, 0.0, (a,), "^$b")
    function _backward()
        a.grad += b * a.data^(b - 1) * out.grad
    end
    out._backward = _backward
    return out
end

function Base.:-(a::Value, b::Union{Value, Int, Float64})
    if b isa Integer || b isa Float64
        b = Value(b)
    end
    out = Value(a.data - b.data, 0.0, (a, b), "-")
    function _backward()
        a.grad += 1.0 * out.grad
        b.grad += -1.0 * out.grad
    end
    out._backward = _backward
    return out
end

function Base.:-(a::Union{Value, Int, Float64}, b::Value)
    if a isa Integer || a isa Float64
        a = Value(a)
    end
    out = Value(a.data - b.data, 0.0, (a, b), "-")
    function _backward()
        a.grad += 1.0 * out.grad
        b.grad += -1.0 * out.grad
    end
    out._backward = _backward
    return out
end

function Base.tanh(a::Value)
    out = Value(tanh(a.data), 0.0, (a,), "tanh")
    function _backward()
        a.grad += (1-out.data^2) * out.grad
    end
    out._backward = _backward
    return out
end

function relu(a::Value)
    out = Value(max(0.0, a.data), 0.0, (a,), "relu")
    function _backward()
        a.grad += (a.data > 0.0 ? 1.0 : 0) * out.grad
    end
    out._backward = _backward
    return out
end

function LinearAlgebra.dot(weights::Vector{Value}, x::Union{Vector{Float64}, Vector{Value}})
    result = Value(0.00)  # initialize the result with a zero value of the same type as weights
    for (w, xi) in zip(weights, x)
        result += w * xi  # assume w * xi is defined for Value and Float64
    end
    return result
end

function backward!(a::Value)
    a.grad = 1.0
    topo::Vector{Value} = Vector{Value}[]   
    visited::Set{Value} = Set{Value}()
    function build_topo(v::Value)
        if v ∉ visited
            push!(visited, v)
            for child in v._children
                build_topo(child)
            end
            push!(topo, v)
        end
    end
    build_topo(a)
    for node ∈ reverse(topo)
        if node._backward ≠ nothing
            node._backward()
        end
    end
end

export Value
export backward!
export relu
