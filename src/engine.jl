mutable struct Axion{T<:Number}  # Add type parameter T
    data::T
    grad::T
    _children::Union{Tuple{Axion{T}}, Tuple{Axion{T}, Axion{T}}, Set{Axion{T}}}
    _op::String
    _backward::Union{Function, Nothing}

    function Axion{T}(data::Number, grad::Number = zero(T), _children::Union{Tuple{Axion{T}}, Tuple{Axion{T}, Axion{T}}, Set{Axion{T}}} = Set{Axion{T}}(), _op::String = "", _backward::Union{Function, Nothing} = nothing) where {T<:Number}
        if typeof(data) !== T
            data = convert(T, data) # Store the converted value
        end
        new{T}(data, grad, _children, _op, _backward)
    end

    function Axion(data::Number; grad::Number = 0.0, _children::Union{Tuple{Axion{Float32}}, Tuple{Axion{Float32}, Axion{Float32}}, Set{Axion{Float32}}} = Set{Axion{Float32}}(), _op::String = "", _backward::Union{Function, Nothing} = nothing)
        if typeof(data) !== Float32 && typeof(grad) !== Float32
            data = convert(Float32, data)
            grad = convert(Float32, grad)
        end
        new{Float32}(data, grad, _children, _op, _backward)
    end
end

function Base.show(io::IO, val::Axion)
    print(io, "Axion(data=$(val.data), grad=$(val.grad), dtype=$(typeof(val.data)))")
end


# Addition Operator
function Base.:+(a::Union{Axion{T}, Number}, b::Union{Axion{T}, Number}) where T <: Number
    # Automatic conversion if either operand is a plain number
    a = a isa Axion{T} ? a : Axion{T}(a) 
    b = b isa Axion{T} ? b : Axion{T}(b)

    # Now, both a and b are guaranteed to be Axion{T}
    out = Axion{T}(a.data + b.data)
    out._children = (a, b)
    out._op = "+"

    function _backward()
        a.grad += out.grad
        b.grad += out.grad
    end
    out._backward = _backward
    return out
end

# Multiplication Operator
function Base.:*(a::Union{Axion{T}, Number}, b::Union{Axion{T}, Number}) where T
    a = a isa Axion{T} ? a : Axion{T}(a) 
    b = b isa Axion{T} ? b : Axion{T}(b)
    out = Axion{T}(a.data * b.data)
    out._children = (a, b)
    out._op = "*"

    function _backward()
        a.grad += b.data * out.grad
        b.grad += a.data * out.grad
    end
    out._backward = _backward
    return out
end

# Division Operator
function Base.:/(a::Union{Axion{T}, Number}, b::Union{Axion{T}, Number}) where T
    a = a isa Axion{T} ? a : Axion{T}(a) 
    b = b isa Axion{T} ? b : Axion{T}(b)
    out = Axion{T}(a.data / b.data) 
    out._children = (a, b)
    out._op = "/"

    function _backward()
        a.grad += out.grad / b.data  
        b.grad -= out.grad * a.data / (b.data ^ 2) 
    end
    out._backward = _backward
    return out
end

# Exponential Function
function Base.exp(a::Axion{T}) where T
    out = Axion{T}(exp(a.data))
    out._children = (a,)
    out._op = "exp"

    function _backward()
        a.grad += out.grad * out.data
    end
    out._backward = _backward
    return out
end

# Power Operator
function Base.:^(a::Axion{T}, b::Real) where T
    out = Axion{T}(a.data^b) 
    out._children = (a,)
    out._op = "^$b"

    function _backward()
        a.grad += b * a.data^(b - one(T)) * out.grad  # Type-stable 1
    end
    out._backward = _backward
    return out
end

# Subtraction Operator
function Base.:-(a::Union{Axion{T}, Number}, b::Union{Axion{T}, Number}) where T
    a = a isa Axion{T} ? a : Axion{T}(a) 
    b = b isa Axion{T} ? b : Axion{T}(b)
    out = Axion{T}(a.data - b.data)
    out._children = (a, b)
    out._op = "-"

    function _backward()
        a.grad += out.grad
        b.grad -= out.grad  
    end
    out._backward = _backward
    return out
end


# Dot Product Function
function dot(weights::Matrix{Axion{T}}, x::Union{Matrix{Axion{}}, Matrix{Number}}) where T <: Number
    result = Axion{T}(zero(T)) 

    for (w, xi) in zip(weights, x)
        result += w * xi  
    end

    return result
end

function backward!(a::Axion{T}) where T<:Number
    a.grad = 1.0
    topo = Vector{Axion{T}}([a])  # Initialize topo with the output node
    while !isempty(topo)
        node = pop!(topo)
        if node._backward !== nothing
            node._backward()
        end
        for child in node._children
            pushfirst!(topo, child)  # Add children in reverse order
        end
    end
end

function to_value(data::Number, dtype::Type{<:Number}=Float32)
    return Axion{dtype}(data)  
end

function to_value(data::AbstractArray, dtype::Type{<:Number}=Float32)
    return [to_value(x, dtype) for x in data]
end

# Conversion Functions (Simplified)
to_value(x::Matrix{Axion{T}}) where T = x[:]  # Flatten to a vector of Axions
to_matrix_value(x::Vector{Axion{T}}) where T = reshape(x, (length(x), 1)) # Reshape to a matrix

export Axion
export backward!
export to_value
export dot
export to_matrix_value
