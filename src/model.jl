mutable struct MLP
    size::Vector{Int}
    layers::Vector{Layer}
    
    function MLP(nin::Int , nouts::Vector{Int})
        size::Vector{Int} = nouts
        pushfirst!(size, nin)
        layers::Vector{Layer} = [Layer(size[i], size[i+1]) for i in 1:length(nouts)-1]
        new(size, layers)
    end
end

function Base.show(io::IO, mlp::MLP)
    print(io, "MLP(layers=$(mlp.layers)")
end

function parameters(mlp::MLP)
    return [p for layer in mlp.layers for p in parameters(layer)]
end

function (mlp::MLP)(x::Vector{Float64})
    for layer in mlp.layers
        x = layer(x)
    end
    return x
end

export MLP
export parameters
