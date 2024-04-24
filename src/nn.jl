include("engine.jl")
using Distributions

mutable struct Neuron
    weights::Vector{Value}
    bias::Value
    
    function Neuron(nin::Int)
        weights::Vector{Value} = [Value(rand(Uniform(-1, 1))) for _ in 1:nin]
        bias::Value = Value(rand(Uniform(-1, 1)))
        new(weights, bias)
    end
end

mutable struct Layer
    neurons::Vector{Neuron}
    
    function Layer(nin::Int, nout::Int)
        neurons::Vector{Neuron} = [Neuron(nin) for _ in 1:nout]
        new(neurons)
    end
end

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

function Base.show(io::IO, neuron::Neuron)
    print(io, "Neuron(weights=$(neuron.weights), bias=$(neuron.bias)")
end

function Base.show(io::IO, layer::Layer)
    print(io, "Layer(neurons=$(layer.neurons)")
end

function Base.show(io::IO, mlp::MLP)
    print(io, "MLP(layers=$(mlp.layers)")
end

function parameters(neuron::Neuron)
    return push!(neuron.weights, neuron.bias)
end

function parameters(layer::Layer)
    return [p for neuron in layer.neurons for p in parameters(neuron)]
end

function parameters(mlp::MLP)
    return [p for layer in mlp.layers for p in parameters(layer)]
end

function (neuron::Neuron)(x::Union{Vector{Float64}, Vector{Value}})
    act = dot(neuron.weights, x) + neuron.bias
    return tanh(act)
end

function (layer::Layer)(x::Union{Vector{Float64}, Vector{Value}})
    outs::Vector{Value} = [n(x) for n ∈ layer.neurons]
    return outs
end

function (mlp::MLP)(x::Vector{Float64})
    for layer in mlp.layers
        x = layer(x)
    end
    return x
end

export MLP
export Layer
export Neuron
export parameters
export parameters
export parameters
