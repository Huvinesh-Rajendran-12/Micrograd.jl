using Distributions

mutable struct Neuron{T<:Number}
    weights::Vector{Axion{T}}
    bias::Axion{T}
    activation::Function

    function Neuron{T}(nin::Int, activation::Function=identity) where T<:Number
        weights = [Axion{T}(rand(Uniform(-1, 1))) for _ in 1:nin]
        bias = Axion{T}(rand(Uniform(-1, 1)))
        new{T}(weights, bias, activation)
    end
end

function Base.show(io::IO, neuron::Neuron)
    print(io, "Neuron(weights=$(neuron.weights), bias=$(neuron.bias)")
end

function parameters(neuron::Neuron)::Vector{Axion}
    return vcat(neuron.weights, [neuron.bias])
end


function (neuron::Neuron{T})(x::Vector{Axion{T}}) where T<:Number
    z = dot(neuron.weights, x) + neuron.bias
    return neuron.activation(z)
end


export Neuron
export parameters
export parameters
export parameters
