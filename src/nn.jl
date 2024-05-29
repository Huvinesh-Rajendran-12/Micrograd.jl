using Distributions

mutable struct Neuron
    weights::Vector{Value}
    bias::Value
    activation::Function
    
    function Neuron(nin::Int, activation::Function = identity)
        weights::Vector{Value} = [Value(rand(Uniform(-1, 1))) for _ ∈ 1:nin]
        bias::Value = Value(rand(Uniform(-1, 1)))
        new(weights, bias, activation)
    end
end


function Base.show(io::IO, neuron::Neuron)
    print(io, "Neuron(weights=$(neuron.weights), bias=$(neuron.bias)")
end

function parameters(neuron::Neuron)::Vector{Value}
    return vcat(neuron.weights, [neuron.bias])
end


function (neuron::Neuron)(x::Vector{Value})
    z = dot(neuron.weights, x) + neuron.bias
    return neuron.activation(z)
end


export Neuron
export parameters
export parameters
export parameters
