mutable struct Layer{T<:Number}
    neurons::Vector{Neuron}
    dropout::Number
    is_training::Bool
    
    function Layer{T}(input_dim::Int, output_dim::Int, activation::Function = identity; dropout::Number = 0.0, is_training::Bool = true) where T<:Number
        neurons = Vector{Neuron{T}}(undef, output_dim)
        for i in 1:output_dim
            neurons[i] = Neuron{T}(input_dim, activation)
        end
        new(neurons, dropout, is_training)
    end
end


function parameters(layer::Layer)::Vector{Axion}
    return [p for neuron ∈ layer.neurons for p ∈  parameters(neuron)]
end

function Base.show(io::IO, layer::Layer)
    print(io, "size=$(length(layer.neurons))")
end

# Function for Matrix Input
function (layer::Layer)(x::AbstractMatrix{<:Number})::AbstractVector{Axion}
    result = [n(x) for n in layer.neurons]

    if layer.is_training && layer.dropout > 0.0
        mask = rand(Bernoulli(1 - layer.dropout), size(x, 2), length(layer.neurons))  # Create a mask for each column of x
        result = [mask[:, i] .* n(x) ./ (1 - layer.dropout) for (i, n) in enumerate(layer.neurons)] 
    end
    return result
end

# Function for Axion Vector Input
function (layer::Layer)(x::AbstractVector{<:Axion})::AbstractVector{Axion}
    result = [n(x) for n in layer.neurons]

    if layer.is_training && layer.dropout > 0.0
        mask = rand(Bernoulli(1 - layer.dropout), length(layer.neurons))
        result = [mask[i] ? n(x) / (1 - layer.dropout) : Axion(zero(eltype(x[1].data))) for (i, n) in enumerate(layer.neurons)]
    end
    return result
end

function zero_grad!(layer::Layer)
    for val  ∈ parameters(layer)
        if val.grad != 0.0
            val.grad = 0.0
        end
    end
end

export Layer
