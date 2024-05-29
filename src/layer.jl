mutable struct Layer
    neurons::Vector{Neuron}
    dropout::Float64
    is_training::Bool
    
    function Layer(input_dim::Int, output_dim::Int, activation::Function = identity)
        neurons = Vector{Neuron}(undef, output_dim)
        for i in 1:output_dim
            neurons[i] = Neuron(input_dim, activation)
        end
        new(neurons)
    end
end


function parameters(layer::Layer)::Vector{Value}
    return [p for neuron ∈ layer.neurons for p ∈  parameters(neuron)]
end

function Base.show(io::IO, layer::Layer)
    print(io, "Layer(neurons=$(layer.neurons)")
end

function (layer::Layer)(x::Union{Vector{Float64}, Vector{Value}})::Vector{Value}
    if layer.is_training && layer.dropout > 0.0
        mask = rand(Bernoulli(1 - layer.dropout), length(layer.neurons))
        return [mask[i] ? n(x) / (1 - layer.dropout) : Value(0.0) for (i, n) ∈ enumerate(layer.neurons)]
    else
        return [n(x) for n in layer.neurons]
    end
end

function zero_grad!(layer::Layer)
    for val  ∈ parameters(layer)
        if val.grad != 0.0
            val.grad = 0.0
        end
    end
end
