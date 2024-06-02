function mean_squared_error(predicted::Vector{Axion}, actual::Vector{Float64})
    @assert length(predicted) == length(actual) "Predicted and actual must have the same length"
    return sum((predicted.data .- actual) .^ 2) / length(predicted)
end

function mean_absolute_error(predicted::Vector{Axion}, actual::Vector{Float64})
    @assert length(predicted) == length(actual) "Predicted and actual must have the same length"
    return sum(abs.(predicted.data .- actual)) / length(predicted)
end

function binary_cross_entropy(predicted::Vector{Axion}, actual::Vector{Float64})
    @assert length(predicted) == length(actual) "Predicted and actual must have the same length"
    epsilon = 1e-10  # Small value to avoid log(0)
    return sum(-(actual .* log.(predicted.data .+ epsilon) + (1 .- actual) .* log.(1 .- predicted.data .+ epsilon))) / length(predicted)
end

function categorical_cross_entropy(predicted::Vector{Axion}, actual::Vector{Int})
    @assert length(predicted) == length(actual) "Predicted and actual must have the same length"
    epsilon = 1e-10  # Small value to avoid log(0)
    return sum(-log.(predicted[i].data .+ epsilon) for i in actual) / length(predicted)
end

export mean_squared_error
export mean_absolute_error
export binary_cross_entropy
export categorical_cross_entropy
