using LinearAlgebra

mutable struct SGD
    learning_rate::Float64
end

function SGD(; learning_rate=0.01)
    return SGD(learning_rate)
end

function (sgd::SGD)(params::Vector{Value})
    for param in params
        param.data -= sgd.learning_rate * param.grad
    end
end

mutable struct Adam
    learning_rate::Float64
    beta1::Float64
    beta2::Float64
    epsilon::Float64
    t::Int  # Iteration counter
    m::Dict{Value,Value}
    v::Dict{Value,Value}

    function Adam(; learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8)
        new(learning_rate, beta1, beta2, epsilon, 0, Dict{Value,Value}(), Dict{Value,Value}())
    end
end

function (adam::Adam)(params::Vector{Value})
    adam.t += 1
    t = adam.t
    corr_factor1 = 1 - adam.beta1^t
    corr_factor2 = 1 - adam.beta2^t

    for param in params
        if !haskey(adam.m, param)
            adam.m[param] = Value(0.0)
            adam.v[param] = Value(0.0)
        end

        # Update biased first moment estimate
        adam.m[param] = adam.beta1 * adam.m[param] + (1 - adam.beta1) * param.grad

        # Update biased second raw moment estimate
        adam.v[param] = adam.beta2 * adam.v[param] + (1 - adam.beta2) * param.grad^2

        # Compute bias-corrected first moment estimate
        m_hat = adam.m[param] / corr_factor1

        # Compute bias-corrected second raw moment estimate
        v_hat = adam.v[param] / corr_factor2

        # Update parameters
        param.data -= adam.learning_rate * m_hat.data / (sqrt(v_hat.data) + adam.epsilon)
    end
end

