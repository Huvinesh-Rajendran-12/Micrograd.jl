mutable struct SGD{T<:Number}
    learning_rate::T

    SGD(learning_rate::T) where T<:Number = new{T}(learning_rate)
end


function (sgd::SGD{T})(params::Vector{Axion{T}}) where T<:Number
    for param in params
        param.data -= sgd.learning_rate * param.grad
    end
end

mutable struct Adam{T<:Number}
    learning_rate::T
    beta1::T
    beta2::T
    epsilon::T
    t::Int  
    m::Dict{Axion{T}, Axion{T}}
    v::Dict{Axion{T}, Axion{T}}

    function Adam(
        ;
        learning_rate::T=convert(T, 0.001),
        beta1::T=convert(T, 0.9),
        beta2::T=convert(T, 0.999),
        epsilon::T=convert(T, 1e-8),
    ) where {T<:Number}
        new{T}(learning_rate, beta1, beta2, epsilon, 0, Dict{Axion{T}, Axion{T}}(), Dict{Axion{T}, Axion{T}}())
    end
end

function (adam::Adam{T})(params::Vector{Axion{T}}) where T
    adam.t += 1
    t = adam.t
    corr_factor1 = one(T) - adam.beta1^t
    corr_factor2 = one(T) - adam.beta2^t

    for param in params
        if !haskey(adam.m, param)
            adam.m[param] = Axion{T}(zero(T))
            adam.v[param] = Axion{T}(zero(T))
        end

        # Update biased first moment estimate (type-stable)
        adam.m[param] = adam.beta1 * adam.m[param] + (one(T) - adam.beta1) * param.grad

        # Update biased second raw moment estimate (type-stable)
        adam.v[param] = adam.beta2 * adam.v[param] + (one(T) - adam.beta2) * param.grad^2

        # Compute bias-corrected first moment estimate
        m_hat = adam.m[param] / corr_factor1

        # Compute bias-corrected second raw moment estimate
        v_hat = adam.v[param] / corr_factor2

        # Update parameters (type-stable)
        param.data -= adam.learning_rate * m_hat.data / (sqrt(v_hat.data) + adam.epsilon)
    end
end
