mutable struct Model{T<:Number}
  layers::Vector{Layer{T}}
  is_training::Bool

  # Constructor
  function Model(layers::Vector{Layer{T}}; is_training::Bool = true) where T<:Number
    new{T}(layers, is_training)
  end
end

function (model::Model{T})(x::Matrix{T}) where T
  for layer ∈ model.layers
    x = layer(x, is_training=model.is_training)
  end
  return x
end

export Model
