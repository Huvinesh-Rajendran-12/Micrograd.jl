# Morpheus

[![Build Status](https://github.com/Huvinesh-Rajendran-12/Morpheus.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/Huvinesh-Rajendran-12/Micrograd.jl/actions/workflows/CI.yml?query=branch%3Amain)

# Usage

## Import Morpheus
```julia
using Morpheus
```
## Create a Value
In Morpheus, the fundamental data structure used for numerical computation is Value which is an abstraction over a scalar numerical value. Value allows us to efficiently manipulate the data and also track additional computational information. 

```julia
x = Value(2.0)
# output: Value(data=2.0, grad=0.0)
```
```julia
x.data
# output: 2.0

x.grad
# output: 0.0
```

```julia
y = Value(3.0)
z = x + y
# output: Value(data=5.0, grad=0.0)

t = x * y
# output: Value(data=6.0, grad=0.0)
```


