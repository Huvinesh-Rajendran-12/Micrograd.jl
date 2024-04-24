# Micrograd

[![Build Status](https://github.com/Huvinesh-Rajendran-12/Micrograd.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/Huvinesh-Rajendran-12/Micrograd.jl/actions/workflows/CI.yml?query=branch%3Amain)

# Usage

## Import Micrograd
```julia
using Micrograd
```
## Create a Value
In Micrograd, the fundamental data structure used for numerical computation is Value which is an abstraction over a scalar numerical value. Value allows us to efficiently manipulate the data and also track additional computational information. 

```julia
x = Value(2.0)
# output: Value(data=2.0, grad=0.0)
```
