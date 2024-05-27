using Morpheus
using LinearAlgebra
using Test

@testset "Morpheus.jl" begin
    # Write your tests here.
    # For example:
    x = Value(2.0)
    y = Value(3.0)
    a = Value(2.0)
    z = x + y
    t = x * y
    b = relu(Value(1.0))
    backward!(t)
    w = z / a
    @test x.data == 2.0
    @test y.data == 3.0
    @test z.data == 5.0
    @test b.data == 1.0
    @test t.data == 6.0
    @test t.grad == 1.0
    @test x.grad == 3.0
    @test y.grad == 2.0
    @test w.data == 2.5
end 
