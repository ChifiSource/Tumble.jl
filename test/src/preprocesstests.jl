using LinearAlgebra
using DataFrames
using Test
println("LATHE.PREPROCESS TESTS")
println("......................")
# DECOMPOSITION
println("DECOMPOSITION")
println(".............")
using Lathe.preprocess: SVD
@testset "svd test for $T" for T in (Float32, Float64)
  A = randn(T, 100, 100)
  x = SVD(A)
  tol = T == Float32 ? 1e-5 : 1e-10
  @test (norm(A - (x.U * diagm(0 => x.S) * x.V'), 2) / norm(A, 2)) <= tol
end
using Lathe.preprocess: OneHotEncoder, FloatEncoder, OrdinalEncoder
# ENCODING
df = DataFrame(:A => [5, 10, 15, 20], :B => ["test", "test2", "test2", "test"])
println("ENCODING")
println("........")
@testset "OneHotEncoder" begin
encoder = OneHotEncoder()
ant = size(df)[2]
data = encoder.predict(df, :A)
len = Set(df[!, :A])
@test size(data)[2] - ant == length(len)
end
