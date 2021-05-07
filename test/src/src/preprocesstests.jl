using LinearAlgebra
using DataFrames
using Test
println("LATHE.PREPROCESS TESTS")
println("......................")
# DECOMPOSITION
using Lathe.preprocess: OneHotEncoder, FloatEncoder, OrdinalEncoder
# ENCODING
df = DataFrame(:A => [5, 10, 15, 20], :B => ["test", "test2", "test2", "test"])
@testset "OneHotEncoder Tests" begin
encoder = OneHotEncoder()
ant = size(df)[2]
data = encoder.predict(df, :A)
len = Set(df[!, :A])
@test size(data)[2] - ant == length(len)

end
@testset "OrdinalEncoder Tests" begin
encoder = OrdinalEncoder(df[!, :B])
encoded = encoder.predict(df[!, :B])
@test encoded[1] == 1
@test encoded[2] == 2
@test encoded[3] == 2
end
@testset "FloatEncoder Tests" begin
  encoder = FloatEncoder()
  encoded = encoder.predict(df[!, :B])
  @test encoded[1] == sum([c = Float64(c) for c in "test"])
end
