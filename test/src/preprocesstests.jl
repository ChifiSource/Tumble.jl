using Lathe.preprocess: SVD, OneHotEncoder
@testset "svd test for $T" for T in (Float32, Float64)
  A = randn(T, 100, 100)
  x = SVD(A)

  tol = T == Float32 ? 1e-5 : 1e-10
  @test (norm(A - (x.U * diagm(0 => x.S) * x.V'), 2) / norm(A, 2)) <= tol
end

df = DataFrame(:A => ["A", "C", "A"], :B => [5,10,15])

@testset "Encoder Tests" begin
encoder = OneHotEncoder()
@test encoder.predict(df)
end
