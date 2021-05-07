println("LATHE.MODELS TESTS")
println("......................")
using Lathe.models: LinearLeastSquare, LinearRegression
x = [5, 10, 15, 20]
y = [5, 10, 15, 20]
@testset "Regressors" begin
m = LinearRegression(x, y)
testy = [5, 10, 15]
yhat = m.predict(testy)
@test yhat == testy
m = LinearLeastSquare(x, y)
yhat = m.predict(testy)
@test yhat == testy
end
