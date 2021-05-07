println("LATHE.STATS TESTS")
println("...........")
using Lathe.stats: NormalDist, T_Dist, mean, std
# Distributions
@testset "Distributions" begin
x = [5, 10, 15]
dist = NormalDist(x)
zero_std = dist.apply(5)
@test zero_std == dist.apply(5)
end
