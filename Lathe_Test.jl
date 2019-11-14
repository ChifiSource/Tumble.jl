include("Lathe.jl/src/Lathe.jl")
import .Lathe.stats: z

using Test

"""
Stats
"""
# z sanity check
@test size(Lathe.stats.z([1 2 3]), 2) == 3
@test Lathe.stats.z([1 2 3])[2] == 0
