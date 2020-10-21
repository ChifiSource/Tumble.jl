#===============================
<-----------Lathe.jl----------->
~~~~~~ 0.1.4 "ButterBall" ~~~~~~
Programmed by Emmett Boudreau
    <emmett@emmettboudreau.com>
        <http://emmettboudreau.com>
MIT General Open Source License
    (V 3.0.0)
        Free for Modification and
        Redistribution
=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=|
         CONTRIBUTORS
=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=|
        ~ emmettgb
        ~ stefanches7
        ~ PallHaraldsson
/><><><><><><><><><><><><><><><><\
Thank you for your forks!
<-----------Lathe.jl----------->
38d8eb38-e7b1-11e9-0012-376b6c802672
#[deps]
DataFrames.jl
Random.jl
================================#
@doc """
      |====== Lathe - Easily ML =====\n
      |= = = = = v. 0.1.4 = = = = = |\n
      |==============================\n
      |__________Lathe.stats\n
      |__________Lathe.preprocess\n
      |__________Lathe.models\n
      |______________________________\n
      Use ?(Lathe.package) for documentation.\n
      [uuid]\n
      38d8eb38-e7b1-11e9-0012-376b6c802672\n
      [deps]\n
      DataFrames.jl\n
      Random.jl\n
       """
module Lathe
#
# <------- DEPS ----->
using DataFrames
using Random
# <------- DEPS ----->
#
# <------- PARTS ----->
include("Stats.jl")
include("Models.jl")
include("Preprocess.jl")
# <------- PARTS ----->
#
end
