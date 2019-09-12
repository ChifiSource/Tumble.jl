# Lathe.jl
Lathe is a conclusive module for Data Science in Julia, with statistics capabilities, data processing capabilities, and model training and validation capabilities. \
This package intends to fill the one package gap for predictive modeling, for example, Python has Sklearn then Tensorflow, but Julia has several libraries that you have to install, import, just to encode, impute, or predict data. \
Lathe includes a statistics module, for basic mathematical calculations, inferential, and bayesian statistics. Lathe also includes preprocessing algortithms that will help your models to interpret your data. Lathe also includes predictive machine learning models, which allow you to make predictions based off of data, and use a computer to infer variables within DataFrames and matrices.
### Using Lathe
For now, using Lathe is not recommended for anything but statistical purposes, as the other modules are still Work In Progress (WIP). \
However, if you would still like to use Lathe, you can Pkg.add it, or use it using push!. \
Firstly, you will need to open a terminal in the folder where Lathe.jl is located \
```julia
~/Projects/JL/Libraries/Lathe$ julia
               _
   _       _ _(_)_     |  Documentation: https://docs.julialang.org
  (_)     | (_) (_)    |
   _ _   _| |_  __ _   |  Type "?" for help, "]?" for Pkg help.
  | | | | | | |/ _` |  |
  | | |_| | | | (_| |  |  Version 1.0.4 (2019-05-16)
 _/ |\__'_|_|_|\__'_|  |  Official https://julialang.org/ release
|__/                   |


julia >
```
Next, you will need to run push!(LOAD_PATH, pwd()), this will update your current path, and add a file registry into the Julia backend.
```julia
#julia > push!(LOAD_PATH, pwd())
4-element Array{String,1}:
 "@"                                       
 "@v#.#"                                   
 "@stdlib"                                 
 "/home/emmett/Projects/JL/Libraries/Lathe"
#julia >
```
And Finally, you can type "using Lathe" to import the package into your Julia command line. To test, you can do something like:
```julia
#julia > using Lathe
[ Info: Recompiling stale cache file /home/emmett/.julia/compiled/v1.0/Lathe.ji for Lathe [top-level]

#julia > array = (8,3,7,5)
(8, 3, 7, 5)
#julia > Lathe.stats.mean(array)
5.75
#julia >
```
Additionally, you can use submodules of Lathe using:
```julia
#julia > using Lathe: stats
#julia > stats.mean(array)
5.75
```
### Adding Lathe
In order to Pkg.add("") Lathe, in order to use it with system wide and with Conda, we will have to use Pkg, which is Julia's built in Package manager. If you have julia, you will have Pkg, unlike Python pip, Pkg comes with Julia. The first step is to enter into the pkg REPL (although this can also be done using Using Pkg, as will be discussed below)
```julia
#julia > ]
#pkg > add https://github.com/emmettgb/Lathe.jl
   Cloning git-repo `https://github.com/emmettgb/Lathe.jl`
  Updating git-repo `https://github.com/emmettgb/Lathe.jl`
[ Info: Assigning UUID 78e343b4-76a6-52db-88f3-27113f9a8314 to Lathe
 Resolving package versions...
 Installed Mocking ─ v0.7.0
  Updating `~/.julia/environments/v1.0/Project.toml`
  [78e343b4] + Lathe v0.0.0 #master (https://github.com/emmettgb/Lathe.jl)
  Updating `~/.julia/environments/v1.0/Manifest.toml`
  [78e343b4] + Lathe v0.0.0 #master (https://github.com/emmettgb/Lathe.jl)
  [78c3b35d] ↑ Mocking v0.6.0 ⇒ v0.7.0
```
We can also do the same using the julia REPL using the Pkg module:
```julia
#julia > using Pkg
#julia > Pkg.add("https://github.com/emmettgb/Lathe.jl")
```
**Note:** *The same can be done inside of Jupyter* \
In order to update Lathe we can call Pkg.update(), and the update will automatically be downloaded and installed for Julia. We can do this using Pkg, or by intering into the Pkg REPL by pressing ]
**REPL**
```julia
#julia > ]
#pkg > update()
Updating registry at `~/.julia/registries/General`
Updating git-repo `https://github.com/JuliaRegistries/General.git`
Updating git-repo `https://github.com/emmettgb/Lathe.jl`
.......
```
**Julia REPL using Pkg**
```julia
#julia > using Pkg
#julia > Pkg.update
Updating registry at `~/.julia/registries/General`
Updating git-repo `https://github.com/JuliaRegistries/General.git`
Updating git-repo `https://github.com/emmettgb/Lathe.jl`
.......
```
## Standard Library
#### Lathe.stats.variance(array)
Returns the variance of an array.
```julia
array = (8,5,4,8)
var = Lathe.stats.standardize(array)
println(var)
julia > 3.1875
```
#### Lathe.stats.confint()
#### Lathe.stats.standardize(array)
Returns the standard deviation of an array (**Note:** This does not standardize the array, but returns the standardized deviation of the array, for standard scaling see **Lathe.preprocess.standardscale**
```julia
array = (8,5,4,8)
std = Lathe.stats.standardize(array)
println(std)
julia > 1.785357107
```
#### Lathe.stats.expo(number,exponent)
Returns an integer or float to an exponential power
```julia
number = 5
five_squared = Lathe.stats.expo(5,2)
println(five_squared)
julia > 25
```
## Inferential Statistics
#### Lathe.stats.student_t(Sample, General)
Returns a p value, used to detect statistical significance between a variable in a population versus μ.
```julia
julia> using Lathe: stats

julia> stats.t_test(array,arrayt)
-0.00554580351562448

julia> using Lathe: stats

julia> stats.t_test(array,arrayt)
-0.00554580351562448

```
#### Lathe.stats.f_test(Sample, General)
Returns an f value, used to detect statistical significance between a variable in a population versus μ.

#### Lathe.stats.inf_sum
Returns a detailed summary of statistics between two given groups: \
```julia
#julia> using Lathe: stats
[ Info: Recompiling stale cache file /home/emmett/.julia/compiled/v1.0/Lathe.ji for Lathe [top-level]
        
#julia> array1 = 5,7,4,5
(5, 7, 4, 5)

#julia> array2 = 18,13,11,9
(18, 13, 11, 9)

#julia> stats.inf_sum(array1,array2)
================
     Lathe.stats Inferential Summary
     _______________________________
: 12.75
N: 4
x̅: 5.25
μ: 12.75
s: 248.0625
σ: 1463.0625
var(X): 61535.00390625
σ2: 2.14055187890625e6
α -0.010252466999871843
Fp: 0.02874726116785
================

```
**Note:** This output is subject to change without documentation updates
## Bayesian Statistics
#### Lathe.stats.bay_ther(Probability, Prior, Evidence)
Returns a percentage, takes 3 integers. Calculates probability using Bayesian inference.
```julia
julia> stats.bay_ther(55,50,5)
30250.0
```
# Lathe.preprocess
# Lathe.model
