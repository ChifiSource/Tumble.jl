# Lathe.jl
Lathe is a conclusive module for Data Science in Julia, with statistics capabilities, data processing capabilities, and model training and validation capabilities.
### Using Lathe
As of right now, Lathe.jl isn't necessarily ready for use. Feel free to use it if you would like, just know that some functions might be experimental or downright unusable. Feel free to fork and submit pull requests!
# Lathe.stats
#### Lathe.stats.mean(array)
Mean calculates the average of an array, giving a generalized idea of what our data is like.
```julia
array = (1,4,5,6)
m = Lathe.stats.mean(array)
println(4)
julia > 4
```
#### Lathe.stats.standardize(array)
Returns the standard deviation of an array (**Note:** This does not standardize the array, but returns the standardized deviation of the array, for standard scaling see **Lathe.preprocess.standardscale**
#### Lathe.stats.expo(number,exponent)
Returns an integer or float to an exponential power
```julia
number = 5
five_squared = Lathe.stats.expo(5,2)
```
#### Lathe.stats.t_test
#### Lathe.stats.bay_ther
#### Lathe.stats.confint
# Lathe.preprocess
# Lathe.model
