## What is Lathe?
Lathe.jl is an all-in-one package for predictive modeling in Julia. It comes packaged with a Stats Library, Preprocessing Tools, Distributions, Machine-Learning Models, and Model Validation. Lathe features easy object-oriented programming methodologies using Julia's dispatch.

```@contents
```

## Adding Lathe
Lathe can be added using Pkg.
```@repl
using Pkg
Pkg.add("Lathe")
```
### Git Repo
Lathe can also be added directly from the Github repo.
```@example
julia> ]
pkg> add https://github.com/emmettgb/Lathe.jl/
```

### Unstable
If you desire, you can also add the unstable branch from Lathe's Github repo.
 This will provide the advantage of newer features, but **the module may be unstable**

```@example
 julia> ]
 pkg> add Lathe #Unstable
```

### Versions
Starting with the release of Lathe Butterball (0.1.0), you can add any version of Lathe.

```@example
julia> ]
pkg> add Lathe#0.1.0
pkg> add Lathe#0.1.1
```
You can also add the final release of a specific version of Lathe using version names.

```@example
julia> ]
pkg> add Lathe#Butterball
```
