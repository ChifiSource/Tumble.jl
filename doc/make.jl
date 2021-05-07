using Documenter
using Lathe: models, stats, preprocess
using Lathe
Documenter.makedocs(root = "./",
       source = "src",
       build = "build",
       clean = true,
       doctest = true,
       modules = Module[Lathe,models,stats,preprocess],
       repo = "https://github.com/emmettgb/Lathe.jl/",
       highlightsig = true,
       sitename = "Lathe.jl",
       expandfirst = [],
       pages = [
               "Index" => "index.md",
               "Stats" => "stats.md",
               "Preprocess" => "preprocess.md",
               "Models" => "models.md"
               ]
       )
