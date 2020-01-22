<div align="center"><img src="http://emmettboudreau.com/Lathe/logo.png" />
  <h1>Lathe 0.0.9<h1>
    <h4>Easily ML<h4>
      <h3>Documentation<h3>
    <a href="http://lathe.ai/doc.html"><img src="https://cdn3.iconfinder.com/data/icons/documentation-blue-red/60/057_-_Document_Info-512.png" width="82" height="86" title="Documentation" alt="Documentation"></a>
</div>
        <h4>For information on release canvases, visit the projects tab</h4>
<div align="left">
  <p> Lathe.jl is an all-in-one package for predictive modeling in Julia. It comes packaged with a Stats Library, DataFrame tools, Preprocessing, Machine-Learning Models, and Model Validation. Lathe features easy one line constructor model fit-predicting, fast dispatch, easy documentation, and a great code-base with strong deployability.</p>
        </div>
        </div>


**(Note that documentation is most up-to-date and well-written inside of the module, in Julia you can use ?() on any Lathe function to find out what said function does.)**
```julia
help?> Lathe.stats.mean
  Calculates the mean of a given array.

  ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

  array = [5,10,15]

  mean = Lathe.stats.mean(array)

  println(mean)

  10

```
**Attention! :** Lathe is moving! We'll be moving to http://lathe.ai starting in Lathe 0.1.0 "Butterball", Lathe will be expanded with a wrapper using JuliaCall in both Python, and R. This will allow a wrapper to be created while still technically using Julia as a backend, controlled from the languages we all know and love. Additionally, Lathe is going to be ported to common Lisp! **Lathe is being ported to Lisp, Python, and R** because of familiarity and cross-platform domain. Starting version 0.1.0, you will be able to use Lathe in **4 different languages**.
## Features
- Pipelines
- An ever-expanding library of models for predicting continous features, and soon categorical features.
- Easy! Machine learning is simplified! Lathe makes it possible to fit and predict in just two lines!
- Internal documentation - use the ?() function on anything and get all the information you need.
- Stats library with distributions, tests, and base statistics.
- Easy Train Test Split in one line.
- Simplified Preprocessing, with most scalars being simple
- Expandable, dependable, and reliable. Lathe has a big advantage in being written in Julia. Many functions, notably the prediction function utilize Julia's function struct dispatch syntax, allowing end users to quickly modify the function in their environment to make it work for them. This also adds to the long term support, as anyone could develop a model for Lathe, even entirely seperately using Lathe as a code-base to work off of.
- Serialization for pipelining, using one prediction function.
- A small, and expanding distributions library.
## Add it!
**LTS** 
 ```julia
 using Pkg; Pkg.add("Lathe")
 ```
 **Unstable** 
 ```julia
 using Pkg; Pkg.add("Lathe"#Unstable)
 ```
 The Unstable version of Lathe is typically updated daily, and may not work as planned from time to time. With that in mind, most users should likely stick to the LTS, as Unstable is for development and prototyping. There are still advantages to using Unstable, though, as you will have access to the latest features often weeks before the canvas is merged to LTS. Expect to run Pkg.update() a lot on the Unstable branch. \
 ## Contributors
 Thank you for considering contributing. Information on commit eticate is available inside of the Wiki section (and that's its only purpose!
# Thank you, everyone for your commits and stars! And thank you for using Lathe! 
