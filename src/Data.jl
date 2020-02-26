#=====
Serialization
	Tools
=====#
module data
struct VectorDataset{A,B} <: Dataset
    x::A
    y::B
end

Base.length(ds::VectorDataset) = length(ds.x)
Base.getindex(ds::VectorDataset, idx) = return (ds.x[idx], ds.y[idx])

struct JLDDataset <: Dataset
	f
    keys

	JLDDataset(f) = new(f,keys(f))
end

Base.length(ds::JLDDataset) = length(ds.keys)

function Base.getindex(ds::JLDDataset, idx)
	key = ds.keys[idx]
	ds.f[key]
end
Base.length(ds::JuliaDBDataset) = length(ds.filenames)

struct ImageDataset <: Dataset
    filenames::Vector{String}
    labels::Vector
	resize::Union{Nothing,Tuple}

	function ImageDataset(filenames, labels; resize=nothing)
		@assert length(filenames) == length(labels)
		try
			@eval import Images
			@eval import ImageMagick
		catch
			@warn "Package Images or ImageMagick not installed"
		end
		new(filenames, labels, resize)
	end

end

Base.length(ds::ImageDataset) = length(ds.filenames)

function Base.getindex(ds::ImageDataset, idx)
	filename = ds.filenames[idx]
	# img = Images.load(filename) has multi-thread issue
    img = ImageMagick.load(filename)
	img = Images.channelview(img)
	img = permutedims(img, [2,3,1])
	img = convert(Array{Float32}, img)
	if ds.resize !== nothing
		img = Images.imresize(img, ds.resize...)
	end
	return (img, ds.labels[idx])
end

struct DFDataset <: Dataset
	df
    X::Vector{Symbol}
	Y::Vector{Symbol}

	DFDataset(df,X,Y) = new(df, makeArray(X), makeArray(Y))
end

Base.length(ds::DFDataset) = size(df,1)

function Base.getindex(ds::DFDataset, idx)
	df = ds.df
	(Vector(df[idx, ds.X]), Vector(df[idx, ds.Y]))
end
#==
TRANSFORMERS
==#
abstract type Transformer <: Dataset end

Base.length(t::Transformer) = length(t.ds)
function (t::Transformer)(ds::Dataset)
    t.ds = ds
    return t
end
mutable struct NoisingTransfomer <: Transformer
    ds::Union{Dataset, Nothing}
	noiselevel
	axis

    NoisingTransfomer(noiselevel=0.01; axis=[:X]) = new(nothing,noiselevel,axis)
end

genNoise(arr::AbstractArray, level::Number) = randn(eltype(arr), size(arr)) .* level

function Base.getindex(t::NoisingTransfomer, idx)
	X,Y = t.ds[idx]
	if :X in t.axis
		X += genNoise(X, t.noiselevel)
	end

	if :Y in t.axis
		Y += genNoise(Y, t.noiselevel)
	end

	return (X,Y)
end
mutable struct Normalizer <: Transformer
    ds::Union{Dataset, Nothing}
	dims::Array{Int}
	means
	stds
	axis

    Normalizer(means=0, stds=1, dims=[1,2]; axis=[:X]) =
		new(nothing,dims, means, stds, axis)
end

function normalize(x, means, stds)
	(x .- means) ./ stds
end

function Base.getindex(t::Normalizer, idx)
	X,Y = t.ds[idx]
	if :X in t.axis
		X += normalize(X, t.means, t.stds)
	end

	if :Y in t.axis
		Y += normalize(Y, t.means, t.stds)
	end

	return (X,Y)
end
mutable struct ImageCrop <: Transformer
    ds::Union{Dataset, Nothing}
	shapeX
	shapeY

    ImageCrop(shapeX , shapeY=nothing) = new(nothing,shapeX, shapeY)
end

function imageCrop(arr::AbstractArray,shape)
	maxX = size(arr,1) - shape[1]
	maxY = size(arr,2) - shape[2]
	offX = rand(1:maxX)
	offY = rand(1:maxY)
	arr[offX:offX+shape[1]-1,offY:offY+shape[2]-1, :]
end


function Base.getindex(t::ImageCrop, idx)
	X,Y = t.ds[idx]
	if t.shapeX !== nothing
		X = imageCrop(X, t.shapeX)
	end

	if t.shapeY !== nothing
		Y = imageCrop(Y, t.shapeY)
	end

	return (X,Y)
end



function onehot(x, labels, dtype::Type)
	result = zeros(dtype, length(labels))
    result[findfirst(x .== labels)] = 1
    result
end


function onehot(X::AbstractArray, labels, dtype::Type)
	result = zeros(dtype, length(labels))
	for x in X
		result[findfirst(x .== labels)] = 1
	end
	result
end

mutable struct OneHotEncoder <: Transformer
    ds::Union{Dataset, Nothing}
	labels
	axis
	dtype::Type

    OneHotEncoder(labels; axis=[:Y], dtype=getContext().dtype) =
		new(nothing, labels, axis, dtype)
end


function Base.getindex(t::OneHotEncoder, idx)
	X,Y = t.ds[idx]
	if :X in t.axis
		X = onehot(X, t.labels, t.dtype)
	end

	if :Y in t.axis
		Y = onehot(Y, t.labels, t.dtype)
	end

	return (X,Y)
end
mutable struct Subset <: Transformer
    ds::Union{Dataset, Nothing}
	idxs::AbstractArray{Int}

    Subset(idxs::AbstractArray{Int}) = new(nothing, idxs)
end

Base.length(t::Subset) = length(t.idxs)
Base.getindex(t::Subset, idx) = t.ds[t.idxs[idx]]

mutable struct Split <: Transformer
    ds::Union{Dataset, Nothing}
	valid_perc::Float64
	shuffle::Bool

    Split(valid_perc=0.2; shuffle=true) =
		new(nothing, valid_perc, shuffle)
end

function (t::Split)(ds::Dataset)
    t.ds = ds
	maxl = length(ds)

	idxs = t.shuffle ? Random.shuffle(1:maxl) : 1:maxl
	cut = round(Int, maxl*(1.0-t.valid_perc))

    a, b = (Subset(idxs[1:cut]), Subset(idxs[cut+1:end]))
	return (a(ds), b(ds))
end


#=
helper functions
=#
function update_mb!(arr::AbstractArray, elem::AbstractArray, idx::Int)
	@assert size(arr)[1:end-1] == size(elem) "$(size(arr)) $(size(elem))"
	idxs = Base.OneTo.(size(elem))
	arr[idxs..., idx] = elem
end


function update_mb!(t::Tuple, elems::Tuple, idx::Int)
	@assert length(t) == length(elems)
	for (arr,elem) in zip(t,elems)
		update_mb!(arr, elem, idx)
	end
end
create_mb(arr::AbstractArray, batchsize) = similar(arr, size(arr)..., batchsize)
create_mb(t::Tuple, batchsize)= Tuple(collect(create_mb(elem, batchsize) for elem in t))
mutable struct MiniBatch <: Transformer
    ds::Union{Dataset, Nothing}
    batchsize::Int
    shuffle::Bool

    MiniBatch(batchsize=8; shuffle=true) =
        new(nothing, batchsize, shuffle)
end

Base.length(dl::MiniBatch) = length(dl.ds) ÷ dl.batchsize

function Base.iterate(dl::MiniBatch, state=undef)
    maxl = length(dl.ds)
    bs = dl.batchsize

    if state == undef
        idxs = dl.shuffle ? Random.shuffle(1:maxl) : 1:maxl
        state = (idxs,1)
    end
    idxs, count = state

    if count > (maxl-bs) return nothing end

	l = Threads.SpinLock()
	minibatch = nothing

    Threads.@threads for i in 1:bs

		idx = i + count - 1
		sample = dl.ds[idx]
		@assert sample isa Tuple "Datasets should return Tuples, not $(typeof(sample))"

		if minibatch === nothing
			Threads.lock(l)
			if minibatch === nothing
				minibatch = create_mb(sample, bs)
			end
			Threads.unlock(l)
		end

		update_mb!(minibatch, sample, i)
    end
	Threads.unlock(l)
    return ((minibatch), (idxs, count + bs))
end
#------------------------------------------------------------
end
