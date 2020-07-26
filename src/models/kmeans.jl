function Kmeans(data, k)
  epsilon = 1e-6
  losses = Float64[]
  N = length(data)
  n = length(data[1])
  centroids, labels = initialize_centroids(data, k)
  while true
    labels = partition_data(data, centroids)
    centroids = update_centroids(data, labels, k, centroids)
    push!(losses, loss(data,centroids,labels))
    if length(losses) >= 2 && abs(losses[end] - losses[end-1]) <= epsilon
      break
    end
  end
  return centroids, labels, losses
end

function partition_data(data, centroids)
  labels = Int[]
  for i in 1:length(data)
    centroid_distances = pairwise_distance(data[i], centroids)
    distance, centroid_index = findmin(centroid_distances)
    push!(labels, centroid_index)
  end
  return labels
end
function update_centroids(data, labels, k, old_centroids)
  centroids = Vector[]
  for i = 1:k
    centroid_pts = data[labels .== i]
    if length(centroid_pts) > 0
      push!(centroids, mean(centroid_pts))
    else
      push!(centroids, old_centroids[i])
    end
  end
  return centroids
end

function loss(data, centroids, labels)
  N = length(data)
  sos_dist = 0.0
  for i = 1:N
    sos_dist += norm(data[i] - centroids[labels[i]])^2
  end
  return sos_dist*(1/N)
end
function initialize_centroids(data, k)
  N = length(data)
  n = length(data[1])
  labels = rand(1:k,N)
  centroids = update_centroids(data, labels, k, zeros(n) for i=1:k)
  return centroids, labels
end

function pairwise_distance(vector, vectors)
  return Float64[norm(vector - vectors[i]) for i=1:length(vectors)]
end
