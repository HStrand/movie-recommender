using PyPlot
using DataFrames
using HDF5

function SGD(R,Theta,X,K,epochs=1000000,alpha=0.0002,beta=0.02)
    errors = []
    ploterrors = []
    X = transpose(X)
    for iter = 1:epochs
        if iter%10000 == 0
            println("Starting iteration ", iter+1)
            emean = mean(errors[end-9998:end])
            println("Moving average error: ", emean)
            push!(ploterrors, emean)
        end
        e = 0
        rnd = rand(1:maximum(R[1]))
        i = R[2][rnd]
        j = R[3][rnd]
        realrating = R[4][rnd]

        eij = realrating - dot(vec(Theta[i,:]),vec(X[:,j]))
        Theta[i,:] = vec(Theta[i,:]) + alpha*(2*eij*vec(X[:,j]) - beta*vec(Theta[i,:]))
        X[:,j] = vec(X[:,j]) + alpha*(2*eij*vec(Theta[i,:]) - beta*vec(X[:,j]))

        e += (realrating - dot(vec(Theta[i,:]),vec(X[:,j])))^2
        push!(errors,e)
    end

    return Theta, transpose(X), ploterrors
end


println("Loading HDF5 file...")
r = h5read("c://Dev//Code//movie-recommender//ratings.h5", "r100k")
println("Loading to data frame...")
R = DataFrame(index=r["axis1"]+1,userId=vec(r["block1_values"][1,:]),movieId=vec(r["block1_values"][2,:]),rating=vec(r["block0_values"]),timestamp=vec(r["block1_values"][3,:]))

N = maximum(R[2])
M = maximum(R[3])
K = 10

println("Generating random user features...")
Theta = rand(N,K)
println("Generating random movie features...")
X = rand(M,K)

println("Starting Stochastic Gradient Descent...")

tic()
nTheta, nX, errors = SGD(R,Theta,X,K)
toc()

nR = nTheta*transpose(nX)

# error = 0
# count = 0
# for i in 1:10
#     for j in 1:10
#         if R[i,j] > 0
#             error += (R[i,j] - nR[i,j])^2
#             count += 1
#         end
#     end
# end
# println("Mean squared error: ", error/count)

predict = dot(vec(nTheta[1,:]),vec(transpose(nX)[:,16]))

plot(errors)
