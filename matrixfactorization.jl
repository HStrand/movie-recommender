using PyPlot

function SGD(R,Theta,X,K,epochs=1000000,alpha=0.0002,beta=0.02)
    errors = []
    X = transpose(X)
    for iter = 1:epochs
        e = 0
        i = rand(1:10)
        j = rand(1:10)
        if R[i,j] == 0
            continue
        end

        eij = R[i,j] - dot(vec(Theta[i,:]),vec(X[:,j]))
        Theta[i,:] = vec(Theta[i,:]) + alpha*(2*eij*vec(X[:,j]) - beta*vec(Theta[i,:]))
        X[:,j] = vec(X[:,j]) + alpha*(2*eij*vec(Theta[i,:]) - beta*vec(X[:,j]))

        e += (R[i,j] - dot(vec(Theta[i,:]),vec(X[:,j])))^2
        push!(errors,e)
    end

    return Theta, transpose(X), errors
end

R = [
    5 3 0 1 0 3 5 0 2 0;
    4 0 0 1 5 4 0 0 2 2;
    1 1 0 5 1 3 5 5 0 0;
    1 0 0 4 0 0 0 2 3 1;
    0 1 5 4 4 4 2 3 0 1;
    0 3 2 4 0 5 0 1 1 1;
    4 5 2 0 2 5 0 1 0 0;
    5 3 0 0 3 2 1 4 0 4;
    0 0 0 5 0 0 5 5 0 5;
    0 5 3 3 0 0 0 0 3 5
]

N = length(R[:,1])
M = length(R[1,:])
K = 5

Theta = rand(N,K)
X = rand(M,K)

tic()
nTheta, nX, errors = SGD(R,Theta,X,K)
toc()

nR = nTheta*transpose(nX)

error = 0
count = 0
for i in 1:10
    for j in 1:10
        if R[i,j] > 0
            error += (R[i,j] - nR[i,j])^2
            count += 1
        end
    end
end
println("Mean squared error: ", error/count)

predict = dot(vec(nTheta[1,:]),vec(transpose(nX)[:,1]))

plot(errors)
