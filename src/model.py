from flax import nnx

class ConstructionNN(nnx.Module):
    def __init__(self, num_params, num_output, rngs: nnx.Rngs):
        # Using 64 neurons to handle 10 output targets
        self.linear1 = nnx.Linear(num_params, 64, rngs=rngs)
        self.linear2 = nnx.Linear(64, 64, rngs=rngs)
        self.linear3 = nnx.Linear(64, num_output, rngs=rngs)

    def __call__(self, x):
        x = nnx.tanh(self.linear1(x)) # Tanh is more stable than ReLU for this data
        x = nnx.tanh(self.linear2(x))
        x = self.linear3(x)
        return x
