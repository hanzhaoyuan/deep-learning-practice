import deeplay as dl
import torch.nn as nn

# mlp_template = dl.MultiLayerPerceptron(
#     in_features=28*28, hidden_features=[32, 32], out_features=10,
# )
# mlp_template[..., "activation"].configure(Sigmoid)
# mlp_model = mlp_template.create()

# print(mlp_model)
linear_layer = dl.Layer(nn.Linear, in_features=10, out_features=5)
print(linear_layer)
