import deeplay as dl
from torch.nn import Sigmoid, MSELoss

mlp_template = dl.MultiLayerPerceptron(
    in_features=28*28, hidden_features=[32, 32], out_features=10,
)
mlp_template[..., "activation"].configure(Sigmoid)
mlp_model = mlp_template.create()

# print(mlp_model)
print(f"{sum(p.numel() for p in mlp_model.parameters())} trainable parameters")

classifier_template = dl.Classifier(
    model=mlp_template, num_classes=10, make_targets_one_hot=True,
    loss=MSELoss(), optimizer=dl.SGD(lr=.1)
)
classifier = classifier_template.create()
# print(classifier)

