from ConfigSpace import ConfigurationSpace

configspace = ConfigurationSpace(
    name="HyperSpace",
    space={
        "lr": (0.0001, 0.1),
        "epochs": (5, 25),
        "batch_size": (16, 256),
        "input_size": 90,
        "output_size": 1,
        "hidden_size": (32, 512),
        "hidden_layers": (3, 10),
        "criterion": ["MSELoss", "HuberLoss"],
        "optimizer": ["Adam", "AdamW"],
    },
)

configspace_new = ConfigurationSpace(
    name="HyperSpace",
    space={
        "lr": (0.0001, 0.1),
        "epochs": (5, 25),
        "batch_size": (16, 256),
        "input_size": 400,
        "output_size": 1,
        "hidden_size": (32, 512),
        "hidden_layers": (2, 20),
        "criterion": ["SoftMarginLoss", "BCELoss"],
        "optimizer": ["Adam", "AdamW"],
        "device": "cpu",
    },
)
