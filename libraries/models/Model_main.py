"""
Factory class to create the model object based on the model name.

@author: Alexander Garzón
@email: j.a.garzondiaz@tudelft.nl
"""

from libraries.models.MLP_model import *
from libraries.models.GNN_model import *


def ModelFactory(model_name):
    available_models = {
        "MLP_Benchmark_metamodel": MLP_Benchmark_metamodel,
        "NN_GINEConv_NN": NN_GINEConv_NN,
    }
    model = available_models[model_name]
    return model
