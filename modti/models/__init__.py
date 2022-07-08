from modti.models.modular import Modular
from modti.models.monolithic import Monolithic


def get_model(name, **kwargs):
    if name == "Modular":
        return Modular(**kwargs)
    elif name == "Monolithic":
        return Monolithic(**kwargs)
    else:
        raise ValueError("Unknown type of model")