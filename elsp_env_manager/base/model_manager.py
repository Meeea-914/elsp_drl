import json
import os


def load_model(model_name: str, env_path=None):
    if env_path is not None:
        if not model_name.endswith(".json"):
            model_name = model_name + ".json"
        assert model_name in os.listdir(env_path)
        model_file = open(os.path.join(env_path, model_name))
    else:
        model_file = open(model_name)
    model = json.loads(model_file.read())
    model_file.close()
    return model

