import yaml


class YamlConfig:
    def __init__(self, yaml_file):
        with open(yaml_file, "r") as f:
            data = yaml.safe_load(f)
        for key, value in data.items():
            setattr(self, key, value)


if __name__ == "__main__":
    pass
