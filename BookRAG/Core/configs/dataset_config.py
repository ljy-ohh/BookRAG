from pydantic import BaseModel, Field
import yaml


class DatasetConfig(BaseModel):
    dataset_path: str = Field(..., description="JSON 数据集文件的路径。")
    working_dir: str = Field(..., description="项目的工作目录。")
    dataset_name: str


def load_dataset_config(path: str) -> DatasetConfig:
    # ... standard YAML loading logic ...
    with open(path, "r") as f:
        data = yaml.safe_load(f)

    data_cfg = DatasetConfig(**data)
    return data_cfg
