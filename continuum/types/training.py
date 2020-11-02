from pydantic.main import BaseModel


class TrainingParameters(BaseModel):
    epochs: int = 10
    num_tasks: int = 7
    num_classes: int = 500
    state_fields = ["state"]
    reward_fields = ["reward"]
    window: int = 10
    window_two: int = 3