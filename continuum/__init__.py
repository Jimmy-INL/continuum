import abc
from enum import Enum
from pydantic import BaseModel


class ExtraResp(Enum):
    IGNORE = "ignore"
    ALLOW = "allow"
    FORBID = "forbid"


class Foundation(BaseModel, abc.ABC):
    class Config:

        extra = ExtraResp.ALLOW.value
        use_enum_values = True
        arbitrary_types_allowed = True

    # def __init__(
    #     self, extra_resp: ExtraResp = ExtraResp.ALLOW, **data
    # ) -> None:
    #     self.Config.extra = extra_resp.value
    #     super().__init__(**data)
