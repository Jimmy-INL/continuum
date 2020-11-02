from pydantic.main import BaseModel


class AllowedBase(BaseModel):
    class Config:
        extra = 'allow'
        arbitrary_types_allowed = True
