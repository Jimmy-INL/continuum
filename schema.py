from enum import Enum
from _pytest.config import Config
from pydantic import BaseModel, BaseConfig, Field
from loguru import logger
from typing import Dict, AnyStr, Any, Optional, Type, Union, List
import json
from pydantic.schema import field_schema
from devtools import debug
import re
from pydantic import BaseModel

# from pydantic.fields import ModelField
from toolz import get_in
from pydantic import *
from pydantic.utils import *
from pydantic.validators import *
# from pydantic.fields import ModelField

# https://en.wikipedia.org/wiki/Postcodes_in_the_United_Kingdom#Validation


def remove_title(schema: dict):
    for prop in schema.get('properties', {}).values():
        prop.pop('title', None)


def extract_props(schema: dict):
    return schema.get("properties")


def get_default(schema: dict, name: str):
    return get_in(['properties', name, 'default'], schema)


class TitleRemover(BaseModel):
    class Config:
        @staticmethod
        def schema_extra(
            schema: Dict[str, Any], model: Type['TraitsLinter']
        ) -> None:
            remove_title(schema)


class TypeSet(str, Enum):
    boolean = "boolean"
    string = "string"
    float = "float"
    set = "set"
    union = "union"
    range = "range"
    nullable = "nullable"
    obj = "object"
    external = "external"


class Range(BaseModel):
    _min: int = Field(..., alias='min')
    _max: int = Field(..., alias='max')


class KeySchema(BaseModel):
    key: str
    description: Optional[str]
    datatype: Optional[List['TypeSchema']] = None


class TypeSchema(BaseModel):

    type: TypeSet = Field(
        None, title="type", description="This is one of the types we can set."
    )
    innerType: Optional['TypeSchema'] = None

    value: Optional[Any] = None

    name: Optional[str] = None

    _fields: Optional[List[KeySchema]] = Field(None, alias="fields")

    class Config:
        title = "datatype"
        extra = "allow"
        arbitrary_types_allowed = True


# KeySchema.update_forward_refs()
TypeSchema.update_forward_refs()


class NameDesc(BaseModel):
    name: str = Field(None, description="The name of the item in question.")
    description: Optional[Union[str, dict]] = Field(
        None,
        description="The description of the field you're trying to enter."
    )


class GeneralFields(NameDesc):
    datatype: TypeSchema = Field(
        None,
        description=
        "The type of data of a given item (property, state, or action)"
    )


class State(GeneralFields):
    class Config:
        title = "state"

        @staticmethod
        def schema_extra(schema: Dict[str, Any], model: Type['State']) -> None:
            remove_title(schema)


class Parameter(GeneralFields):
    required: bool = False


class Validation(GeneralFields):
    expression: Optional[List[Any]] = []


class Action(NameDesc):
    """ An Action To Act On The Trait"""
    parameters: List[Parameter] = []
    validations: List[Validation] = []

    # effects:

    class Config:
        title = "action"

        @staticmethod
        def schema_extra(schema: Dict[str, Any], model: Type['State']) -> None:
            remove_title(schema)


class TraitsLinter(GeneralFields):
    """
        A traits linter and autocompletion.
    """

    type: str = "trait"

    states: List[State] = []
    actions: List[Action] = []
    properties: List[Parameter] = []

    class Config:
        title = "Traits Yaml Linter"

    # @staticmethod
    # def schema_extra(
    #     schema: Dict[str, Any], model: Type['TraitsLinter']
    # ) -> None:
    # schema['$schema'] = "http://json-schema.org/draft-07/schema#"
    # # schema.update(**{"$schema": "http://json-schema.org/draft-07/schema#"})

    # props = schema.get('properties', {})
    # for prop in props.values():
    #     prop.pop('title', None)

    # snippets = {
    #     "prop": {
    #         "defaultSnippets": [{
    #             "label":
    #             "New property",
    #             "description":
    #             "Creates a new property to describe a state.",
    #             "body":
    #             "name: ${1}\ndatatype:\n\ttype:${2}required:${3}description:${description}"
    #         }],
    #     }
    # }
    # props.update(**snippets)


if __name__ == "__main__":
    linter = TraitsLinter()
    lnt = linter.schema(by_alias=True)
    lnt['$schema'] = "http://json-schema.org/draft-07/schema#"
    snippets = {
        "prop": {
            "defaultSnippets": [{
                "label":
                "New property",
                "description":
                "Creates a new property to describe a state.",
                "body":
                "\nname: ${1}\ndatatype:\n\ttype:${2}\n\trequired:${3|true, false|}\n\tdescription:${desc:A basic description}"
            }],
        }
    }
    x = {
        "title":
        'TypeSet',
        "type":
        "string",
        'description':
        'A set of available types for the type function.',
        "oneOf": [
            {
                "const": "boolean",
                "description": "A boolean type"
            },
            {
                "const": "int",
                "description": "A integer type"
            },
            {
                "const": "string",
                "description": "A string type"
            },
            {
                "const": "float",
                "description": "A float type"
            },
            {
                "const": 'set',
                "description": "A set type"
            },
            {
                "const": 'union',
                "description": "A union type"
            },
            {
                "const": 'range',
                "description": "A range type"
            },
            {
                "const": 'nullable',
                "description": "nullable values"
            },
            {
                "const": 'object',
                "description": "An object type"
            },
            {
                "const": 'external',
                "description": "An external type"
            },
        ]
    }
    lnt['definitions']['TypeSet'] = x
    logger.success(lnt.get('definitions'))
    lnt['properties'].update(**snippets)

    with open('type-trait.json', 'w+') as fo:

        fo.write(json.dumps(lnt))
    # logger.error(lnt)
    # logger.success(dict(lnt['definitions']['DataTypes']['enum'][0]))