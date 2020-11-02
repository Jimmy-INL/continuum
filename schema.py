from enum import Enum
from pydantic import BaseModel, Field
from loguru import logger
from typing import Dict, Any, Optional, Type, Union, List
import json
# from pydantic.schema import field_schema
# from devtools import debug
import re
from pydantic import BaseModel

# from pydantic.fields import ModelField
from toolz import get_in
from pydantic import *
from pydantic.utils import *
from pydantic.validators import *

import re
from typing import List, Union
from loguru import logger
from pydantic import BaseModel
from devtools import debug
import addict as adt


def remove_title(schema: dict):
    # schema = keyfilter(lambda x: x != "title", schema)
    for prop in schema.get('properties', {}).values():
        prop.pop('title', None)

    for prop in schema.get('definitions', {}).values():
        prop.pop('title', None)


def remove_title_hard(schema: dict):
    for prop in schema.values():
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


# https://en.wikipedia.org/wiki/Postcodes_in_the_United_Kingdom#Validation
post_code_regex = re.compile(
    r'(?:'
    r'([A-Z]{1,2}[0-9][A-Z0-9]?|ASCN|STHL|TDCU|BBND|[BFS]IQQ|PCRN|TKCA) ?'
    r'([0-9][A-Z]{2})|'
    r'(BFPO) ?([0-9]{1,4})|'
    r'(KY[0-9]|MSR|VG|AI)[ -]?[0-9]{4}|'
    r'([A-Z]{2}) ?([0-9]{2})|'
    r'(GE) ?(CX)|'
    r'(GIR) ?(0A{2})|'
    r'(SAN) ?(TA1)'
    r')'
)

reference_code_regex = re.compile(
    r'(?:^|(?<= ))'
    r'(property|state|parameter)'
    r'(@)'
    r'(\w+)$'
)


class ReferenceStr(str):
    """
    Partial UK postcode validation. Note: this is just an example, and is not
    intended for use in production; in particular this does NOT guarantee
    a postcode exists, just that it has a valid format.
    """
    @classmethod
    def __get_validators__(cls):
        # one or more validators may be yielded which will be called in the
        # order to validate the input, each validator will receive as an input
        # the value returned from the previous validator
        # logger.error(dir(cls))
        yield cls.validate

    @classmethod
    def __modify_schema__(cls, field_schema):
        # __modify_schema__ should mutate the dict it receives in place,
        # the returned value will be ignored

        field_schema.update(                                     # simplified regex here for brevity, see the wikipedia link above
            pattern='(?:^|(?<= ))(property|state|parameter)(@)(\w+)$',
            description="A reference refers to another property inside of the traits yaml. Must begin with (property | state | parameter), and the property must exist.",
            examples=['property@code', 'state@code', 'parameter@code'],
        )

    @classmethod
    def validate(cls, v):
        if not isinstance(v, str):
            raise TypeError("Reference should be a string.")
        m = reference_code_regex.fullmatch(v)
        if not m:
            # debug(m)
            raise ValueError(
                'Invalid reference format. The format must be (property|state|parameter)@(value_name). The value must exist in schema.'
            )
        # # you could also return a string here which would mean model.post_code
        # # would be a string, pydantic won't care but you could end up with some
        # # confusion since the value's type won't match the type annotation
        grps = m.groups()
        location, value = grps[0], grps[2]

        return cls(f'{location}@{value}')

    def __repr__(self):
        return f'ref({super().__repr__()})'


class Ref(TitleRemover):
    ref: ReferenceStr


GnUnion = Union[Ref, bool, int, float, str]


class Equal(TitleRemover):
    eq: List[GnUnion]


class Negate(TitleRemover):
    neq: Union[List[GnUnion], GnUnion]


class HardTitleRemover(BaseModel):
    class Config:
        @staticmethod
        def schema_extra(
            schema: Dict[str, Any], model: Type['TraitsLinter']
        ) -> None:
            remove_title_hard(schema)


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
    expression: Union[Equal, Ref, Negate]


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


def main():
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


if __name__ == "__main__":
    main()
    # definitions_printing()
    # linter = TraitsLinter()
    # lnt = linter.schema(by_alias=True)
    # lnt['$schema'] = "http://json-schema.org/draft-07/schema#"
    # snippets = {
    #     "prop": {
    #         "defaultSnippets": [{
    #             "label":
    #             "New property",
    #             "description":
    #             "Creates a new property to describe a state.",
    #             "body":
    #             "\nname: ${1}\ndatatype:\n\ttype:${2}\n\trequired:${3|true, false|}\n\tdescription:${desc:A basic description}"
    #         }],
    #     }
    # }
    # x = {
    #     "title":
    #     'TypeSet',
    #     "type":
    #     "string",
    #     'description':
    #     'A set of available types for the type function.',
    #     "oneOf": [
    #         {
    #             "const": "boolean",
    #             "description": "A boolean type"
    #         },
    #         {
    #             "const": "int",
    #             "description": "A integer type"
    #         },
    #         {
    #             "const": "string",
    #             "description": "A string type"
    #         },
    #         {
    #             "const": "float",
    #             "description": "A float type"
    #         },
    #         {
    #             "const": 'set',
    #             "description": "A set type"
    #         },
    #         {
    #             "const": 'union',
    #             "description": "A union type"
    #         },
    #         {
    #             "const": 'range',
    #             "description": "A range type"
    #         },
    #         {
    #             "const": 'nullable',
    #             "description": "nullable values"
    #         },
    #         {
    #             "const": 'object',
    #             "description": "An object type"
    #         },
    #         {
    #             "const": 'external',
    #             "description": "An external type"
    #         },
    #     ]
    # }
    # lnt['definitions']['TypeSet'] = x
    # logger.success(lnt.get('definitions'))
    # lnt['properties'].update(**snippets)

    # with open('type-trait.json', 'w+') as fo:

    #     fo.write(json.dumps(lnt))
    # logger.error(lnt)
    # logger.success(dict(lnt['definitions']['DataTypes']['enum'][0]))