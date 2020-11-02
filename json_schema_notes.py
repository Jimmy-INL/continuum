from typing import Dict, Any, List
import jsonschema
from jsonschema import Draft7Validator
from devtools import debug
from loguru import logger
from pydantic import BaseModel
from toolz import curried
from toolz.functoolz import curry
import addict as adt


class TestSchema(BaseModel):
    description: str = "Should ..."
    data: Dict[str, Any] = {}
    valid: bool = True


schema = {
    "type": "object",
    "properties": {
        "validations": {
            "default": [],
            "type": "array",
            "items": {
                "type": "string"
            }
        },
        "answer": {
            "type": "string"
        }
    },
    "anyOf": [{
        "not": {
            "$ref": "#/definitions/is-answer"
        }
    }],
    "required": ["validations"],
    "definitions": {
        "is-answer": {
            "properties": {
                "validations": {
                    "contains": {
                        "$ref": "#/properties/answer"
                    }
                }
            }
        },
    }
}

# http://json-schema.org/draft-07/schema#
validator = Draft7Validator(schema)


def run_test(val: Draft7Validator, test: TestSchema):
    logger.info(test.description)
    print()
    expected = test.valid

    try:
        val.validate(test.data)
        if expected is False:
            logger.error("Test passed when it shouldn't")
        else:
            logger.success("Test succeeded")
    except Exception as e:
        if expected is True:
            logger.error("Test didn't pass when it should")
            logger.exception(e)
        else:
            logger.success("Test succeeded")


run_test = curry(run_test, validator)
not_cores = adt.Dict()
not_cores.validations = ['red', 'blue', 'green']
not_cores.answer = "blue"

tests: List[TestSchema] = [
    TestSchema(
        description="Should NOT pass without cores existing.",
        data=not_cores.to_dict(),
        valid=True
    ),
]
for test in tests:
    run_test(test)