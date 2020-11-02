import re
from typing import List, Union
from schema import TitleRemover
from loguru import logger
from pydantic import BaseModel
from devtools import debug
import addict as adt

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


class Equal(TitleRemover):
    eq: List[Union[Ref, bool, int, float, str]]


def main():
    model1 = Ref(ref="property@poop")
    modeq = Equal(eq=[1, 1])
    # model2 = Ref(ref="properties@poop")
    # logger.success(model2)
    #> post_code=PostCode('SW8 5EL')
    logger.info(model1.ref)
    #> SW8 5EL
    debug(Ref.schema())
    debug(modeq.schema())


if __name__ == "__main__":
    main()