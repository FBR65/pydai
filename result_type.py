from pydantic import BaseModel, Field

class ResultType(BaseModel):
    type: str = Field(
        ...,
        description="The type of the result. Possible values are 'problem' (for problem definitions) and 'requery' (for queries)."
    )
    value: str = Field(
        ...,
        description="The content of the result, which is a string provided by the corresponding agent."
    )