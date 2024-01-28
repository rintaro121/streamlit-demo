import os
import json
from langchain.prompts import PromptTemplate
from openai import OpenAI

BASE_TEMPLATE = """
You are the agent who will eventually create an ER diagram from the given json with Mermaid notation.
The generation is divided into two steps: the first step is to generate only tables from the json schema, and the second step is to create the relationship information from the json schema.
"""

FIRST_PROMPT_TEMPLATE = """
{base_template}

You are in charge of the first step.
Create the table information from the json schema given in ### schema.
Your output will be used as input for the second step, so output only the necessary information.
When creating information for a table, you MUST follow the instructions given in the ### instructions.

### instructions
- Do not generate attribute in the table. Only generate the type and name of the field.
- Fields in the table should be generated from the items listed in "fields".
- For fields that have "sourceId" as an item, do not display it as a table item, but create a relationship with another table. In such a case, the label should be written literally as "sourceId". Do not describe labels for other relationships.
- Also, the item to which "sourceId" refers should be listed as a field.
- If a field has a type of "type": "nested", display it as nested instead of expanding it to display the elements of the field.
- Comments in an ER diagram should be denoted using %%.


### schema
{schema}

### output
"""

PRE_SECONDE_PROMPT_TEMPLATE = """
You will be given a schema in json format.
In the given schema, list only the table name with "sourceId" as a field.

### schema
{schema}

### output
-
"""
# Do not create relationships with other tables for fields that do not have "sourceId" as an item.
SECOND_PROMPT_TEMPLATE = """
{base_template}

You are responsible for the second step.
Create the relation information based on the json schema given in ###schema.
When creating information for a relation, you MUST follow the instructions given in the ### instructions.

### instructions
- Each table contains a "description" as a comment, but ignore the "description" of the one-to-one and one-to-many relationships.
- Make the relation one-to-one only if the field "unique:True". Fields marked "required: true" should be mandatory.
- The label for relationship should be written as "sourceId". Do not describe labels for other relationships.
- Create relationships ONLY for tables that contain "sorceId" in the items for the tables fields.
- A list of tables with "sourceID" created for you by another agent is given in ### Table list with "sourceId". Refer to this information to create relationships.


### schema
{schema}

### Table list with "sourceId"
{table_info}
"""


def first_step_llm(schema, openai_api_key):
    """Return table info from given schema"""
    prompt = PromptTemplate(
        template=FIRST_PROMPT_TEMPLATE,
        input_variables=["base_template", "schema"],
    )
    prompt_text = prompt.format(
        base_template=BASE_TEMPLATE,
        schema=schema,
    )

    client = OpenAI(api_key=openai_api_key)
    response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[
            {
                "role": "user",
                "content": prompt_text,
            }
        ],
        temperature=0.0,
    )
    first_step_output = response.choices[0].message.content
    return first_step_output


def pre_second_step_llm(schema, openai_api_key):
    prompt = PromptTemplate(template=PRE_SECONDE_PROMPT_TEMPLATE, input_variables=["schema"])
    prompt_text = prompt.format(schema=schema)

    client = OpenAI(api_key=openai_api_key)
    response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[
            {
                "role": "user",
                "content": prompt_text,
            }
        ],
        temperature=0.0,
    )
    pre_seconde_output = response.choices[0].message.content
    return pre_seconde_output


def second_step_llm(schema, pre_seconde_output, openai_api_key):
    prompt = PromptTemplate(
        template=SECOND_PROMPT_TEMPLATE,
        input_variables=["base_template", "schema", "table_info"],
        # input_variables=["base_template", "schema", "table"],
    )
    second_prompt_text = prompt.format(
        base_template=BASE_TEMPLATE,
        schema=schema,
        table_info=pre_seconde_output
        # table=first_step_output,
    )
    client = OpenAI(api_key=openai_api_key)
    response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[
            {
                "role": "user",
                "content": second_prompt_text,
            }
        ],
        temperature=0.0,
    )
    seconde_output = response.choices[0].message.content
    return seconde_output
