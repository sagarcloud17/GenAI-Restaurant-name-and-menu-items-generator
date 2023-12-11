from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains import SequentialChain

from secret_key import openai_api_key
import os
os.environ["OPENAI_API_KEY"]=openai_api_key

llm = OpenAI(temperature=0.6)

def generate_restaurant_name_and_items(cuisine):
    prompt_template_name= PromptTemplate(
    input_variables = ['cuisine'],
    template = "I want to open a restaurant for {cuisine} food, Generate a fancy restaurant name."
    )

    name_chain = LLMChain(llm=llm, prompt=prompt_template_name,output_key = "restaurant_name")

    prompt_template_name= PromptTemplate(
    input_variables = ['restaurant_name'],
    template = "Suggest me the menu items for {restaurant_name}"
    )

    food_items_chain = LLMChain(llm=llm, prompt=prompt_template_name, output_key = "Menu_items")

    chain = SequentialChain(
    chains=[name_chain,food_items_chain],
    input_variables=['cuisine'],
    output_variables=['restaurant_name','Menu_items']
    )

    #response =chain({'cuisine' : cuisine})
    response = chain(cuisine)

    return response

if __name__ == "__main__":
    print(generate_restaurant_name_and_items("Indian"))