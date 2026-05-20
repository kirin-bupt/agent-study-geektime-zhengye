from dotenv import load_dotenv
from openai import OpenAI
import os
import re

load_dotenv()

DEFAULT_MODEL = "gpt-4o-mini"

# 使用 OPENAI_API_BASE 环境变量
base_url = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
client = OpenAI(base_url=base_url)


class Agent:
    def __init__(self, system=""):
        self.system = system
        self.messages = []
        if self.system:
            self.messages.append({"role": "system", "content": system})

    def invoke(self, message):
        self.messages.append({"role": "user", "content": message})
        result = self.execute()
        self.messages.append({"role": "assistant", "content": result})
        return result

    def execute(self):
        completion = client.chat.completions.create(
            model=DEFAULT_MODEL, messages=self.messages, temperature=0
        )
        return completion.choices[0].message.content


prompt = """
You run in a loop of Thought, Action, PAUSE, Observation.
At the end of the loop you output an Answer
Use Thought to describe your thoughts about the question you have been asked.
Use Action to run one of the actions available to you - then return PAUSE.
Observation will be the result of running those actions.

Your available actions are:

calculate:
e.g. calculate: 4 * 7 / 3
Runs a calculation and returns the number - uses Python so be sure to use floating point syntax if necessary

ask_fruit_unit_price:
e.g. ask_fruit_unit_price: 苹果
Asks the user for the price of a fruit

Example session:

Question: What is the unit price of apple?
Thought: I need to ask the user for the price of an apple to provide the unit price. 
Action: ask_fruit_unit_price: 苹果
PAUSE

You will be called again with this:

Observation: Apple unit price is 10/kg

You then output:

Answer: The unit price of apple is 10 per kg.
""".strip()


def calculate(what):
    return eval(what)


def ask_fruit_unit_price(fruit):
    if fruit.casefold() == "苹果":
        return "苹果单价是 10/kg"
    elif fruit.casefold() == "香蕉":
        return "香蕉单价是 6/kg"
    else:
        return "{} unit price is 20/kg".format(fruit)


action_re = re.compile(r"^Action: (\w+): (.*)$")

known_actions = {"calculate": calculate, "ask_fruit_unit_price": ask_fruit_unit_price}


def query(question, max_turns=5):
    i = 0
    agent = Agent(prompt)
    next_prompt = question
    while i < max_turns:
        i += 1
        result: str = agent.invoke(next_prompt) or ""
        print(result)
        actions = [action_re.match(a) for a in result.split("\n") if action_re.match(a)]
        if actions and (first_action := actions[0]):
            # There is an action to run
            action, action_input = first_action.groups()
            if action not in known_actions:
                raise Exception("Unknown action: {}: {}".format(action, action_input))
            print(" -- running {} {}".format(action, action_input))
            observation = known_actions[action](action_input)
            print("Observation:", observation)
            next_prompt = "Observation: {}".format(observation)
        else:
            return


query("我现在有3kg的苹果和2kg的香蕉,请问一共多少钱")
