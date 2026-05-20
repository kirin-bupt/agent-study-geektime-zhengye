from dotenv import load_dotenv
from openai import OpenAI
import os

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


agent = Agent(system="You are a helpful assistant.")

while True:
    user_input = input("You:> ")
    if user_input.lower() == "exit":
        break
    result = agent.invoke(user_input)
    print(f"Assistant: {result}")
    print()
