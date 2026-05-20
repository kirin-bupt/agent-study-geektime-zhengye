from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

load_dotenv()


@tool
def calculate(what: str) -> float:
    """Runs a calculation and returns the number - uses Python so be sure to use floating point syntax if necessary"""
    return eval(what)


@tool
def ask_fruit_unit_price(fruit: str) -> str:
    """Asks the user for the price of a fruit"""
    if fruit.casefold() == "苹果":
        return "苹果单价是 10/kg"
    elif fruit.casefold() == "香蕉":
        return "香蕉单价是 6/kg"
    else:
        return "{} 单价是 20/kg".format(fruit)


tools = [calculate, ask_fruit_unit_price]
model = ChatOpenAI(model="gpt-4o-mini")

prompt = "你是一个助手，可以用工具来帮助回答用户的问题。"

agent = create_agent(model=model, tools=tools, system_prompt=prompt)

# 维护对话历史（短期记忆）
conversation_history = []


def print_stream(user_input: str):
    global conversation_history

    # 把用户输入加入历史
    conversation_history.append(user_input)

    print(f"\n👤 用户: {user_input}\n")
    print("🤖 AI: ", end="", flush=True)

    for chunk in agent.stream({"messages": conversation_history}):
        # 处理 model 节点（AI 响应）
        if "model" in chunk:
            msg_list = chunk["model"].get("messages", [])
            for msg in msg_list:
                # 显示 AI 的思考（tool_calls）
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        print(
                            f"\n💭 思考: 调用 {tc['name']}({tc['args']})",
                            end="",
                            flush=True,
                        )
                # 显示最终回复
                if hasattr(msg, "content") and msg.content:
                    print(msg.content, end="", flush=True)
                    # 记录 AI 回复到历史
                    conversation_history.append(msg.content)

        # 处理 tool 调用
        if "tools" in chunk:
            msg_list = chunk["tools"].get("messages", [])
            for msg in msg_list:
                if hasattr(msg, "content") and msg.content:
                    print(f"\n🔧 工具返回: {msg.content}", end="", flush=True)

    print("\n")


# 主循环
if __name__ == "__main__":
    print("🤖 Agent 已启动，输入 q 退出")
    while True:
        user_input = input("👤 你: ").strip()
        if user_input.lower() == "q":
            break
        if user_input:
            print_stream(user_input)
