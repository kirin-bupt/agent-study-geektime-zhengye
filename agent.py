from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, Tool
from langchain.tools import DuckDuckGoSearchRun

# 1. 初始化大模型
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# 2. 定义工具（搜索 + 计算器）
tools = [
    Tool(
        name="Search",
        func=DuckDuckGoSearchRun().run,
        description="用于联网搜索信息"
    ),
    Tool(
        name="Calculator",
        func=lambda x: str(eval(x)),
        description="用于数学计算，输入数学表达式"
    )
]

# 3. 初始化 Agent（ReAct 模式，最经典）
agent = initialize_agent(
    tools,
    llm,
    agent="zero-shot-react-description",
    verbose=True  # 打开日志，看Agent思考过程
)

# 4. 运行 Hello World
if __name__ == "__main__":
    res = agent.run("帮我算一下 123 * 456 等于多少？")
    print("\n最终结果：", res)