from langchain.agents import Tool
import math

def simple_calculator(query: str) -> str:
    try:
        result = eval(query)
        return f"The result is: {result}"
    except Exception as e:
        return f"Error: {str(e)}"

calculator_tool = Tool(
    name="Simple Calculator",
    func=simple_calculator,
    description="Useful for simple math calculations. Input: a valid math expression."
)
