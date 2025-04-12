from dotenv import load_dotenv
from openai import OpenAI
import os
import requests
import json

load_dotenv()

client = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

def get_weather(city:str):
    url = f"https://wttr.in/{city}?format=%C+%t"
    response = requests.get(url)

    if response.status_code == 200:
        return f"The weather in {city} is {response.text}"
    return "Something went wrong, please try again."

available_tools = {
    "get_weather": {
        "fn" : get_weather,
        "description":"Takes a city name as an input and returns the current weather for the city."
    }
}

system_prompt = f"""
You are an helpfull AI Assistant who is specilaized in resolving user query.
For the given appropriate query you deligently try to resolve the query by working on start,plan,action and observer mode.

For the given user query and available tools, plan the step by step by execution. Based on the planning, select the relevant tool from the available tool.
And based on the tool selected you perform an action to call the tool.
Then wait for the observation and based on the observation from the tool called resolve the user query.

Rules:
1. Follow the Ouput JSON format
2. Always perform one step at a time and wait for the next input
3. Carefully analyse the user query

Output JSON format:
{{
    "step":"string",
    "content":"string",
    "function":"The name of the function if the step is action",
    "input":"The input parameter for the function"
}}

Available Tools:
- get_weather : Takes the city name as an input and returns the current weather for the city

Example:
User Query: What is the weather of delhi?
Output:{{"step":"start", "content":"The user is interested in weather information of delhi"}}
Output:{{"step":"plan","content":"From the available tools I should call get_weather"}}
Output:{{"step":"action","function":"get_weather", "input":"delhi"}}
Output:{{"step":"observe","output":"20 deg celcius"}}
Output:{{"step":"output","content":"The weather for the new york seems to be 12 degrees celcius."}}
"""

messages = [
    {"role":"system", "content":system_prompt}
]

while True:
    user_query = input(">")
    messages.append({"role":"user","content":user_query})
    
    while True:
        response = client.chat.completions.create(
            model="gemini-2.0-flash",
            response_format={"type":"json_object"},
            messages=messages
        )
    
        parsed_output = json.loads(response.choices[0].message.content)
        messages.append({"role":"assistant","content":json.dumps(parsed_output)})
    
        if parsed_output.get("step") == "start" or parsed_output.get("step") == "plan":
            print(f"🧠: {parsed_output.get("content")}")
            continue
        
        if parsed_output.get("step") == "action":
            tool_name = parsed_output.get("function")
            tool_input = parsed_output.get("input")
    
            if tool_name in available_tools:
                output = available_tools[tool_name].get("fn")(tool_input)
                messages.append({"role":"assistant","content":json.dumps({"step":"observe","output":output})})
                continue
            
        if parsed_output.get("step") == "output":
            print(f"🤖: {parsed_output.get("content")}")
            break



