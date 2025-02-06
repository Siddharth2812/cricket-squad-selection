from langchain_anthropic import ChatAnthropic
model = ChatAnthropic(model="gpt-4o-mini", temperature=1, api_key="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")
print(model.invoke("Hey how are you?"))