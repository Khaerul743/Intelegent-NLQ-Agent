from src.agent import AgentNLQ
from src.base import BaseAgentStateModel
from src.schema import DatasetDetailInformation

dataset = DatasetDetailInformation(
    available_datasets=["customers"],
    dataset_descriptions={"customers": "Data customer"},
)


agent = AgentNLQ(dataset, "dataset", "openai", "gpt-4o-mini")

response_times = []
while True:
    user_message = input("👤 User: ")
    if user_message.lower() in ["exit", "e", "ex"]:
        try:
            print("=========Conversation Detail==========")
            print(f"Response time: {sum(response_times) / len(response_times)}")
            print(f"Total token: {agent.get_token_usage()}")
            print("======================================")
            print("\n========HISTORY CONVERSATION=========")
            print(agent.show_execute_detail())
        except ZeroDivisionError:
            print("NO CONVERSATION")
        break

    agent.execute(BaseAgentStateModel(user_message=user_message), "default")
    print(f"🤖 AI: {agent.get_response()}")
    response_times.append(agent.get_response_time())
