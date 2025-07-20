"""
Tiny wrapper so UI / CLI only calls agent().
"""

from graphs.mh_graph import graph_executor
import mlflow

mlflow.set_experiment("mh-agent-dev")
mlflow.langchain.autolog()

    
def agent(user_input: str) -> str:
    with mlflow.start_run(nested=True):
        state_in  = {"user_input": user_input}
        state_out = graph_executor.invoke(state_in)
    return state_out["answer"]
