import pandas as pd
from prompt import generate_qna_system_message, generate_demographic_prompt


class SyntheticAgent:
    def __init__(
        self,
        experiment_id: str,
        experiment_context: str,
        demographic_info: pd.Series,
        assigned_treatment: str,
        model_info: str,
    ):
        self.experiment_id = experiment_id
        self.experiment_context = experiment_context
        self.demographic_info = generate_demographic_prompt(demographic_info)
        self.assigned_treatment = assigned_treatment
        self.model_info = model_info
        self.qna_system_message = generate_qna_system_message(
            self.experiment_context, self.demographic_info, self.assigned_treatment
        )

    def get_demographic_info(self) -> str:
        """Return the demographic information of the synthetic agent."""
        return self.demographic_info

    def get_assigned_treatment(self) -> str:
        """Return the assigned treatment of the synthetic agent."""
        return self.assigned_treatment

    def get_model_info(self) -> str:
        """Return the model information of the synthetic agent."""
        return self.model_info

    def get_experiment_id(self) -> str:
        """Return the experiment ID of the synthetic agent."""
        return self.experiment_id

    def respond(self, question: str) -> str:
        """Generate a response based on the synthetic agent's model."""
        try:
            # Implement response generation based on model
            pass
        except Exception as e:
            # Log the exception
            print(f"Error during response generation: {e}")
            return None
