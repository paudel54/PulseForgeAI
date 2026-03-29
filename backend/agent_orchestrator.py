import json

class PulseForgeOrchestrator:
    """
    Multi-Agent LLM Orchestrator.
    Manages Role-Based AI routing, context structuring, and prompt injection
    for the Duty Doctor (MedGemma) and Nurse Copilot.
    """
    def __init__(self):
        self.patient_prompt = (
            "You are the Talk to Your Heart Wellness Companion—a warm, supportive assistant.\n"
            "CRITICAL RULES: You are a WELLNESS companion, NOT a medical provider. "
            "NEVER use diagnostic language. NEVER recommend medication changes. "
            "Frame observations as trends. Use simple, warm language."
        )
        
        self.clinician_prompt = (
            "You are the Talk to Your Heart Clinical Review Agent (MedGemma-27B).\n"
            "Generate actionable clinical evaluations based strictly on the provided physiologic "
            "telemetry, history, and retrieved literature context. Use precise medical terminology."
            "Cite the provided Literature explicitly if utilized."
        )

    def assemble_prompt(self, role: str, patient_data: dict, 
                        retrieved_context: str, cohort_context: str, 
                        safety_bounds: tuple, query: str) -> dict:
        """
        Assembles a comprehensive Prompt Dictionary based on the active requested persona.
        """
        is_safe, alert_level, safety_reason = safety_bounds
        
        # Inject deterministic safety context into the LLM's brain
        safety_status = (
            f"DETERMINISTIC GUARDRAIL RULING:\n"
            f"- Status: {'SAFE' if is_safe else 'ALERT TRIGGERED'}\n"
            f"- Alert Level: {alert_level.upper()}\n"
            f"- Reason: {safety_reason}\n"
        )

        system_role = self.patient_prompt if role == "patient" else self.clinician_prompt

        dynamic_context = f"""
{safety_status}

Patient Physiological Data & History (Polar H10):
{json.dumps(patient_data, indent=2)}

Knowledge Base Context (from uploaded medical documents):
{retrieved_context}
{cohort_context}

User Query:
{query}
"""
        return {
            "system": system_role,
            "prompt": dynamic_context,
            "alert": alert_level
        }
