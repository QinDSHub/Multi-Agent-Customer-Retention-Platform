"""
Challenge 1: Build Agents — Insurance Claims Processing
Claims Triage Agent and Claims Decision Agent for ClaimSight Insurance.

Usage:
    python agents.py

Builds both agents with system prompts, tools, and conversation handling.
"""

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import FunctionTool, PromptAgentDefinition
from azure.identity import DefaultAzureCredential
from openai.types.responses.response_input_param import FunctionCallOutput


# Resolve repo root by finding .env in parent directories.
def _find_repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / ".env").exists():
            return parent
    return Path(__file__).resolve().parents[2]


REPO_ROOT = _find_repo_root()

# Load environment
env_path = REPO_ROOT / ".env"
load_dotenv(env_path)

PROJECT_CONNECTION_STRING = os.getenv("PROJECT_CONNECTION_STRING")
# MODEL_DEPLOYMENT_NAME = os.getenv("MODEL_DEPLOYMENT_NAME", "gpt-5.4")
MODEL_DEPLOYMENT_NAME = os.getenv("MODEL_DEPLOYMENT_NAME", "gpt-chat-latest")
# CLAIMS_DATA_PATH = Path(__file__).resolve().parent / "claims_data.json"
CLAIMS_DATA_PATH = Path(__file__).resolve().parent / "churn_data.json"


def _load_claim_batch() -> list[dict]:
    """Load claim records to send as a batch payload in demo requests."""
    with open(CLAIMS_DATA_PATH, "r") as f:
        data = json.load(f)
    return data.get("claims", [])


# =============================================================================
# Tool Function: assess_claim
# This is already implemented — agents can call this to get claim risk analysis
# =============================================================================

def assess_claim(claim_id: str) -> str:
    """
    Reads claims_data.json and checks if a claim's metrics are within acceptable thresholds.
    Returns a JSON string with the analysis.
    """
    with open(CLAIMS_DATA_PATH, "r") as f:
        data = json.load(f)

    claim = None
    for c in data["claims"]:
        if c["claim_id"] == claim_id:
            claim = c
            break

    if not claim:
        return json.dumps({"error": f"Claim '{claim_id}' not found"})

    results = {
        "claim_id": claim["claim_id"],
        # "type": claim["type"],
        # "claimant": claim["claimant"],
        # "date_filed": claim["date_filed"],
        # "status": claim["status"],
        # "documents_submitted": claim["documents_submitted"],
        "mile": claim["vehicle_info"]["mile_today"],
        "vehicle_age_years": claim["vehicle_info"]["vehicle_age_years"],
        "warranty_expired": claim["vehicle_info"]["warrenty_expired_this_year"],
        # "recommended_action": claim["recommended_promotion"],
        "flags": [],
        "all_metrics": {},
    }

    for metric, reading in claim["metrics"].items():
        value = reading["value"]
        threshold = claim["thresholds"][metric]
        in_spec = threshold["min"] <= value <= threshold["max"]

        results["all_metrics"][metric] = {
            "value": value,
            "unit": reading["unit"],
            "min": threshold["min"],
            "max": threshold["max"],
            "in_spec": in_spec,
        }

        if not in_spec:
            deviation = ""
            if value > threshold["max"]:
                pct = ((value - threshold["max"]) / threshold["max"]) * 100
                deviation = f"{pct:.1f}% above max"
            elif value < threshold["min"]:
                pct = ((threshold["min"] - value) / threshold["min"]) * 100
                deviation = f"{pct:.1f}% below min"

            results["flags"].append({
                "metric": metric,
                "value": value,
                "unit": reading["unit"],
                "threshold_min": threshold["min"],
                "threshold_max": threshold["max"],
                "deviation": deviation,
            })

    return json.dumps(results, indent=2)


# Tool definition for the agent (Foundry FunctionTool format)
ASSESS_CLAIM_TOOL = FunctionTool(
    name="assess_claim",
    # description=(
    #     "Assess a vehicle owner's churn metrics against acceptable thresholds. "
    #     "Returns flags if any metric is outside its acceptable range "
    #     "(e.g., customer value too low, churn risk too high, loyalty too low, "
    #     "vehicle lifecycle too late)."
    # ),
    description=(
    "Assess vehicle customer churn using customer value, churn risk, RFM loyalty, "
    "and vehicle lifecycle metrics. Flags out-of-range values and returns deviations."
    ),
    parameters={
        "type": "object",
        "properties": {
            "claim_id": {
                "type": "string",
                "description": "The claim ID (VIN, e.g., 'VIN001') to assess",
            }
        },
        "required": ["claim_id"],
        "additionalProperties": False,
    },
    strict=False,
)

# =============================================================================
# Claims Triage Agent
# =============================================================================


class ClaimsTriageAgent:
    def __init__(self):
        self.agent = None
        self.client = None
        self.openai = None

    def create(self):
        """Create the claims triage agent in Foundry."""
        self.client = AIProjectClient(
            endpoint=PROJECT_CONNECTION_STRING,
            credential=DefaultAzureCredential(),
        )
        self.openai = self.client.get_openai_client()

        # system_prompt = """
        # You are an insurance claims triage specialist for ClaimSight Insurance.
        # When asked to assess claims, use the assess_claim tool for each claim.
        # For each claim, report:
        # - Claim ID, type, and claimant name
        # - Risk classification (normal / warning / critical)
        # - Each metric that is flagged: current value, threshold violated, deviation
        # - Missing documents if completeness is below threshold
        # Use ⚠️ for warning and 🔴 for critical flags.
        # If all metrics are within acceptable ranges, mark the claim as normal (✅).
        # Be concise and structured.
        # """

        # system_prompt = """
        # You are a customer churn triage specialist for a 4S dealership under Vehicle Group.
        # When asked to assess claims, use the assess_claim tool for each claim.

        # Each claim contains four churn metrics, and each metric has:
        # - a current value
        # - an acceptable range [min, max]

        # A metric is "flagged" when its value falls OUTSIDE its acceptable range.
        # The deviation is how far the value is outside the range.

        # The four metrics to evaluate are:
        # - customer_value_score          (higher is better)
        # - churn_risk_score              (lower is better)
        # - rfm_loyalty_score             (higher is better)
        # - vehicle_usage_lifecycle_score (lower is better; higher = later lifecycle = higher churn risk)

        # For each claim, report:
        # - Claim ID (VIN)
        # - Vehicle info summary: current mileage, vehicle age, and whether warranty expired
        # - Evaluate churn risk (high / medium / low)
        # - Each flagged metric: current value, acceptable range, and deviation

        # Risk rules:
        # - high   (🔴): 3 or more metrics flagged
        # - medium (⚠️): 1–2 metrics flagged
        # - low    (✅): no metric flagged

        # Be concise and structured.
        # """

        system_prompt = "You are a customer churn triage specialist for a 4S dealership under Vehicle Group. Use the assess_claim tool for each customer. Each customer has four churn metrics with acceptable ranges [min,max]. A metric is flagged if its value is outside the range; deviation is the distance outside the range. Metrics: customer_value_score (higher is better), churn_risk_score (lower is better), rfm_loyalty_score (higher is better), vehicle_usage_lifecycle_score (lower is better; higher means later lifecycle and higher churn risk). Report VIN, vehicle summary (mileage, age, warranty expired), churn risk (high/medium/low), and each flagged metric with current value, range, and deviation. Risk: high 🔴=3+ flagged, medium ⚠️=1–2, low ✅=0. Be concise and structured."

        self.agent = self.client.agents.create_version(
            # agent_name="claims-triage-agent",
            agent_name="churn-triage-agent",
            definition=PromptAgentDefinition(
                model=MODEL_DEPLOYMENT_NAME,
                instructions=system_prompt,
                tools=[ASSESS_CLAIM_TOOL],
            ),
        )

        return self.agent

    def run(self, input_text: str) -> str:
        """Run the claims triage agent with the given input."""
        conversation = self.openai.conversations.create()

        response = self.openai.responses.create(
            input=input_text,
            conversation=conversation.id,
            extra_body={"agent_reference": {
                "name": self.agent.name, "type": "agent_reference"}},
        )

        # Handle function call loops
        while True:
            function_calls = [
                item for item in response.output if item.type == "function_call"]
            if not function_calls:
                break

            input_list = []
            for item in function_calls:
                if item.name == "assess_claim":
                    args = json.loads(item.arguments)
                    result = assess_claim(args["claim_id"])
                else:
                    result = json.dumps(
                        {"error": f"Unknown tool '{item.name}'"})

                input_list.append(
                    FunctionCallOutput(
                        type="function_call_output",
                        call_id=item.call_id,
                        output=result,
                    )
                )

            response = self.openai.responses.create(
                input=input_list,
                conversation=conversation.id,
                extra_body={"agent_reference": {
                    "name": self.agent.name, "type": "agent_reference"}},
            )

        self.openai.conversations.delete(conversation_id=conversation.id)
        return response.output_text

    def cleanup(self):
        """Delete the agent version and close connections."""
        if self.agent:
            self.client.agents.delete_version(
                agent_name=self.agent.name,
                agent_version=self.agent.version,
            )
        if self.client:
            self.client.close()


# =============================================================================
# Claims Decision Agent
# =============================================================================

class ClaimsDecisionAgent:
    def __init__(self):
        self.agent = None
        self.client = None
        self.openai = None

    def create(self):
        """Create the claims decision agent in Foundry."""
        self.client = AIProjectClient(
            endpoint=PROJECT_CONNECTION_STRING,
            credential=DefaultAzureCredential(),
        )
        self.openai = self.client.get_openai_client()

        # system_prompt = """
        # You are a senior claims adjuster and decision specialist for ClaimSight Insurance.
        # Given a list of flags from a claim assessment, your job is to:
        # 1. Determine the recommended action based on the pattern of flags:
        #    - High fraud risk score alone → Investigate for potential fraud
        #    - Low completeness alone → Request missing documents before proceeding
        #    - High fraud risk + low damage-estimate match → Likely inflated claim, escalate to SIU
        #    - Low policy coverage match → Partial denial, cover only matched items
        #    - Multiple critical flags → Compound risk, full investigation required
        # 2. Recommend specific, actionable next steps for the claims adjuster.
        # 3. Estimate urgency: IMMEDIATE (potential fraud), WITHIN 48H (missing docs), or STANDARD (routine).
        # Be concise. Format your response as:
        # RECOMMENDED ACTION: APPROVE / REQUEST DOCUMENTS / INVESTIGATE / DENY
        # REASONING: ...
        # NEXT STEPS: ...
        # URGENCY: ...
        # """


        # system_prompt = """
        # You are a promotion decision specialist for a 4S dealership under Vehicle Group.

        # Input: churn triage output — VIN, churn risk (high/medium/low), flagged metrics,
        # and vehicle info (mile_today, vehicle_age_years, warrenty_expired_this_year).

        # Recommend exactly ONE promotion:
        # - extended_warranty_insurance: warranty expired/expiring this year, mileage not extreme
        # - new_car_trade_in: mileage >= 80,000 km and vehicle_age_years == 4
        # - maintenance_package: otherwise (still active, or best chance to lock in future visits)

        # Priority: if multiple apply, pick the one addressing the strongest churn signal.

        # Explain in <= 2 sentences. Format:
        # RECOMMENDED ACTION: <action>
        # REASONING: <1-2 sentences>
        # """

        system_prompt = "You are a promotion decision specialist for a 4S dealership under Vehicle Group. Input: churn triage output with VIN, churn risk (high/medium/low), flagged metrics, and vehicle info (mile_today, vehicle_age_years, warrenty_expired_this_year). Recommend exactly ONE promotion: extended_warranty_insurance if warranty expired/expiring this year and mileage <80,000 km; new_car_trade_in if mileage >=80,000 km and vehicle_age_years ==4; otherwise maintenance_package. If multiple apply, choose the promotion addressing the strongest churn signal. Explain in <=2 sentences. Format: RECOMMENDED ACTION: <action> REASONING: <1-2 sentences>"

        self.agent = self.client.agents.create_version(
            # agent_name="claims-decision-agent",
            agent_name="churn-decision-agent",
            definition=PromptAgentDefinition(
                model=MODEL_DEPLOYMENT_NAME,
                instructions=system_prompt,
            ),
        )

        return self.agent

    def run(self, input_text: str) -> str:
        """Run the claims decision agent with the given input."""
        conversation = self.openai.conversations.create()

        response = self.openai.responses.create(
            input=input_text,
            conversation=conversation.id,
            extra_body={"agent_reference": {
                "name": self.agent.name, "type": "agent_reference"}},
        )

        self.openai.conversations.delete(conversation_id=conversation.id)
        return response.output_text

    def cleanup(self):
        """Delete the agent version and close connections."""
        if self.agent:
            self.client.agents.delete_version(
                agent_name=self.agent.name,
                agent_version=self.agent.version,
            )
        if self.client:
            self.client.close()


# =============================================================================
# Mail Content Generation Agent
# =============================================================================

class MailContentAgent:
    def __init__(self):
        self.agent = None
        self.client = None
        self.openai = None

    def create(self):
        """Create the mail content generation agent in Foundry."""
        self.client = AIProjectClient(
            endpoint=PROJECT_CONNECTION_STRING,
            credential=DefaultAzureCredential(),
        )
        self.openai = self.client.get_openai_client()

        system_prompt = "You are a customer promotion email specialist for a 4S dealership under Vehicle Group. Input: customer VIN, churn risk, vehicle info, recommended promotion, and reasoning. Generate a concise, friendly, personalized email based strictly on the recommended promotion within 200 words. Do not change or re-evaluate the recommendation. Tailor the message to the vehicle's age, mileage, warranty status, and churn signal when relevant. Clearly explain the promotion benefit and include a natural call to action. Do not invent prices, discounts, deadlines, or other unsupported details. Format: SUBJECT: <subject> EMAIL: <email body>"
        self.agent = self.client.agents.create_version(
            # agent_name="claims-decision-agent",
            agent_name="email-content-agent",
            definition=PromptAgentDefinition(
                model=MODEL_DEPLOYMENT_NAME,
                instructions=system_prompt,
            ),
        )

        return self.agent

    def run(self, input_text: str) -> str:
        """Run the mail content generation agent with the given input."""
        conversation = self.openai.conversations.create()

        response = self.openai.responses.create(
            input=input_text,
            conversation=conversation.id,
            extra_body={"agent_reference": {
                "name": self.agent.name, "type": "agent_reference"}},
        )

        self.openai.conversations.delete(conversation_id=conversation.id)
        return response.output_text

    def cleanup(self):
        """Delete the agent version and close connections."""
        if self.agent:
            self.client.agents.delete_version(
                agent_name=self.agent.name,
                agent_version=self.agent.version,
            )
        if self.client:
            self.client.close()


# =============================================================================
# Main — Test three agents
# =============================================================================

def main():
    if not PROJECT_CONNECTION_STRING:
        print("❌ PROJECT_CONNECTION_STRING not set. Run challenge 0 first!")
        sys.exit(1)

    print("=== Claims Triage Agent ===")
    print("Creating agent...")

    triage_agent = ClaimsTriageAgent()
    triage_agent.create()
    print(
        f"✅ Created: {triage_agent.agent.name} (version {triage_agent.agent.version})")

    print("\nAssessing all claims...")
    claim_batch = _load_claim_batch()
    claim_ids = [claim["claim_id"] for claim in claim_batch]
    # triage_result = triage_agent.run(
    #     "You are receiving a batch payload of claims that must be assessed in one run. "
    #     "Use assess_claim for each claim_id in the payload and summarize all flags.\n\n"
    #     f"BATCH_CLAIM_IDS: {json.dumps(claim_ids)}\n"
    #     "BATCH_CLAIM_DATA:\n"
    #     f"{json.dumps(claim_batch, indent=2)}"
    # )

    triage_result = triage_agent.run(
    "Assess every customer in this batch using assess_claim exactly once per VIN. "
    "Return the churn classification and all flagged metrics for each customer.\n"
    f"BATCH_VINS: {json.dumps(claim_ids)}\n"
    f"BATCH_CUSTOMER_DATA: {json.dumps(claim_batch)}")

    print(triage_result)

    print("\n=== Claims Decision Agent ===")
    print("Creating agent...")

    decision_agent = ClaimsDecisionAgent()
    decision_agent.create()
    print(
        f"✅ Created: {decision_agent.agent.name} (version {decision_agent.agent.version})")

    print("\nDeciding on high-risk claim batch...")
    high_risk_batch = [
        claim for claim in claim_batch if claim["status"] in {"high", "medium"}]
    # decision_result = decision_agent.run(
    #     "You are receiving a batch payload of high and medium risk claims. For each claim, provide: "
    #     "recommended action and reasoning.\n\n"
    #     "HIGH_RISK_CLAIM_BATCH:\n"
    #     f"{json.dumps(high_risk_batch, indent=2)}"
    # )

    decision_result = decision_agent.run(
    "For each high- or medium-risk customer in this batch, recommend exactly one promotion and provide a brief reason.\n"
    f"HIGH_MEDIUM_RISK_CUSTOMER_BATCH: {json.dumps(high_risk_batch)}")
    
    print(decision_result)


    print("\n=== Email Content Agent ===")
    print("Creating agent...")
    mail_agent = MailContentAgent()
    mail_agent.create()
    print(f"✅ Created: {mail_agent.agent.name} (version {mail_agent.agent.version})")

    generation_batch = [customer for customer in claim_batch]
    content_result = mail_agent.run(
    "For each customer in this batch, generate a concise, friendly, personalized email based strictly on the provided recommended promotion and customer information. Do not re-evaluate churn risk or change the recommended promotion. Include a clear promotion benefit and a natural call to action.\n"
    f"CHURN_CUSTOMER_BATCH: {json.dumps(generation_batch)}"
)

    print(content_result)

    # Cleanup — comment out to keep agents visible in the Foundry portal
    # print("\nCleaning up agents...")
    # triage_agent.cleanup()
    # decision_agent.cleanup()
    # print("✅ Done!")


if __name__ == "__main__":
    main()
