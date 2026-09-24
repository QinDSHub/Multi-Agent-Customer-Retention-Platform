"""
Challenge 4: Production Workflow — Claims Processing
Multi-agent orchestration workflow for ClaimSight Insurance.
"""

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv


def _find_repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / ".env").exists():
            return parent
    return Path(__file__).resolve().parents[2]


env_path = _find_repo_root() / ".env"
load_dotenv(env_path)

PROJECT_CONNECTION_STRING = os.getenv("PROJECT_CONNECTION_STRING")
MODEL_DEPLOYMENT_NAME = os.getenv("MODEL_DEPLOYMENT_NAME", "gpt-5.4")
CLAIMS_DATA_PATH = Path(__file__).resolve().parent.parent / "challenge-1-build" / "claims_data.json"

CLAIMS = ["CLM-001", "CLM-002", "CLM-003", "CLM-004", "CLM-005"]
TRIAGE_AGENT_NAME = "claims-triage-agent"
DECISION_AGENT_NAME = "claims-decision-agent"
# Set WORKFLOW_AGENT_NAME in .env after creating the workflow in the Foundry portal
WORKFLOW_AGENT_NAME = os.getenv("WORKFLOW_AGENT_NAME", "")


def assess_claim(claim_id: str) -> str:
    with open(CLAIMS_DATA_PATH, "r") as f:
        data = json.load(f)
    claim = next(
        (c for c in data["claims"] if c["claim_id"] == claim_id),
        None,
    )
    if not claim:
        return json.dumps({"error": f"Claim not found: {claim_id}"})
    results = {
        "claim_id": claim["claim_id"],
        "type": claim["type"],
        "claimant": claim["claimant"],
        "status": claim["status"],
        "documents_submitted": claim["documents_submitted"],
        "flags": [],
        "all_metrics": {},
    }
    for metric, reading in claim["metrics"].items():
        value = reading["value"]
        threshold = claim["thresholds"][metric]
        in_spec = threshold["min"] <= value <= threshold["max"]
        results["all_metrics"][metric] = {
            "value": value, "unit": reading["unit"],
            "min": threshold["min"], "max": threshold["max"], "in_spec": in_spec,
        }
        if not in_spec:
            if value > threshold["max"]:
                pct = ((value - threshold["max"]) / threshold["max"]) * 100
                deviation = f"{pct:.1f}% above max"
            else:
                pct = ((threshold["min"] - value) / threshold["min"]) * 100
                deviation = f"{pct:.1f}% below min"
            results["flags"].append({
                "metric": metric, "value": value,
                "unit": reading["unit"], "deviation": deviation,
            })
    return json.dumps(results, indent=2)


def ensure_agents_deployed() -> tuple:
    """Create both agents if not already deployed; reuse existing ones."""
    print("=== Step 1: Ensure Agents Are Deployed ===")

    from azure.ai.projects import AIProjectClient
    from azure.ai.projects.models import FunctionTool, PromptAgentDefinition
    from azure.identity import DefaultAzureCredential

    assess_claim_tool = FunctionTool(
        name="assess_claim",
        description="Assess an insurance claim's metrics against thresholds.",
        parameters={
            "type": "object",
            "properties": {
                "claim_id": {"type": "string", "description": "Claim ID e.g. CLM-001"},
            },
            "required": ["claim_id"],
        },
        strict=False,
    )

    client = AIProjectClient(
        endpoint=PROJECT_CONNECTION_STRING,
        credential=DefaultAzureCredential(),
    )
    existing_names = {a.name for a in client.agents.list()}

    if TRIAGE_AGENT_NAME not in existing_names:
        client.agents.create_version(
            agent_name=TRIAGE_AGENT_NAME,
            definition=PromptAgentDefinition(
                model=MODEL_DEPLOYMENT_NAME,
                instructions=(
                    "You are an insurance claims triage specialist for ClaimSight Insurance. "
                    "When asked to assess claims, use the assess_claim tool for each claim ID. "
                    "Report every metric that is flagged: claim ID, metric name, current value, "
                    "threshold violated, and deviation. "
                    "Use WARNING or CRITICAL labels. Be concise and structured."
                ),
                tools=[assess_claim_tool],
            ),
        )
        print(f"  Deployed: {TRIAGE_AGENT_NAME}")
    else:
        print(f"  Found existing: {TRIAGE_AGENT_NAME}")

    if DECISION_AGENT_NAME not in existing_names:
        client.agents.create_version(
            agent_name=DECISION_AGENT_NAME,
            definition=PromptAgentDefinition(
                model=MODEL_DEPLOYMENT_NAME,
                instructions=(
                    "You are a senior claims adjuster for ClaimSight Insurance. "
                    "Given flags from a claim assessment, recommend an action: "
                    "APPROVE, REQUEST DOCUMENTS, INVESTIGATE, or DENY. "
                    "Provide reasoning and estimate urgency: IMMEDIATE, WITHIN 48H, or STANDARD. "
                    "Format: RECOMMENDED ACTION: ... / REASONING: ... / NEXT STEPS: ... / URGENCY: ..."
                ),
            ),
        )
        print(f"  Deployed: {DECISION_AGENT_NAME}")
    else:
        print(f"  Found existing: {DECISION_AGENT_NAME}")

    client.close()
    return TRIAGE_AGENT_NAME, DECISION_AGENT_NAME


def run_claims_triage(triage_agent_name: str) -> str:
    """Call the claims triage agent for all claims; handle function call loop."""
    print("\n=== Step 2a: Claims Triage ===")

    from azure.ai.projects import AIProjectClient
    from azure.identity import DefaultAzureCredential
    from openai.types.responses.response_input_param import FunctionCallOutput

    client = AIProjectClient(
        endpoint=PROJECT_CONNECTION_STRING,
        credential=DefaultAzureCredential(),
    )
    openai_client = client.get_openai_client()
    agent_ref = {"agent_reference": {"name": triage_agent_name, "type": "agent_reference"}}

    conversation = openai_client.conversations.create()
    response = openai_client.responses.create(
        input=(
            f"Assess all claims: {', '.join(CLAIMS)}. "
            "Report every metric that is outside acceptable thresholds."
        ),
        conversation=conversation.id,
        extra_body=agent_ref,
    )

    while any(item.type == "function_call" for item in response.output):
        tool_outputs = []
        for item in response.output:
            if item.type == "function_call":
                args = json.loads(item.arguments)
                result = assess_claim(args.get("claim_id", ""))
                tool_outputs.append(
                    FunctionCallOutput(
                        type="function_call_output",
                        call_id=item.call_id,
                        output=result,
                    )
                )
        response = openai_client.responses.create(
            input=tool_outputs,
            conversation=conversation.id,
            extra_body=agent_ref,
        )

    report = response.output_text
    openai_client.conversations.delete(conversation_id=conversation.id)
    client.close()
    return report


def run_claims_decision(decision_agent_name: str, claim_id: str, flags: list) -> str:
    """Call the claims decision agent for a single flagged claim."""
    from azure.ai.projects import AIProjectClient
    from azure.identity import DefaultAzureCredential

    client = AIProjectClient(
        endpoint=PROJECT_CONNECTION_STRING,
        credential=DefaultAzureCredential(),
    )
    openai_client = client.get_openai_client()
    agent_ref = {"agent_reference": {"name": decision_agent_name, "type": "agent_reference"}}

    flag_text = "\n".join(
        f"  - {f['metric']}: {f['value']} {f['unit']} ({f['deviation']})"
        for f in flags
    )
    input_text = (
        f"Claim {claim_id} has the following flags:\n"
        f"{flag_text}\n"
        "Recommend an action and provide next steps."
    )

    conversation = openai_client.conversations.create()
    response = openai_client.responses.create(
        input=input_text,
        conversation=conversation.id,
        extra_body=agent_ref,
    )
    decision = response.output_text
    openai_client.conversations.delete(conversation_id=conversation.id)
    client.close()
    return decision


def run_claims_workflow(triage_agent: str, decision_agent: str) -> dict:
    """Orchestrate: triage all claims -> per-claim decision -> consolidated report."""
    triage_report = run_claims_triage(triage_agent)
    print(triage_report)

    print("\n=== Step 2b: Claims Decisions ===")
    decisions = {}
    flagged_claims = []

    for claim_id in CLAIMS:
        result = json.loads(assess_claim(claim_id))
        if result.get("flags"):
            flagged_claims.append(claim_id)
            print(f"  Deciding on {claim_id}...")
            decision = run_claims_decision(decision_agent, claim_id, result["flags"])
            decisions[claim_id] = decision

    return {
        "triage_report": triage_report,
        "flagged_claims": flagged_claims,
        "decisions": decisions,
        "total_claims": len(CLAIMS),
        "problematic_claims": len(flagged_claims),
    }


def print_claims_report(report: dict):
    print("\n" + "=" * 60)
    print("CLAIMSIGHT INSURANCE — CLAIMS PROCESSING REPORT")
    print("=" * 60)
    print(f"  Claims assessed    : {report['total_claims']}")
    print(f"  Claims flagged     : {report['problematic_claims']}")

    if report["flagged_claims"]:
        print(f"  Flagged claims     : {', '.join(report['flagged_claims'])}")
        print("\n--- Decisions ---")
        for claim_id, decision in report["decisions"].items():
            print(f"\n{claim_id}:")
            print(decision)
    else:
        print("\n  All claims passed triage — no flags detected.")

    print("=" * 60)


def create_workflow_agent(workflow_agent_name: str = "claims-processing-workflow") -> str:
    """
    Create a workflow agent via the SDK using WorkflowAgentDefinition.

    The workflow appears in the Foundry portal under Build → Agents (kind: workflow).
    Requires allow_preview=True on AIProjectClient.

    Note: WorkflowAgentDefinition agents are visible in the Foundry portal
    and can be invoked from the portal UI. Programmatic invocation via the
    Responses API returns a 'wfresp_' tracking object.

    Returns:
        The workflow agent name.
    """
    from azure.ai.projects import AIProjectClient
    from azure.ai.projects.models import WorkflowAgentDefinition
    from azure.identity import DefaultAzureCredential

    client = AIProjectClient(
        endpoint=PROJECT_CONNECTION_STRING,
        credential=DefaultAzureCredential(),
        allow_preview=True,
    )

    # Exact portal YAML format: flat InvokeAzureAgent actions with agent.name,
    # conversationId, input/output, and a final EndConversation action.
    workflow_yaml = (
        "kind: Workflow\n"
        f"name: {workflow_agent_name}\n"
        "description: ClaimSight insurance claims processing - triage then decide\n"
        "trigger:\n"
        "  kind: OnConversationStart\n"
        "  id: trigger_start\n"
        "  actions:\n"
        "    - kind: InvokeAzureAgent\n"
        "      id: step_triage\n"
        "      agent:\n"
        "        name: claims-triage-agent\n"
        "      conversationId: =System.ConversationId\n"
        "      input:\n"
        '        messages: ""\n'
        "      output:\n"
        "        autoSend: true\n"
        "    - kind: InvokeAzureAgent\n"
        "      id: step_decide\n"
        "      agent:\n"
        "        name: claims-decision-agent\n"
        "      conversationId: =System.ConversationId\n"
        "      input:\n"
        '        messages: ""\n'
        "      output:\n"
        "        autoSend: true\n"
        "    - kind: EndConversation\n"
        "      id: step_end\n"
    )

    existing_names = {a.name for a in client.agents.list()}
    if workflow_agent_name in existing_names:
        result = client.agents.create_version(
            agent_name=workflow_agent_name,
            definition=WorkflowAgentDefinition(workflow=workflow_yaml),
            description="ClaimSight insurance claims workflow (SDK-created)",
        )
        print(f"  Updated workflow agent: {result.name} (version {result.version})")
    else:
        result = client.agents.create_version(
            agent_name=workflow_agent_name,
            definition=WorkflowAgentDefinition(workflow=workflow_yaml),
            description="ClaimSight insurance claims workflow (SDK-created)",
        )
        print(f"  Created workflow agent: {result.name} (version {result.version})")
    print(f"  Visible in Foundry portal → Build → Agents (kind: workflow)")
    client.close()
    return result.name


def run_portal_workflow(workflow_name: str) -> str:
    """
    Invoke a WorkflowAgentDefinition agent via the Responses API.

    Embeds the claims data directly in the input so the claims-triage-agent
    can process all claims without needing to call the assess_claim tool
    (workflow steps cannot handle function-call loops). Both agents execute
    sequentially.

    Returns:
        The workflow's combined text output.
    """
    import time
    from azure.ai.projects import AIProjectClient
    from azure.identity import DefaultAzureCredential

    client = AIProjectClient(
        endpoint=PROJECT_CONNECTION_STRING,
        credential=DefaultAzureCredential(),
        allow_preview=True,
    )
    openai_client = client.get_openai_client()

    print(f"\n=== Portal Workflow: {workflow_name} ===")

    portal_base = PROJECT_CONNECTION_STRING.split("/api/projects/")[0] if "/api/projects/" in PROJECT_CONNECTION_STRING else ""
    if portal_base:
        print(f"\n  View in Foundry portal:")
        print(f"  {portal_base.replace('services.ai.azure.com', 'ai.azure.com')}/build/agents")

    print(f"\n  Workflow steps:")
    print(f"    1. claims-triage-agent    — triage claims for completeness and fraud indicators")
    print(f"    2. claims-decision-agent  — make approval/denial decisions for triaged claims")

    # Embed claims data in the input so agents don't need tool calls.
    # The claims-triage-agent is instructed to call assess_claim per claim,
    # but workflow steps cannot handle function-call loops. We provide all claim
    # details upfront and explicitly instruct the agent to work from the provided data.
    with open(CLAIMS_DATA_PATH, "r") as f:
        claims_data = json.load(f)
    claims_text = json.dumps(claims_data["claims"], indent=2)
    query = (
        "All claims data for today is provided below — do NOT call assess_claim. "
        "Analyse the data directly from this message.\n\n"
        + claims_text
        + "\n\nFor each claim, assess completeness, check for fraud indicators, and "
        "classify risk (normal/warning/critical). Then make a clear approval or denial "
        "decision for each claim with justification."
    )

    conversation = openai_client.conversations.create()
    print(f"\n  Submitting workflow run (background)...")

    resp = openai_client.responses.create(
        conversation=conversation.id,
        extra_body={"agent_reference": {"name": workflow_name, "type": "agent_reference"}},
        input=query,
        background=True,
    )
    print(f"  Response ID : {resp.id}")
    print(f"  Initial status: {resp.status}")

    output_text = ""
    for attempt in range(12):
        time.sleep(8)
        r = openai_client.responses.retrieve(resp.id)
        tokens = getattr(r.usage, "total_tokens", 0)
        print(f"  [{attempt + 1}] status={r.status}  tokens={tokens}")
        if r.status in ("completed", "failed", "cancelled"):
            output_text = r.output_text
            break

    if output_text:
        print("\nWorkflow output:")
        print(output_text)
    else:
        print(
            "\n  Note: Workflow invocation returned no text output via the API.\n"
            "  The agent is deployed and visible in Foundry portal → Build → Agents."
        )

    openai_client.conversations.delete(conversation_id=conversation.id)
    client.close()
    return output_text


def main():
    if not PROJECT_CONNECTION_STRING:
        print("PROJECT_CONNECTION_STRING not set. Run challenge 0 first!")
        sys.exit(1)

    # --- Part A: Python orchestration (agents called step-by-step from code) ---
    triage_agent, decision_agent = ensure_agents_deployed()
    report = run_claims_workflow(triage_agent, decision_agent)
    print_claims_report(report)

    print("\nWorkflow complete! Agents remain deployed for future runs.")

    # --- Part B: SDK workflow creation + portal invocation ---
    print("\n" + "=" * 60)
    print("CREATING WORKFLOW AGENT VIA SDK")
    print("=" * 60)
    workflow_name = WORKFLOW_AGENT_NAME if WORKFLOW_AGENT_NAME and not WORKFLOW_AGENT_NAME.startswith("<") else "claims-processing-workflow"
    workflow_name = create_workflow_agent(workflow_agent_name=workflow_name)

    print("\n" + "=" * 60)
    print("INVOKING WORKFLOW (BACKGROUND POLL)")
    print("=" * 60)
    run_portal_workflow(workflow_name)

    print("\n" + "=" * 60)
    print("CHALLENGE 4 COMPLETE")
    print("=" * 60)
    print("  Part A: Multi-agent SDK orchestration  ✓")
    print(f"  Part B: Workflow agent deployed        ✓  ({workflow_name})")
    print("          → View in Foundry portal → Build → Agents")


if __name__ == "__main__":
    main()
