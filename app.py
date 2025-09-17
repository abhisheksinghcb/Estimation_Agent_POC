import os
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from typing import TypedDict, List, Dict, Optional
from langgraph.graph import StateGraph, END
import json
from datetime import datetime

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")


if not OPENAI_API_KEY:
    st.error(
        "⚠️ OpenAI API key not found. Please set OPENAI_API_KEY in your environment variables or Streamlit secrets."
    )
    st.stop()


class ConversationState(TypedDict):
    """Type definition for conversation state."""

    project_context: Dict[str, str]
    chat_history: List[Dict[str, str]]
    current_user_message: str
    active_agents: List[str]
    estimation_ready: bool
    final_estimate: str
    agent_contributions: List[str]
    unified_response: str


def create_conversation_state() -> ConversationState:
    """Create initial conversation state."""
    return {
        "project_context": {},
        "chat_history": [],
        "current_user_message": "",
        "active_agents": [],
        "estimation_ready": False,
        "final_estimate": "",
        "agent_contributions": [],
        "unified_response": "",
    }


CONFIG = {
    "model": "gpt-4o-mini",
    "temperature": 0.4,
    "context_window": 5,
    "agent_context_window": 3,
    "detailed_info_threshold": 5,
}

try:
    llm = ChatOpenAI(
        model=CONFIG["model"], temperature=CONFIG["temperature"], api_key=OPENAI_API_KEY
    )
except Exception as e:
    st.error(f"Failed to initialize AI model: {str(e)}")
    st.stop()

coordinator_prompt = ChatPromptTemplate.from_template(
    """
    You are a friendly software development consultant having a natural conversation with a client.
    
    CONTEXT:
    - Project Context: {project_context}
    - Chat History: {chat_history}  
    - User Message: {user_message}
    
    APPROACH:
    - Have a natural, consultative conversation like a real developer would
    - Ask questions that naturally arise from what the client is telling you
    - Don't follow a rigid script or checklist
    - Be curious about their business and technical needs
    - Adapt your questions based on the conversation flow
    
    CONVERSATION STYLE:
    - Be conversational and professional, like consulting with a real client
    - Ask thoughtful questions that show you understand their business
    - Focus on what you need to know to help them succeed
    - Let the conversation evolve naturally based on their responses
    - Be flexible and responsive to their needs and pace
    
    WHEN TO ACTIVATE SPECIALISTS (can activate multiple):
    - business_analyst: To understand business requirements, user stories, or feature details.
    - technical_expert: To discuss technology stack, architecture, or implementation specifics.
    
    WHEN TO ESTIMATE:
    - DO NOT set "is_ready_for_estimation" to true until the user has provided a clear picture of the project.
    - You must have a general understanding of: 1. Core Features, 2. Target Platform (Web, Mobile, etc.), 3. Data Sources (How will data be populated?), 4. User Roles (e.g., admin, user).
    - If the user asks for an estimate prematurely, explain that you need more details to provide an accurate one and ask a clarifying question.
    
    KEY RESPONSIBILITIES:
    1.  **Synthesize Full Context**: Your most critical task is to read the entire `chat_history` and the latest `user_message` to create a complete, up-to-date summary of the project. Put this summary in the `updated_project_context` field. Do not just update the last value; create a full picture.
    2.  **Assess User Intent**: Analyze the user's latest message. If they seem to be finished providing details (e.g., "that's my whole idea," "no more features") or are directly asking for an estimate, it's time to move on.
    3.  **Activate Agents**: Based on the context, decide if any specialist agents are needed to identify any *critical* remaining gaps. Avoid activating them for minor details if the user seems ready to move on.
    4.  **Gatekeep Estimation**: Set `is_ready_for_estimation` to `true` if the core requirements are met AND the user seems finished providing information.
    5.  **Handle User Override**: If the user insists on an estimate (e.g., "give me the estimate now"), you MUST set `is_ready_for_estimation` to `true`.
    6.  **Converse**: Formulate a natural, conversational message. If you are ready to estimate, proactively suggest it (e.g., "It sounds like I have enough information to create an estimate for you. Shall I proceed?").
    
    RESPONSE FORMAT (JSON only):
    {{
        "reasoning": "Briefly explain your thought process.",
        "message": "Your natural, consultative response to the user.",
        "agents_to_activate": ["list_of_agents_or_empty"],
        "is_ready_for_estimation": boolean,
        "updated_project_context": {{ "key": "value", "another_key": "value" }}
    }}
    
    Act like an experienced consultant. Your primary goal is to determine if you have enough information to create a reliable estimate, but also to be a helpful and responsive partner to the client.
    """
)

technical_expert_prompt = ChatPromptTemplate.from_template(
    """
    You are a senior technical architect. Your role is to analyze the project's technical requirements and identify crucial missing information.

    CONTEXT:
    - Project Context: {project_context}
    - Chat History: {chat_history}
    - User Message: {user_message}

    TASK:
    - Based on the context, identify the single MOST CRITICAL technical question that remains unanswered.
    - Focus on topics like: Target Platform (Web, iOS, Android, etc.), Technology Stack preferences, Data Sources & APIs, Scalability Needs, and third-party integrations.
    - Phrase this as a single, clear question.

    OUTPUT:
    - A single question. Do not output a list.
    - Example: What platform are you targeting for the MVP: Web, iOS, or Android?
    """
)

business_analyst_prompt = ChatPromptTemplate.from_template(
    """
    You are a business analyst. Your role is to analyze the project's business requirements and identify crucial missing information.

    CONTEXT:
    - Project Context: {project_context}
    - Chat History: {chat_history}
    - User Message: {user_message}

    TASK:
    - Based on the context, identify the single MOST CRITICAL business-related question that remains unanswered.
    - Focus on topics like: Target Audience, Monetization Strategy, User Roles (e.g., is an admin panel needed?), Core Feature Gaps, and Success Metrics.
    - Phrase this as a single, clear question.

    OUTPUT:
    - A single question. Do not output a list.
    - Example: Is an admin panel required to manage users and content?
    """
)

cost_estimator_prompt = ChatPromptTemplate.from_template(
    """
    You are a project estimation expert. Your task is to provide a comprehensive and realistic estimate based on all available project information.

    CONTEXT:
    - Project Context: {project_context}
    - Chat History: {chat_history}
    
    IMPORTANT:
    - Use all the information gathered throughout the conversation.
    - Make reasonable assumptions for any missing details and explicitly state them in the "Assumptions Made" section. Acknowledge that the estimate is preliminary and will be refined.

    PROVIDE A COMPLETE ESTIMATE IN MARKDOWN FORMAT, INCLUDING:

    **1. Project Overview:**
    - Brief summary of what is being built.

    **2. Features Breakdown:**
    - List all identified core features.
    - Mention any features you've assumed are necessary for a Minimum Viable Product (MVP).

    **3. Estimated Timeline:**
    - Provide a phase-based timeline (e.g., Discovery, Design, Development, Testing, Deployment).
    - Give a realistic total project duration range (e.g., 3-5 months).

    **4. Estimated Costs (in USD):**
    - Provide a cost range for a Basic MVP and a Full-featured version.
    - Briefly explain what drives the cost difference.
    - Example: "Basic MVP: $15,000 - $25,000", "Full-featured: $40,000 - $70,000".

    **5. Team Recommendations:**
    - Suggest a suitable team composition (e.g., 1 Project Manager, 2 Developers, 1 QA).

    **6. Assumptions Made:**
    - Clearly list every assumption made to create this estimate (e.g., "Assumes standard UI/UX complexity," "Assumes no third-party API integration costs"). This is the most important section.

    **7. Next Steps:**
    - Suggest immediate next steps for the client (e.g., "Refine feature list," "Schedule a detailed technical call").

    STYLE:
    - Be professional, clear, and structured. Use Markdown for formatting.
    """
)

coordinator_agent = coordinator_prompt | llm
technical_expert_agent = technical_expert_prompt | llm
business_analyst_agent = business_analyst_prompt | llm
cost_estimator_agent = cost_estimator_prompt | llm


def coordinator_node(state: ConversationState) -> ConversationState:
    """Coordinator determines conversation flow and activates relevant agents."""
    try:
        response_str = coordinator_agent.invoke(
            {
                "project_context": json.dumps(state["project_context"]),
                "chat_history": json.dumps(
                    state["chat_history"][-CONFIG["context_window"] :]
                ),
                "user_message": state["current_user_message"],
            }
        ).content

        try:
            clean_response = response_str[
                response_str.find("{") : response_str.rfind("}") + 1
            ]
            parsed_response = json.loads(clean_response)

            state["active_agents"] = parsed_response.get("agents_to_activate", [])
            if "updated_project_context" in parsed_response:
                state["project_context"] = parsed_response["updated_project_context"]
            state["unified_response"] = parsed_response.get("message", "")
            state["estimation_ready"] = parsed_response.get(
                "is_ready_for_estimation", False
            )

        except (json.JSONDecodeError, KeyError) as e:
            st.warning(
                f"Could not parse coordinator's response. Raw response: {response_str}. Error: {e}"
            )
            state["unified_response"] = (
                "I'm having a little trouble organizing my thoughts. Could you please rephrase that?"
            )
            state["active_agents"] = []
            state["estimation_ready"] = False

    except Exception as e:
        st.error(f"An error occurred in the coordinator: {str(e)}")
        state["unified_response"] = (
            "I seem to have encountered a technical issue. Let's try that again."
        )
        state["active_agents"] = []
        state["estimation_ready"] = False

    return state


def _invoke_agent_with_context(agent, state: ConversationState) -> str:
    """Common helper to invoke agents with context."""
    return agent.invoke(
        {
            "project_context": json.dumps(state["project_context"]),
            "chat_history": json.dumps(
                state["chat_history"][-CONFIG["agent_context_window"] :]
            ),
            "user_message": state["current_user_message"],
        }
    ).content


def _add_agent_contribution(
    state: ConversationState, prefix: str, response: str
) -> None:
    """Helper to add agent contribution to state."""
    if "agent_contributions" not in state:
        state["agent_contributions"] = []
    state["agent_contributions"].append(f"{prefix}: {response}")


def run_agent(agent, state: ConversationState) -> ConversationState:
    """Dynamically runs an agent and adds its contribution to the state."""
    agent_name_map = {
        "technical_expert": (technical_expert_agent, "Technical"),
        "business_analyst": (business_analyst_agent, "Business"),
        "cost_estimator": (cost_estimator_agent, "Cost"),
    }

    agent_to_run, prefix = agent_name_map[agent.__name__]

    response = _invoke_agent_with_context(agent_to_run, state)
    _add_agent_contribution(state, prefix, response)

    if agent.__name__ == "cost_estimator":
        state["final_estimate"] = response

    return state


def technical_expert(state: ConversationState):
    return run_agent(technical_expert, state)


def business_analyst(state: ConversationState):
    return run_agent(business_analyst, state)


def cost_estimator(state: ConversationState):
    return run_agent(cost_estimator, state)


def joiner(state: ConversationState) -> ConversationState:
    """A simple node that just passes the state through.
    Used to join parallel branches."""
    return state


def synthesizer_node(state: ConversationState) -> ConversationState:
    """Synthesize all agent contributions into a single unified response."""
    contributions = state.get("agent_contributions", [])

    if contributions or state.get("final_estimate"):
        synthesis_prompt = ChatPromptTemplate.from_template(
            """
            You are a friendly and professional AI project consultant. Your task is to craft a single, natural, and conversational response to the client.

            CONTEXT:
            - Full Project Summary: {project_context}
            - User's Latest Message: {user_message}
            - Internal Team Questions & Analysis: {contributions}
            - Final Estimate (if available): {final_estimate}

            INSTRUCTIONS:

            **IF A FINAL ESTIMATE IS PROVIDED:**
            1.  Present the key takeaways from the estimate in a friendly, conversational way (e.g., the cost range and timeline).
            2.  Mention that the full, detailed estimate is also available for review.
            3.  Your goal is to transition to the next steps. Ask a question that helps refine the estimate or plan the project, such as "Does this initial estimate align with your budget?" or "Are you ready to move forward with the discovery phase?".
            4.  **Crucially, DO NOT ask for information that is already present in the 'Full Project Summary'.**

            **IF NO FINAL ESTIMATE IS PROVIDED:**
            1.  Your primary goal is to gather the information your team needs.
            2.  **Before asking anything, review the 'Full Project Summary' to see what you already know.** Do not ask redundant questions.
            3.  Synthesize the internal team's questions into a natural, conversational message. Ask the most logical next question(s). You can ask one or two closely related questions if it's more efficient.
            4.  Do NOT sound like you are reporting from different sources. Speak as a single, unified voice.

            EXAMPLE (Presenting an Estimate):
            - Project Summary: {{ "platform": "iOS and Android", ... }}
            - Final Estimate: "Basic MVP: $15k-$25k..."
            - Your Response: "Great news! We've put together an initial estimate for your project. For an MVP on both iOS and Android, you're looking at a range of $15,000 - $25,000 over 3-5 months. The full details are ready for you to review. How does this initial estimate align with the budget you had in mind?"

            Now, provide a response based on the context above.
            """
        )

        synthesizer_agent = synthesis_prompt | llm

        unified_response = synthesizer_agent.invoke(
            {
                "user_message": state["current_user_message"],
                "contributions": "\n\n".join(contributions),
                "project_context": json.dumps(state["project_context"]),
                "final_estimate": state.get("final_estimate", ""),
            }
        ).content

        state["unified_response"] = unified_response
    else:
        state["unified_response"] = state.get(
            "unified_response", "I understand your question. Let me help you with that."
        )

    state["chat_history"].append(
        {
            "agent": "assistant",
            "message": state["unified_response"],
            "timestamp": datetime.now().isoformat(),
        }
    )

    state["agent_contributions"] = []

    return state


def merge_state(old_state: ConversationState, update: dict) -> ConversationState:
    """Merge state updates while preserving existing data."""
    if isinstance(update, dict) and len(update) == 1:
        node_update = next(iter(update.values()))
        if isinstance(node_update, dict):
            return {**old_state, **node_update}
    return {**old_state, **update}


def route_to_agents_or_estimator(state: ConversationState) -> str:
    """Routes to specialist agents or the cost estimator based on the coordinator's decision."""
    if state.get("estimation_ready"):
        return ["cost_estimator"]

    active_agents = state.get("active_agents", [])
    if not active_agents:
        return ["synthesizer"]

    return active_agents


graph = StateGraph(ConversationState)

graph.add_node("coordinator", coordinator_node)
graph.add_node("technical_expert", technical_expert)
graph.add_node("business_analyst", business_analyst)
graph.add_node("cost_estimator", cost_estimator)
graph.add_node("joiner", joiner)
graph.add_node("synthesizer", synthesizer_node)

graph.set_entry_point("coordinator")

graph.add_conditional_edges(
    "coordinator",
    route_to_agents_or_estimator,
)

graph.add_edge("technical_expert", "joiner")
graph.add_edge("business_analyst", "joiner")
graph.add_edge("cost_estimator", "joiner")
graph.add_edge("joiner", "synthesizer")
graph.add_edge("synthesizer", END)

app_graph = graph.compile()


st.set_page_config(
    page_title="AI Project Estimation Team", page_icon="🤖", layout="wide"
)

st.title("Project Estimation Assistant")

if "conversation_state" not in st.session_state:
    st.session_state.conversation_state = create_conversation_state()

if "is_processing" not in st.session_state:
    st.session_state.is_processing = False

if st.session_state.conversation_state["chat_history"]:
    st.subheader("💬 Conversation")

    for message in st.session_state.conversation_state["chat_history"]:
        agent = message["agent"]
        content = message["message"]

        if agent == "assistant":
            with st.chat_message("assistant", avatar="🤖"):
                st.write(content)
        elif agent == "user":
            with st.chat_message("user"):
                st.write(content)

user_input = st.chat_input(
    "Tell me about your project or ask any questions...",
    disabled=st.session_state.is_processing,
)

if user_input and not st.session_state.is_processing:
    st.session_state.conversation_state["chat_history"].append(
        {
            "agent": "user",
            "message": user_input,
            "timestamp": datetime.now().isoformat(),
        }
    )

    st.session_state.is_processing = True
    st.session_state.conversation_state["current_user_message"] = user_input
    st.rerun()

if (
    st.session_state.is_processing
    and st.session_state.conversation_state["current_user_message"]
):

    with st.spinner("Thinking..."):
        try:
            for update in app_graph.stream(st.session_state.conversation_state):
                st.session_state.conversation_state = merge_state(
                    st.session_state.conversation_state, update
                )
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.session_state.conversation_state["chat_history"].append(
                {
                    "agent": "coordinator",
                    "message": "I apologize, but we encountered an issue. Please try rephrasing your question.",
                    "timestamp": datetime.now().isoformat(),
                }
            )

    st.session_state.conversation_state["current_user_message"] = ""
    st.session_state.conversation_state["active_agents"] = []
    st.session_state.is_processing = False
    st.rerun()

if st.session_state.conversation_state.get("final_estimate"):
    st.success("🎉 **Project Estimate Complete!**")
    with st.expander("📊 **View Complete Project Estimate**", expanded=True):
        st.markdown(st.session_state.conversation_state["final_estimate"])

print(st.session_state.conversation_state)
if st.button("🗑️ Reset Conversation", disabled=st.session_state.is_processing):
    st.session_state.conversation_state = create_conversation_state()
    st.session_state.is_processing = False
    st.rerun()
