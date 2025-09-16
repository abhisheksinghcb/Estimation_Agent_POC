import os
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from typing import TypedDict, List
from langgraph.graph import StateGraph, END

load_dotenv()

# Handle both .env file and Streamlit secrets
try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
except:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    st.error(
        "⚠️ OpenAI API key not found. Please set OPENAI_API_KEY in your environment variables or Streamlit secrets."
    )
    st.stop()


class AgentState(TypedDict):
    idea: str
    questions: List[str]
    answers: List[str]
    finalized: bool
    estimate: str


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.4, api_key=OPENAI_API_KEY)

clarifier_prompt = ChatPromptTemplate.from_template(
    """
    You are an expert business analyst gathering requirements for cost estimation.
    
    Project idea: {idea}
    Q&A History: {context}

    CRITICAL: If you have enough information to provide a meaningful cost estimate, respond ONLY with "ENOUGH_INFO".
    
    You have enough info if you know:
    - What type of application/system is being built
    - The target platform(s) 
    - The core functionality/features
    - Approximate scale or user base
    - The preferred tech stack or frameworks to use
    - If the project is a new project or a maintenance project
    - If the project already has a design
    - If the project already has a development team
    
    If ANY of these key areas are unclear, ask ONE specific question to clarify the most important missing piece.
    
    DO NOT ask about minor details, edge cases, or nice-to-have features.
    Remember: Cost estimates can be ranges - you don't need perfect precision.
    """
)

clarifier_agent = clarifier_prompt | llm


def clarifier_node(state: AgentState):
    context = "\n".join(
        [f"Q: {q}\nA: {a}" for q, a in zip(state["questions"], state["answers"])]
    )
    resp = clarifier_agent.invoke(
        {"idea": state["idea"], "context": context}
    ).content.strip()
    if resp != "ENOUGH_INFO":
        state["questions"].append(resp)
    return state


evaluator_prompt = ChatPromptTemplate.from_template(
    """
    You are an evaluator deciding if we can provide a cost estimate.

    REQUIREMENTS FOR YES:
    - We know what type of system/app is being built
    - We know the platform (web, mobile, etc.)
    - We know the preferred tech stack
    - We know if the project is a new project or a maintenance project
    - We know if the project already has a design
    - We know if the project already has a development team
    - We understand the main features/functionality  
    - We have some sense of scale/complexity

    IMPORTANT:
    - Cost estimates can be ranges (e.g., $10k-50k USD)
    - Minor details don't significantly affect estimates
    - If we can ballpark the work involved, that's enough
    - Don't demand perfect specification
    
    History: {context}

    Return YES if we can provide a meaningful cost estimate range.
    Return NO only if critical information is missing.
    
    Answer: YES or NO
    """
)

evaluator_agent = evaluator_prompt | llm


def evaluator_node(state: AgentState):
    context = "\n".join(
        [f"Q: {q}\nA: {a}" for q, a in zip(state["questions"], state["answers"])]
    )

    resp = evaluator_agent.invoke({"context": context}).content.strip().upper()
    state["finalized"] = resp == "YES"
    return state


estimator_prompt = ChatPromptTemplate.from_template(
    """
    You are a senior technical consultant providing project cost estimation.
    
    PROJECT: {idea}
    REQUIREMENTS ANALYSIS: {context}
    
    **ESTIMATION METHODOLOGY:**
    
    1. **EFFORT BREAKDOWN:**
       - Backend development (APIs, database, business logic)
       - Frontend development (UI/UX, user interfaces)
       - Integration work (third-party services, APIs)
       - DevOps & infrastructure (deployment, monitoring, security)
       - Quality assurance (testing, code review)
       - Project management (planning, coordination, communication)
    
    2. **COMPLEXITY MULTIPLIERS:**
       - Simple: Basic CRUD, minimal integrations (1.0x)
       - Medium: User auth, some integrations, moderate scale (1.5x)
       - Complex: Real-time features, heavy integrations, high scale (2.5x+)
    
    3. **TEAM COMPOSITION:**
       - Junior developers: $50-80/hour
       - Senior developers: $80-150/hour
       - Specialists (DevOps, Security, etc.): $100-200/hour
       - Project manager: $75-120/hour
       - Designer: $60-120/hour
    
    4. **RISK FACTORS:**
       - Technology unknowns (+20-40%)
       - Tight timelines (+15-30%)
       - Complex integrations (+25-50%)
       - Regulatory compliance (+20-40%)
    
    **OUTPUT FORMAT:**
    Provide a structured estimate including:
    - **Project Summary:** Key features and complexity assessment
    - **Effort Breakdown:** Hours per major component
    - **Team Recommendation:** Roles and experience levels needed
    - **Timeline:** Development phases and duration
    - **Cost Range:** Low/medium/high scenarios in USD
    - **Key Assumptions:** What could change the estimate
    - **Risk Factors:** Potential cost drivers and mitigation strategies
    
    Be specific about assumptions and provide both optimistic and conservative scenarios.
    """
)
estimator_agent = estimator_prompt | llm


def estimator_node(state: AgentState):
    context = "\n".join(
        [f"Q: {q}\nA: {a}" for q, a in zip(state["questions"], state["answers"])]
    )
    resp = estimator_agent.invoke({"idea": state["idea"], "context": context}).content
    state["estimate"] = resp
    return state


def merge_state(old_state: AgentState, update: dict) -> AgentState:
    if isinstance(update, dict) and len(update) == 1:
        node_update = next(iter(update.values()))
        if isinstance(node_update, dict):
            return {**old_state, **node_update}
    return {**old_state, **update}


graph = StateGraph(AgentState)
graph.add_node("clarifier_agent", clarifier_node)
graph.add_node("evaluator_agent", evaluator_node)
graph.add_node("estimator_agent", estimator_node)

graph.set_entry_point("clarifier_agent")
graph.add_edge("clarifier_agent", "evaluator_agent")
graph.add_conditional_edges(
    "evaluator_agent",
    lambda s: "estimator_agent" if s["finalized"] else "clarifier_agent",
    {"clarifier_agent": "clarifier_agent", "estimator_agent": "estimator_agent"},
)
graph.add_edge("estimator_agent", END)
app_graph = graph.compile()


st.title("Project Estimator Agent")

if "state" not in st.session_state:
    st.session_state.state = {
        "idea": "",
        "questions": [],
        "answers": [],
        "finalized": False,
        "estimate": "",
    }

# Initialize loading state
if "is_processing" not in st.session_state:
    st.session_state.is_processing = False

if st.session_state.state["idea"]:
    st.caption(f"Project Idea: {st.session_state.state['idea']}")

if not st.session_state.state["idea"]:
    idea = st.text_area(
        "Enter your project idea:",
        height="content",
        placeholder="e.g. 'I want to build a web application that allows users to manage their finances'",
        disabled=st.session_state.is_processing,
    )

    # Show processing state
    if st.session_state.is_processing:
        with st.spinner("Analyzing your project..."):
            st.empty()  # Placeholder to keep spinner visible

    button_disabled = st.session_state.is_processing
    if st.button("Get Estimation", disabled=button_disabled) and idea:
        st.session_state.is_processing = True
        st.session_state.state["idea"] = idea
        st.rerun()

# Process the initial request after the idea is set
if (
    st.session_state.is_processing
    and st.session_state.state["idea"]
    and not st.session_state.state["questions"]
    and not st.session_state.state["estimate"]
):
    with st.spinner("Analyzing your project..."):
        for update in app_graph.stream(st.session_state.state):
            st.session_state.state = merge_state(st.session_state.state, update)
            if st.session_state.state["questions"] and len(
                st.session_state.state["questions"]
            ) > len(st.session_state.state["answers"]):
                break
    st.session_state.is_processing = False
    st.rerun()

for q, a in zip(st.session_state.state["questions"], st.session_state.state["answers"]):
    st.write(f"**Q:** {q}")
    st.write(f"**A:** {a}")

if (
    st.session_state.state["questions"]
    and len(st.session_state.state["questions"])
    > len(st.session_state.state["answers"])
    and not st.session_state.state["estimate"]
):
    pending_q = st.session_state.state["questions"][-1]
    st.info(f"{pending_q}")

    # Show processing state for follow-up questions
    if st.session_state.is_processing:
        with st.spinner("Processing your response..."):
            st.empty()  # Placeholder to keep spinner visible

    user_answer = st.text_input(
        "Your answer:",
        key=f"answer_{len(st.session_state.state['answers'])}",
        disabled=st.session_state.is_processing,
    )

    answer_button_disabled = st.session_state.is_processing
    if st.button("Submit Answer", disabled=answer_button_disabled) and user_answer:
        st.session_state.is_processing = True
        st.session_state.state["answers"].append(user_answer)
        st.rerun()

# Process follow-up answer after rerun to show loading state
if (
    st.session_state.is_processing
    and st.session_state.state["questions"]
    and len(st.session_state.state["answers"])
    == len(st.session_state.state["questions"])
):
    with st.spinner("Processing your response..."):
        for update in app_graph.stream(st.session_state.state):
            st.session_state.state = merge_state(st.session_state.state, update)
            if (
                st.session_state.state["questions"]
                and len(st.session_state.state["questions"])
                > len(st.session_state.state["answers"])
            ) or st.session_state.state["estimate"]:
                break
    st.session_state.is_processing = False
    st.rerun()


if st.session_state.state["estimate"]:
    st.success(st.session_state.state["estimate"])

if st.button("Reset", disabled=st.session_state.is_processing):
    st.session_state.state = {
        "idea": "",
        "questions": [],
        "answers": [],
        "finalized": False,
        "estimate": "",
    }
    st.session_state.is_processing = False
    st.rerun()
