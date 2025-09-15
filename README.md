# Multi-Agent Project Cost Estimator

A sophisticated AI-powered project cost estimation tool that uses multiple specialized agents to gather requirements and provide accurate cost estimates.

## Features

- **Multi-Agent Architecture**: Three specialized AI agents (Clarifier, Evaluator, Estimator)
- **Intelligent Questioning**: Context-aware questions that adapt based on project complexity
- **Professional Estimates**: Detailed cost breakdowns with team composition and timeline recommendations
- **Interactive UI**: Clean, responsive interface built with Streamlit

## How It Works

1. **Clarifier Agent**: Asks targeted questions to understand project requirements
2. **Evaluator Agent**: Determines when enough information has been gathered
3. **Estimator Agent**: Provides detailed cost estimates with breakdowns and assumptions

## Deployment

This app is ready for Streamlit Cloud deployment. Simply:

1. Push this repository to GitHub
2. Connect to Streamlit Cloud
3. Add your OpenAI API key as a secret
4. Deploy!

## Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key for GPT-4o-mini access

## Tech Stack

- **Frontend**: Streamlit
- **AI Framework**: LangChain + LangGraph
- **LLM**: OpenAI GPT-4o-mini
- **Architecture**: Multi-Agent State Graph
