import streamlit as st
from datetime import datetime
from dateutil.relativedelta import relativedelta
import json
from langchain_core.messages import HumanMessage

from agents.fundamentals import fundamentals_agent
from agents.portfolio_manager import portfolio_management_agent
from agents.technicals import technical_analyst_agent
from agents.risk_manager import risk_management_agent
from agents.sentiment import sentiment_agent
from graph.state import AgentState
from agents.valuation import valuation_agent
from utils.display import print_trading_output
from langgraph.graph import END, StateGraph

st.set_page_config(page_title="AI Hedge Fund Streamlit", layout="wide")

def parse_hedge_fund_response(response):
    try:
        return json.loads(response)
    except:
        st.error(f"Error parsing response: {response}")
        return None

def create_workflow(selected_analysts=None):
    """Create the workflow with selected analysts."""
    workflow = StateGraph(AgentState)
    workflow.add_node("start_node", lambda state: state)
    
    # Default to all analysts if none selected
    if selected_analysts is None:
        selected_analysts = ["technical_analyst", "fundamentals_analyst", "sentiment_analyst", "valuation_analyst"]
    
    # Dictionary of all available analysts
    analyst_nodes = {
        "technical_analyst": ("technical_analyst_agent", technical_analyst_agent),
        "fundamentals_analyst": ("fundamentals_agent", fundamentals_agent),
        "sentiment_analyst": ("sentiment_agent", sentiment_agent),
        "valuation_analyst": ("valuation_agent", valuation_agent),
    }
    
    # Add selected analyst nodes
    for analyst_key in selected_analysts:
        node_name, node_func = analyst_nodes[analyst_key]
        workflow.add_node(node_name, node_func)
        workflow.add_edge("start_node", node_name)
    
    # Always add risk and portfolio management
    workflow.add_node("risk_management_agent", risk_management_agent)
    workflow.add_node("portfolio_management_agent", portfolio_management_agent)
    
    # Connect selected analysts to risk management
    for analyst_key in selected_analysts:
        node_name = analyst_nodes[analyst_key][0]
        workflow.add_edge(node_name, "risk_management_agent")
    
    workflow.add_edge("risk_management_agent", "portfolio_management_agent")
    workflow.add_edge("portfolio_management_agent", END)
    
    workflow.set_entry_point("start_node")
    return workflow

def run_hedge_fund(
    ticker: str,
    start_date: str,
    end_date: str,
    portfolio: dict,
    show_reasoning: bool = False,
    selected_analysts: list = None,
):
    # Create a new workflow if analysts are customized
    workflow = create_workflow(selected_analysts)
    agent = workflow.compile()

    final_state = agent.invoke(
        {
            "messages": [
                HumanMessage(
                    content="Make a trading decision based on the provided data.",
                )
            ],
            "data": {
                "ticker": ticker,
                "portfolio": portfolio,
                "start_date": start_date,
                "end_date": end_date,
                "analyst_signals": {},
            },
            "metadata": {
                "show_reasoning": show_reasoning,
            },
        },
    )
    return {
        "decision": parse_hedge_fund_response(final_state["messages"][-1].content),
        "analyst_signals": final_state["data"]["analyst_signals"],
    }

def display_signal(signal_data, level=0):
    """Recursively display nested signal data in a user-friendly format."""
    if isinstance(signal_data, dict):
        for key, value in signal_data.items():
            # Format key to be more readable
            display_key = key.replace('_', ' ').title()
            indent = "&nbsp;&nbsp;&nbsp;&nbsp;" * level
            
            # Handle signal types
            if key == 'signal':
                signal_color = {
                    "bullish": "🟢 Bullish",
                    "bearish": "🔴 Bearish",
                    "neutral": "🟡 Neutral",
                    "hold": "🟡 Hold",
                }.get(str(value).lower(), str(value))
                st.markdown(f"{indent}• **Signal:** {signal_color}", unsafe_allow_html=True)
            
            # Handle confidence values
            elif key == 'confidence':
                st.markdown(f"{indent}• **Confidence:** {value}%", unsafe_allow_html=True)
                st.progress(float(value) / 100)
            
            # Handle nested reasoning
            elif key == 'reasoning':
                st.divider()
                # st.markdown(f"{indent} *<span style='color: grey'>**Reasoning:**</span>*", unsafe_allow_html=True)
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        st.markdown(f"{indent}**{sub_key.replace('_', ' ').title()}**")
                        display_signal(sub_value, level + 1)
                else:
                    st.markdown(f"{indent}&nbsp;&nbsp;{value}", unsafe_allow_html=True)
            
            # Handle details directly
            elif key == 'details':
                metrics = value.split(', ')
                cols = st.columns(min(3, len(metrics)))
                for idx, metric in enumerate(metrics):
                    with cols[idx % 3]:
                        st.markdown(f"{indent}• {metric}", unsafe_allow_html=True)
            
            # Handle other nested dictionaries
            elif isinstance(value, dict):
                # st.markdown(f"{indent}• <span style='color: grey'>{display_key}:</span>", unsafe_allow_html=True)
                st.markdown(f"{indent} *<span style='color: grey'>**{display_key}:**</span>*", unsafe_allow_html=True)
                display_signal(value, level + 1)
            
            # Handle other simple values
            else:
                # st.markdown(f"{indent}• **{display_key}:** {value}", unsafe_allow_html=True)
                st.markdown(f"{indent} *<span style='color: grey'>**{display_key}:**</span>*", unsafe_allow_html=True)
    else:
        st.markdown(f"{indent}• {signal_data}", unsafe_allow_html=True)

def main():
    st.title("AI Hedge Fund Trading System")
    
    # Sidebar for inputs
    with st.sidebar:
        st.header("Configuration")
        
        # Ticker input
        ticker = st.text_input("Stock Ticker Symbol", "AAPL").upper()
        
        # Portfolio settings
        st.subheader("Portfolio Settings")
        initial_cash = st.number_input("Initial Cash ($)", value=100000.0, min_value=0.0)
        initial_stock = st.number_input("Initial Stock Position (shares)", value=0, min_value=0)
        
        # Date settings
        st.subheader("Date Range")
        end_date = st.date_input("End Date", datetime.now())
        start_date = st.date_input(
            "Start Date",
            end_date - relativedelta(months=3),
            max_value=end_date
        )
        
        # Show reasoning checkbox
        show_reasoning = st.checkbox("Show Agent Reasoning", value=False)

    # Main area for analyst selection and results
    st.header("Select AI Analysts")
    
    # Analyst selection using checkboxes
    analysts = {
        "technical_analyst": "Technical Analyst",
        "fundamentals_analyst": "Fundamentals Analyst",
        "sentiment_analyst": "Sentiment Analyst",
        "valuation_analyst": "Valuation Analyst"
    }
    
    selected_analysts = []
    cols = st.columns(2)
    for i, (key, name) in enumerate(analysts.items()):
        if cols[i % 2].checkbox(name, value=True, key=key):
            selected_analysts.append(key)
    
    if not selected_analysts:
        st.warning("You must select at least one analyst.")
        return
    
    # Run button
    if st.button("Run Analysis", type="primary"):
        with st.spinner("Running hedge fund analysis..."):
            try:
                portfolio = {
                    "cash": initial_cash,
                    "stock": initial_stock
                }
                
                result = run_hedge_fund(
                    ticker=ticker,
                    start_date=start_date.strftime("%Y-%m-%d"),
                    end_date=end_date.strftime("%Y-%m-%d"),
                    portfolio=portfolio,
                    show_reasoning=show_reasoning,
                    selected_analysts=selected_analysts
                )
                
                # Display results
                st.header("Analysis Results")
                
                if result["decision"]:
                    decision = result["decision"]
                    st.subheader("Trading Decision", divider="rainbow")
                    
                    # Create columns for key metrics
                    col1, col2, col3 = st.columns(3)
                    
                    # Action with color coding
                    with col1:
                        action_color = {
                            "buy": "green",
                            "sell": "red",
                            "hold": "orange"
                        }.get(decision["action"].lower(), "blue")
                        
                        st.markdown(f"**Action:**")
                        st.markdown(f":{action_color}[{decision['action'].upper()}]")
                    
                    # Quantity
                    with col2:
                        st.markdown("**Quantity:**")
                        st.markdown(f"{decision['quantity']}")
                    
                    # Confidence with progress bar
                    with col3:
                        st.markdown("**Confidence:**")
                        st.progress(float(decision['confidence']))
                        st.markdown(f"{int(float(decision['confidence']) * 100)}%")
                    
                    # Reasoning in a box
                    st.markdown("**Reasoning:**")
                    st.info(decision["reasoning"])

                    # Display analyst signals if show_reasoning is True
                    if show_reasoning and result["analyst_signals"]:
                        st.subheader("Analyst Signals", divider="gray")
                        for analyst, signal in result["analyst_signals"].items():
                            with st.expander(f"## {analyst.replace('_', ' ').title()}"):
                                display_signal(signal)
                else:
                    st.error("No valid trading decision was generated.")
                    
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()